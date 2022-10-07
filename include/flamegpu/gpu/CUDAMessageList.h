#ifndef INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_

#include <string>
#include <map>
#include <utility>

namespace flamegpu {

class CUDAScatter;
class CUDAMessage;


/**
 * This is the internal device memory handler for CUDAMessage
 * @todo This could just be merged with CUDAMessage
 */
class CUDAMessageList {
 public:
    /**
     * Map used to map a variable name to buffer
     */
    typedef std::map<std::string, void*> MessageMap;
    /**
     * Key Value pair of CUDAMessageMap
     */
    typedef std::pair<std::string, void*> MessageMapPair;
    /**
     * Initially allocates message lists based on cuda_message.getMaximumListSize()
     */
    explicit CUDAMessageList(CUDAMessage& cuda_message, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId);
    /**
     * Frees all message list memory
     */
    virtual ~CUDAMessageList();

    /**
     * Release all variable array memory in each list to 0
     */
    void cleanupAllocatedData();
    /**
     * Returns a pointer to the memory for the named variable in d_list
     * @param variable_name Name of the variable to get pointer to
     * @return void pointer to variable array in device memory
     */
    void* getReadMessageListVariablePointer(std::string variable_name);
    /**
     * Returns a pointer to the memory for the named variable in d_swap_list
     * @param variable_name Name of the variable to get pointer to
     * @return void pointer to variable array in device memory
     */
    void* getWriteMessageListVariablePointer(std::string variable_name);
    /**
     * Resize the internal message list buffers to the length of the parent CUDAMessage
     * Retain keep_len items from d_list during the resize (d_swap_list data is lost)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param keep_len If specified, number of items to retain through the resize
     * @throw If keep_len exceeds the new buffer length
     * @note This class has no way of knowing if keep_len exceeds the old buffer length size
     */
    void resize(CUDAScatter& scatter, cudaStream_t stream, unsigned int streamId = 0, unsigned int keep_len = 0);
    /**
     * Memset all variable arrays in each list to 0
     */
    void zeroMessageData(cudaStream_t stream);
    /**
     * Swap d_list and d_swap_list
     */
    virtual void swap();
    /**
     * Perform a compaction using d_message_scan_flag and d_message_position
     * @param newCount Number of new messages to be scattered
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param append If true scattered messages will append to the existing message list, otherwise truncate
     * @return Total number of messages now in list (includes old + new counts if appending)
     */
    virtual unsigned int scatter(unsigned int newCount, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId, bool append);
    /**
     * Copy all message data from d_swap_list to d_list
     * This ALWAYS performs and append to the existing message list count
     * Used by swap() when appending messagelists
     * @param newCount Number of new messages to be scattered
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @return Total number of messages now in list (includes old + new counts)
     */
    virtual unsigned int scatterAll(unsigned int newCount, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId);
    /**
     * @return Returns the map<variable_name, device_ptr> for reading message data
     */
    const MessageMap &getReadList() { return d_list; }
    /**
     * @return Returns the map<variable_name, device_ptr> for writing message data (aka swap buffers)
     */
    const MessageMap &getWriteList() { return d_swap_list; }

 protected:
    /**
     * Allocates device memory for the provided message list
     * @param memory_map Message list to perform operation on
     */
    void allocateDeviceMessageList(MessageMap &memory_map);
    /**
     * Frees device memory for the provided message list
     * @param memory_map Message list to perform operation on
     */
    void releaseDeviceMessageList(MessageMap &memory_map);
    /**
     * Zeros device memory for the provided message list
     * @param memory_map Message list to perform operation on
     * @param stream The CUDAStream to use for CUDA operations
     * @param skip_offset Number of items at the start of the list to not zero
     */
    void zeroDeviceMessageList_async(MessageMap &memory_map, cudaStream_t stream, unsigned int skip_offset = 0);

 private:
     /**
      * Message storage for reading
      */
    MessageMap d_list;
    /**
     * Message storage for writing
     */
    MessageMap d_swap_list;
    /**
     * Parent which this provides storage for
     */
    const CUDAMessage& message;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_
