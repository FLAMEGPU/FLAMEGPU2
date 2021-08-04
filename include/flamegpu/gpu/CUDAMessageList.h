#ifndef INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_

#include <string>
#include <map>
#include <utility>

namespace flamegpu {

class CUDAScatter;
class CUDAMessage;

/**
 * Map used to map a variable name to buffer
 */
typedef std::map <std::string, void*> CUDAMessageMap;
/**
 * Key Value pair of CUDAMessageMap
 */
typedef std::pair <std::string, void*> CUDAMessageMapPair;

/**
 * This is the internal device memory handler for CUDAMessage
 * @todo This could just be merged with CUDAMessage
 */
class CUDAMessageList {
 public:
     /**
      * Initially allocates message lists based on cuda_message.getMaximumListSize()
      */
    explicit CUDAMessageList(CUDAMessage& cuda_message, CUDAScatter &scatter, const unsigned int &streamId);
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
     * Memset all variable arrays in each list to 0
     */
    void zeroMessageData();
    /**
     * Swap d_list and d_swap_list
     */
    virtual void swap();
    /**
     * Perform a compaction using d_message_scan_flag and d_message_position
     * @param newCount Number of new messages to be scattered
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param append If true scattered messages will append to the existing message list, otherwise truncate
     * @return Total number of messages now in list (includes old + new counts if appending)
     */
    virtual unsigned int scatter(const unsigned int &newCount, CUDAScatter &scatter, const unsigned int &streamId, const bool &append);
    /**
     * Copy all message data from d_swap_list to d_list
     * This ALWAYS performs and append to the existing message list count
     * Used by swap() when appending messagelists
     * @param newCount Number of new messages to be scattered
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @return Total number of messages now in list (includes old + new counts)
     */
    virtual unsigned int scatterAll(const unsigned int &newCount, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * @return Returns the map<variable_name, device_ptr> for reading message data
     */
    const CUDAMessageMap &getReadList() { return d_list; }
    /**
     * @return Returns the map<variable_name, device_ptr> for writing message data (aka swap buffers)
     */
    const CUDAMessageMap &getWriteList() { return d_swap_list; }

 protected:
     /**
      * Allocates device memory for the provided message list
      * @param memory_map Message list to perform operation on
      */
     void allocateDeviceMessageList(CUDAMessageMap &memory_map);
     /**
      * Frees device memory for the provided message list
      * @param memory_map Message list to perform operation on
      */
     void releaseDeviceMessageList(CUDAMessageMap &memory_map);
     /**
      * Zeros device memory for the provided message list
      * @param memory_map Message list to perform operation on
      */
     void zeroDeviceMessageList(CUDAMessageMap &memory_map);

 private:
     /**
      * Message storage for reading
      */
    CUDAMessageMap d_list;
    /**
     * Message storage for writing
     */
    CUDAMessageMap d_swap_list;
    /**
     * Parent which this provides storage for
     */
    const CUDAMessage& message;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_
