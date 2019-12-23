 /**
 * @file CUDAMessageList.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#ifndef INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_

#include <string>
#include <map>
#include <utility>

class CUDAMessage;
// class AgentStateMemory;

// #define UNIFIED_GPU_MEMORY

typedef std::map <std::string, void*> CUDAMsgMap;
typedef std::pair <std::string, void*> CUDAMsgMapPair;

/**
 * This is the internal device memory handler for CUDAMessage
 * @todo This could just be merged with CUDAMessage
 */
class CUDAMessageList {
 public:
     /**
      * Initially allocates message lists based on cuda_message.getMaximumListSize()
      */
    explicit CUDAMessageList(CUDAMessage& cuda_message);
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
     * Perform a compaction using d_scan_flag and d_position
     */
    virtual void scatter();

 protected:
     /**
      * Allocates device memory for the provided message list
      * @param Message_list Message list to perform operation on
      */
     void allocateDeviceMessageList(CUDAMsgMap &Message_list);
     /**
      * Frees device memory for the provided message list
      * @param Message_list Message list to perform operation on
      */
     void releaseDeviceMessageList(CUDAMsgMap &Message_list);
     /**
      * Zeros device memory for the provided message list
      * @param Message_list Message list to perform operation on
      */
     void zeroDeviceMessageList(CUDAMsgMap &Message_list);

 private:
     /**
      * Message storage for reading
      */
    CUDAMsgMap d_list;
    /**
     * Message storage for writing
     */
    CUDAMsgMap d_swap_list;
    /**
     * Parent which this provides storage for
     */
    const CUDAMessage& message;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_
