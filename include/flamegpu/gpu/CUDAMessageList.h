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

class CUDAMessageList {
 public:
    explicit CUDAMessageList(CUDAMessage& cuda_message);
    virtual ~CUDAMessageList();

    void cleanupAllocatedData();

    void* getReadMessageListVariablePointer(std::string variable_name);
    void* getWriteMessageListVariablePointer(std::string variable_name);

    void zeroMessageData();

    virtual void swap();
    virtual void scatter();

 protected:
    void allocateDeviceMessageList(CUDAMsgMap &Message_list);

    void releaseDeviceMessageList(CUDAMsgMap &Message_list);

    void zeroDeviceMessageList(CUDAMsgMap &Message_list);

 private:
    CUDAMsgMap d_list;
    CUDAMsgMap d_swap_list;  // may not need this later

    unsigned int current_list_size;  // ???

    const CUDAMessage& message;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAMESSAGELIST_H_
