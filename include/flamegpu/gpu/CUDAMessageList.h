 /**
 * @file CUDAMessageList.h
 * @author
 * @date
 * @brief
 *
 * \todo longer description
 */

#ifndef CUDAMESSAGELIST_H_
#define CUDAMESSAGELIST_H_

#include <string>
#include <memory>
#include <vector>
#include <map>

class CUDAMessage;
// class AgentStateMemory;

// #define UNIFIED_GPU_MEMORY

typedef std::map <std::string, void*> CUDAMsgMap;
typedef std::pair <std::string, void*> CUDAMsgMapPair;

class CUDAMessageList {
public:
    CUDAMessageList(CUDAMessage& cuda_message);
    virtual ~CUDAMessageList();

    void cleanupAllocatedData();

    void* getMessageListVariablePointer(std::string variable_name);

    void zeroMessageData();

protected:


    void allocateDeviceMessageList(CUDAMsgMap &Message_list);

    void releaseDeviceMessageList(CUDAMsgMap &Message_list);

    void zeroDeviceMessageList(CUDAMsgMap &Message_list);


private:
    CUDAMsgMap d_list;
    CUDAMsgMap d_swap_list; // may not need this later
    CUDAMsgMap d_new_list;

    unsigned int current_list_size; // ???

    const CUDAMessage& message;
};

#endif
