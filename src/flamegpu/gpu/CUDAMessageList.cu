 /**
 * @file CUDAMessageList.cpp
 * @authors
 * @date
 * @brief
 *
 * @see
 * @warning Not done, will not compile
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAMessageList.h"

#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/pop/AgentStateMemory.h"
#include "flamegpu/model/MessageDescription.h"

/**
* CUDAMessageList class
* @brief populates CUDA message map
*/
CUDAMessageList::CUDAMessageList(CUDAMessage& cuda_message) : message(cuda_message) {
    // allocate message lists
    allocateDeviceMessageList(d_list);
    allocateDeviceMessageList(d_swap_list);
    allocateDeviceMessageList(d_new_list);
}

/**
 * A destructor.
 * @brief Destroys the CUDAMessageList object
 */
CUDAMessageList::~CUDAMessageList() {
}

void CUDAMessageList::cleanupAllocatedData() {
    // clean up
    releaseDeviceMessageList(d_list);
    releaseDeviceMessageList(d_swap_list);
    releaseDeviceMessageList(d_new_list);
}

/**
* @brief Allocates Device  message list
* @param variable of type CUDAMessageMap struct type
* @return none
*/
void CUDAMessageList::allocateDeviceMessageList(CUDAMsgMap &memory_map) {
    // we use the  messages memory map to iterate the  message variables and do allocation within our GPU hash map
    const VariableMap &mem = message.getMessageDescription().getVariableMap();

    // for each variable allocate a device array and add to map
    for (const VariableMapPair& mm : mem) {
        // get the variable name
        std::string var_name = mm.first;

        // get the variable size from  message description
        size_t var_size = message.getMessageDescription().getMessageVariableSize(mm.first);

        // do the device allocation
        void * d_ptr;

#ifdef UNIFIED_GPU_MEMORY
        // unified memory allocation
        gpuErrchk(cudaMallocManaged(reinterpret_cast<void**>(&d_ptr), var_size *  message.getMaximumListSize()))
#else
        // non unified memory allocation
        gpuErrchk(cudaMalloc(reinterpret_cast<void**>(&d_ptr), var_size * message.getMaximumListSize()));
#endif

        // store the pointer in the map
        memory_map.insert(CUDAMsgMap::value_type(var_name, d_ptr));
    }
}

/**
* @brief Frees
* @param variable of type CUDAMsgMap struct type
* @return none
*/
void CUDAMessageList::releaseDeviceMessageList(CUDAMsgMap& memory_map) {
    // for each device pointer in the cuda memory map we need to free these
    for (const CUDAMsgMapPair& mm : memory_map) {
        // free the memory on the device
        gpuErrchk(cudaFree(mm.second));
    }
}

/**
* @brief
* @param variable of type CUDAMsgMap struct type
* @return none
*/
void CUDAMessageList::zeroDeviceMessageList(CUDAMsgMap& memory_map) {
    // for each device pointer in the cuda memory map set the values to 0
    for (const CUDAMsgMapPair& mm : memory_map) {
        // get the variable size from message description
        size_t var_size = message.getMessageDescription().getMessageVariableSize(mm.first);

        // set the memory to zero
        gpuErrchk(cudaMemset(mm.second, 0, var_size*message.getMaximumListSize()));
    }
}

void* CUDAMessageList::getMessageListVariablePointer(std::string variable_name) {
    CUDAMsgMap::iterator mm = d_list.find(variable_name);
    if (mm == d_list.end()) {
        // TODO: Error variable not found in message list
        return 0;
    }

    return mm->second;
}

void CUDAMessageList::zeroMessageData() {
    zeroDeviceMessageList(d_list);
    zeroDeviceMessageList(d_swap_list);
    zeroDeviceMessageList(d_new_list);
}

