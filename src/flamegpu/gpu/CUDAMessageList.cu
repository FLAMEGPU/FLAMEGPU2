#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/gpu/CUDAMessageList.h"

#include "flamegpu/gpu/CUDAMessage.h"
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/runtime/messaging/MessageBruteForce/MessageBruteForceHost.h"
#include "flamegpu/gpu/CUDAScatter.cuh"
#include "flamegpu/util/detail/cuda.cuh"

namespace flamegpu {

/**
* CUDAMessageList class
* @brief populates CUDA message map
*/
CUDAMessageList::CUDAMessageList(CUDAMessage& cuda_message, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId)
    : message(cuda_message) {
    // allocate message lists
    allocateDeviceMessageList(d_list);
    allocateDeviceMessageList(d_swap_list);
    zeroDeviceMessageList_async(d_list, stream);
    zeroDeviceMessageList_async(d_swap_list, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

/**
 * A destructor.
 * @brief Destroys the CUDAMessageList object
 */
CUDAMessageList::~CUDAMessageList() {
    cleanupAllocatedData();
}

void CUDAMessageList::cleanupAllocatedData() {
    // clean up
    releaseDeviceMessageList(d_list);
    releaseDeviceMessageList(d_swap_list);
}

void CUDAMessageList::allocateDeviceMessageList(CUDAMessageMap &memory_map) {
    // we use the  messages memory map to iterate the  message variables and do allocation within our GPU hash map
    const auto &mem = message.getMessageData().variables;

    // for each variable allocate a device array and add to map
    for (const auto &mm : mem) {
        // get the variable name
        std::string var_name = mm.first;

        // get the variable size from  message description
        size_t var_size = mm.second.type_size * mm.second.elements;

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
        memory_map.insert(CUDAMessageMap::value_type(var_name, d_ptr));
    }
}
void CUDAMessageList::resize(CUDAScatter& scatter, cudaStream_t stream, unsigned int streamId, unsigned int keep_len) {
    // Release d_swap_list, we don't retain this data
    releaseDeviceMessageList(d_swap_list);
    // Allocate the new d_list
    CUDAMessageMap d_list_old;
    std::swap(d_list, d_list_old);
    allocateDeviceMessageList(d_list);
    if (keep_len && keep_len <= message.getMessageCount()) {
        // Copy data from d_list_old to d_list
        // Note, if keep_len exceeds length of d_swap_list_old, this will crash
        scatter.scatterAll(streamId,
            stream,
            message.getMessageData().variables,
            d_list_old, d_list,
            keep_len,
            0);
    }
    // Release d_list_old
    releaseDeviceMessageList(d_list_old);
    // Allocate the new d_swap_list
    allocateDeviceMessageList(d_swap_list);
    // Zero any new buffers with undefined data
    zeroDeviceMessageList_async(d_list, stream, keep_len);
    zeroDeviceMessageList_async(d_swap_list, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}

void CUDAMessageList::releaseDeviceMessageList(CUDAMessageMap& memory_map) {
    // for each device pointer in the cuda memory map we need to free these
    for (const auto &mm : memory_map) {
        // free the memory on the device
        gpuErrchk(flamegpu::util::detail::cuda::cudaFree(mm.second));
    }
    memory_map.clear();
}

void CUDAMessageList::zeroDeviceMessageList_async(CUDAMessageMap& memory_map, cudaStream_t stream, unsigned int skip_offset) {
    if (skip_offset >= message.getMaximumListSize())
        return;
    // for each device pointer in the cuda memory map set the values to 0
    for (const auto &mm : memory_map) {
        // get the variable size from message description
        const auto var = message.getMessageData().variables.at(mm.first);
        const size_t var_size = var.type_size * var.elements;

        // set the memory to zero
        gpuErrchk(cudaMemsetAsync(static_cast<char*>(mm.second) + (var_size * skip_offset), 0, var_size * (message.getMaximumListSize() - skip_offset), stream));
    }
}

void* CUDAMessageList::getReadMessageListVariablePointer(std::string variable_name) {
    CUDAMessageMap::iterator mm = d_list.find(variable_name);
    if (mm == d_list.end()) {
        THROW exception::InvalidMessageVar("Variable '%s' was not found in message '%s', "
          "in CUDAMessageList::getReadMessageListVariablePointer()",
          variable_name.c_str(), message.getMessageData().name.c_str());
    }

    return mm->second;
}
void* CUDAMessageList::getWriteMessageListVariablePointer(std::string variable_name) {
    CUDAMessageMap::iterator mm = d_swap_list.find(variable_name);
    if (mm == d_swap_list.end()) {
        THROW exception::InvalidMessageVar("Variable '%s' was not found in message '%s', "
            "in CUDAMessageList::getWriteMessageListVariablePointer()",
            variable_name.c_str(), message.getMessageData().name.c_str());
    }

    return mm->second;
}

void CUDAMessageList::zeroMessageData(cudaStream_t stream) {
    zeroDeviceMessageList_async(d_list, stream);
    zeroDeviceMessageList_async(d_swap_list, stream);
    gpuErrchk(cudaStreamSynchronize(stream));
}


void CUDAMessageList::swap() {
    std::swap(d_list, d_swap_list);
}

unsigned int CUDAMessageList::scatter(unsigned int newCount, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId, bool append) {
    if (append) {
        unsigned int oldCount = message.getMessageCount();
        return oldCount + scatter.scatter(streamId,
            stream,
            CUDAScatter::Type::MESSAGE_OUTPUT,
            message.getMessageData().variables,
            d_swap_list, d_list,
            newCount,
            oldCount);
    } else {
        return scatter.scatter(streamId,
            stream,
            CUDAScatter::Type::MESSAGE_OUTPUT,
            message.getMessageData().variables,
            d_swap_list, d_list,
            newCount,
            0);
    }
}
unsigned int CUDAMessageList::scatterAll(unsigned int newCount, CUDAScatter &scatter, cudaStream_t stream, unsigned int streamId) {
    unsigned int oldCount = message.getMessageCount();
    return oldCount + scatter.scatterAll(streamId,
        stream,
        message.getMessageData().variables,
        d_swap_list, d_list,
        newCount,
        oldCount);
}

}  // namespace flamegpu
