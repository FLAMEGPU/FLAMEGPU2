#include "flamegpu/runtime/flamegpu_host_agent_api.h"

__global__ void initToThreadIndex(unsigned int *output, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        output[TID] = TID;
    }
}

void HostAgentInstance::fillTIDArray(unsigned int *buffer, const unsigned int &threadCount) {
    initToThreadIndex<<<(threadCount/512)+1, 512 >>>(buffer, threadCount);
    gpuErrchkLaunch();
}

__global__ void sortBuffer_kernel(char *dest, char*src, unsigned int *position, size_t typeLen, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        memcpy(dest + TID * typeLen, src + position[TID] * typeLen, typeLen);
    }
}

void HostAgentInstance::sortBuffer(void *dest, void*src, unsigned int *position, const size_t &typeLen, const unsigned int &threadCount) {
    sortBuffer_kernel<<<(threadCount/512)+1, 512 >>>(static_cast<char*>(dest), static_cast<char*>(src), position, typeLen, threadCount);
    gpuErrchkLaunch();
}
