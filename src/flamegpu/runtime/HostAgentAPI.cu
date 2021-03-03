#include "flamegpu/runtime/HostAgentAPI.h"

__global__ void initToThreadIndex(unsigned int *output, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        output[TID] = TID;
    }
}

void HostAgentAPI::fillTIDArray(unsigned int *buffer, const unsigned int &threadCount, const cudaStream_t &stream) {
    initToThreadIndex<<<(threadCount/512)+1, 512, 0, stream>>>(buffer, threadCount);
    gpuErrchkLaunch();
}

__global__ void sortBuffer_kernel(char *dest, char*src, unsigned int *position, size_t typeLen, unsigned int threadCount) {
    const unsigned int TID = blockIdx.x * blockDim.x + threadIdx.x;
    if (TID < threadCount) {
        memcpy(dest + TID * typeLen, src + position[TID] * typeLen, typeLen);
    }
}

void HostAgentAPI::sortBuffer(void *dest, void*src, unsigned int *position, const size_t &typeLen, const unsigned int &threadCount, const cudaStream_t &stream) {
    sortBuffer_kernel<<<(threadCount/512)+1, 512, 0, stream >>>(static_cast<char*>(dest), static_cast<char*>(src), position, typeLen, threadCount);
    gpuErrchkLaunch();
}
