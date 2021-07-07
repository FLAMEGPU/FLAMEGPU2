#include "flamegpu/runtime/HostAgentAPI.h"
#include "flamegpu/pop/DeviceAgentVector_impl.h"

namespace flamegpu {

HostAgentAPI::~HostAgentAPI() {
    if (population) {
        population->syncChanges();
        population.reset();
    }
}

HostNewAgentAPI HostAgentAPI::newAgent() {
    // Create the agent in our backing data structure
    newAgentData.emplace_back(NewAgentStorage(agentOffsets, agent.nextID(1)));
    // Point the returned object to the created agent
    return HostNewAgentAPI(newAgentData.back());
}

unsigned HostAgentAPI::count() {
    if (population) {
        // If the user has a DeviceAgentVector out, use that instead
        return population->size();
    }
    return agent.getStateSize(stateName);
}

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

DeviceAgentVector HostAgentAPI::getPopulationData() {
    // Create and return a new AgentVector
    if (!population) {
        population = std::make_shared<DeviceAgentVector_impl>(static_cast<CUDAAgent&>(agent), stateName, agentOffsets, newAgentData, api.scatter, api.streamId, api.stream);
    }
    return *population;
}

}  // namespace flamegpu
