#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_

#include <cuda_runtime.h>  // required for cudaStream_t. This doesn't require nvcc however, as no device code.
#include <string>
#include <utility>
#include <functional>
#include <unordered_map>
#include <vector>

#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/runtime/utility/HostRandom.cuh"
#include "flamegpu/runtime/utility/HostEnvironment.cuh"
#include "flamegpu/runtime/HostAPI_macros.h"
#include "flamegpu/runtime/HostNewAgentAPI.h"

namespace flamegpu {

class CUDAScatter;
class CUDASimulation;
class HostAgentAPI;
class CUDAMacroEnvironment;

/**
 * @brief    A flame gpu api class for use by host functions only
 * This class should only be used by init/step/exit/exitcondition functions.
 */
class HostAPI {
    /**
     * Requires internal access for resizeTempStorage()
     * @todo Could move this behaviour to a seperate singleton class 
     */
    friend class HostAgentAPI;
    /**
     * CUDAFatAgent::assignIDs() makes use of resizeTempStorage()
     */
    friend class CUDAFatAgent;

 public:
    // Typedefs repeated from CUDASimulation
    typedef std::vector<NewAgentStorage> AgentDataBuffer;
    typedef std::unordered_map<std::string, AgentDataBuffer> AgentDataBufferStateMap;
    typedef std::unordered_map<std::string, VarOffsetStruct> AgentOffsetMap;
    typedef std::unordered_map<std::string, AgentDataBufferStateMap> AgentDataMap;

    /**
     * Initailises pointers to 0
     * Stores reference of CUDASimulation
     */
     explicit HostAPI(CUDASimulation&_agentModel,
          RandomManager &rng,
          CUDAScatter &scatter,
          const AgentOffsetMap &agentOffsets,
          AgentDataMap &agentData,
          CUDAMacroEnvironment &macro_env,
          const unsigned int &streamId,
         cudaStream_t stream);
    /**
     * Frees held device memory
     */
     ~HostAPI();
    /**
     * Returns methods that work on all agents of a certain type currently in a given state
     */
    HostAgentAPI agent(const std::string &agent_name, const std::string &stateName = ModelData::DEFAULT_STATE);
    /**
     * Host API access to seeded random number generation
     */
    const HostRandom random;
    /**
     * Host API access to environmental properties
     */
    const HostEnvironment environment;

    /**
     * Access the current stepCount
     * @return the current step count, 0 indexed unsigned.
     */
    unsigned int getStepCounter() const;

 private:
    template<typename T>
    void resizeOutputSpace(const unsigned int &items = 1);
    CUDASimulation &agentModel;
    void *d_output_space;
    size_t d_output_space_size;
    /*
     * Owned by CUDASimulation, this provides memory offsets for agent variables
     * Used for host agent creationg
     */
    const AgentOffsetMap &agentOffsets;
    /*
     * Owned by CUDASimulation, this provides storage for new agents
     * Used for host agent creation, this should be emptied end of each step 
     * when new agents are copied to device.
     */
    AgentDataMap &agentData;
    /**
     * Cuda scatter singleton
     */
    CUDAScatter &scatter;
    /**
     * Stream index for stream-specific resources
     */
    const unsigned int streamId;
    /**
     * CUDA stream object for CUDA operations
     */
    cudaStream_t stream;
};

template<typename T>
void HostAPI::resizeOutputSpace(const unsigned int &items) {
    if (sizeof(T) * items > d_output_space_size) {
        if (d_output_space_size) {
            gpuErrchk(cudaFree(d_output_space));
        }
        gpuErrchk(cudaMalloc(&d_output_space, sizeof(T) * items));
        d_output_space_size = sizeof(T) * items;
    }
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_
