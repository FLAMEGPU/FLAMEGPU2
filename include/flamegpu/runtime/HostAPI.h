#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_

#include <cuda_runtime.h>  // required for cudaStream_t. This doesn't require nvcc however, as no device code.
#include <string>
#include <utility>
#include <functional>
#include <unordered_map>
#include <vector>
#include <memory>

#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/runtime/random/HostRandom.cuh"
#include "flamegpu/runtime/environment/HostEnvironment.cuh"
#include "flamegpu/runtime/HostAPI_macros.h"
#include "flamegpu/runtime/agent/HostNewAgentAPI.h"
#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {
namespace detail {
class CUDAFatAgent;
class CUDAScatter;
class CUDAMacroEnvironment;
}  // namespace detail
class CUDASimulation;
class HostAgentAPI;

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
    friend class detail::CUDAFatAgent;

 public:
    // Typedefs repeated from CUDASimulation
    typedef std::vector<NewAgentStorage> AgentDataBuffer;
    typedef std::unordered_map<std::string, AgentDataBuffer> AgentDataBufferStateMap;
    typedef std::unordered_map<std::string, VarOffsetStruct> AgentOffsetMap;
    typedef std::unordered_map<std::string, AgentDataBufferStateMap> AgentDataMap;
    typedef std::unordered_map<std::string, std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>> CUDADirectedGraphMap;

    /**
     * Initailises pointers to 0
     * Stores reference of CUDASimulation
     */
     explicit HostAPI(CUDASimulation&_agentModel,
        detail::RandomManager &rng,
        detail::CUDAScatter &scatter,
        const AgentOffsetMap &agentOffsets,
        AgentDataMap &agentData,
        const std::shared_ptr<detail::EnvironmentManager> &env,
        detail::CUDAMacroEnvironment &macro_env,
        CUDADirectedGraphMap &directed_graph_map,
        unsigned int streamId,
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

#ifdef FLAMEGPU_ADVANCED_API
    /**
     * Returns the cudaStream_t assigned to the current instance of HostAPI (and it's child objects)
     */
    cudaStream_t getCUDAStream() { return stream; }
#endif

 private:
    template<typename T>
    void resizeOutputSpace(unsigned int items = 1);
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
    detail::CUDAScatter &scatter;
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
void HostAPI::resizeOutputSpace(const unsigned int items) {
    if (sizeof(T) * items > d_output_space_size) {
        if (d_output_space_size) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(d_output_space));
        }
        gpuErrchk(cudaMalloc(&d_output_space, sizeof(T) * items));
        d_output_space_size = sizeof(T) * items;
    }
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_
