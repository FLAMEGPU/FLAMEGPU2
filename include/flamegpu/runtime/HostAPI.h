#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_

#include <cuda_runtime_api.h>
#include <string>
#include <utility>
#include <functional>
#include <unordered_map>
#include <vector>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/runtime/utility/HostRandom.cuh"
#include "flamegpu/runtime/utility/HostEnvironment.cuh"
#include "flamegpu/runtime/HostAPI_macros.h"
#include "flamegpu/runtime/HostNewAgentAPI.h"

class CUDAScatter;
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
    /**
     * Used internally for tracking what CUB has already calculated temp memory for
     * Pete claims this calc launches kernel, so best to avoid where possible
     */
    enum CUB_Operation {
        MIN,
        MAX,
        SUM,
        CUSTOM_REDUCE,
        HISTOGRAM_EVEN,
        SORT
    };
    // Can't put type_info in map, deleted constructors, so we use it's hash code
    // Histogram always has type int, so we use number of bins instead
    typedef std::pair<CUB_Operation, size_t> CUB_Config;
    /**
     * Need to define a hash_fn for tuple
     */
    struct key_hash : public std::unary_function<CUB_Config, std::size_t> {
        std::size_t operator()(const CUB_Config& k) const {
            return static_cast<size_t>(std::get<0>(k)) ^ std::get<1>(k);
        }
    };
    std::unordered_map<CUB_Config, unsigned int, key_hash> cub_largestAllocatedOp;
    bool tempStorageRequiresResize(const CUB_Config &cc, const unsigned int &items);
    void resizeTempStorage(const CUB_Config &cc, const unsigned int &items, const size_t &newSize);
    template<typename T>
    void resizeOutputSpace(const unsigned int &items = 1);
    CUDASimulation &agentModel;
    void *d_cub_temp;
    size_t d_cub_temp_size;
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

#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_H_
