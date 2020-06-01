#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_

#include <cassert>
#include <string>
#include <utility>
#include <functional>
#include <unordered_map>
#include <vector>

#include <cuda_runtime_api.h>

#include "flamegpu/gpu/CUDAErrorChecking.h"
#include "flamegpu/runtime/utility/HostRandom.cuh"
#include "flamegpu/runtime/utility/HostEnvironment.cuh"
#include "flamegpu/runtime/flamegpu_host_api_macros.h"
#include "flamegpu/runtime/flamegpu_host_new_agent_api.h"

class CUDAAgentModel;
class HostAgentInstance;

/**
 * @brief    A flame gpu api class for use by host functions only
 * This class should only be used by init/step/exit/exitcondition functions.
 */
class FLAMEGPU_HOST_API {
    /**
     * Requires internal access for resizeTempStorage()
     * @todo Could move this behaviour to a seperate singleton class 
     */
    friend class HostAgentInstance;
    // Typedefs repeated from CUDAAgentModel
    typedef std::vector<NewAgentStorage> AgentDataBuffer;
    typedef std::unordered_map<std::string, AgentDataBuffer> AgentDataBufferStateMap;
    typedef std::unordered_map<std::string, VarOffsetStruct> AgentOffsetMap;
    typedef std::unordered_map<std::string, AgentDataBufferStateMap> AgentDataMap;

 public:
    /**
     * Initailises pointers to 0
     * Stores reference of CUDAAgentModel
     */
     explicit FLAMEGPU_HOST_API(CUDAAgentModel&_agentModel,
         const AgentOffsetMap &agentOffsets,
         AgentDataMap &agentData);
    /**
     * Frees held device memory
     */
     ~FLAMEGPU_HOST_API();
    /**
     * Returns methods that work on all agents of a certain type currently in a given state
     */
    HostAgentInstance agent(const std::string &agent_name, const std::string &stateName = ModelData::DEFAULT_STATE);
    /**
     * Creates a new agent of the named type and returns an object for configuring it's member variables
     * The agent is created in their initial state as defined in model description hierarchy
     * @param agent_name Name of the agent type to be created
     * @throws InvalidAgentName If an agent with the provided name does not exist withint he model description hierarchy
     */
    FLAMEGPU_HOST_NEW_AGENT_API newAgent(const std::string &agent_name);
    /**
     * Creates a new agent of the named type and returns an object for configuring it's member variables
     * The agent is created in their initial state as defined in model description hierarchy
     * @param agent_name Name of the agent type to be created
     * @param state Name of the state the agent should be created in
     * @throws InvalidAgentName If an agent with the provided name does not exist withint he model description hierarchy
     * @throws InvalidStateName If state name does not apply to named agent
     */
    FLAMEGPU_HOST_NEW_AGENT_API newAgent(const std::string &agent_name, const std::string &state);
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
        HISTOGRAM_EVEN
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
    CUDAAgentModel &agentModel;
    void *d_cub_temp;
    size_t d_cub_temp_size;
    void *d_output_space;
    size_t d_output_space_size;
    /*
     * Owned by CUDAAgentModel, this provides memory offsets for agent variables
     * Used for host agent creationg
     */
    const AgentOffsetMap &agentOffsets;
    /*
     * Owned by CUDAAgentModel, this provides storage for new agents
     * Used for host agent creation, this should be emptied end of each step 
     * when new agents are copied to device.
     */
    AgentDataMap &agentData;
};

template<typename T>
void FLAMEGPU_HOST_API::resizeOutputSpace(const unsigned int &items) {
    if (sizeof(T) * items > d_output_space_size) {
        if (d_output_space_size) {
            gpuErrchk(cudaFree(d_output_space));
        }
        gpuErrchk(cudaMalloc(&d_output_space, sizeof(T) * items));
        d_output_space_size = sizeof(T) * items;
    }
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_
