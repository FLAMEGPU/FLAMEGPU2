#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_

#include <memory>
#include <map>
#include <utility>
#include <string>
#include <mutex>
#include <unordered_map>
#include <list>

// include sub classes
#include "flamegpu/util/detail/JitifyCache.h"
#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/model/AgentFunctionData.cuh"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/runtime/detail/curve/curve_rtc.cuh"
#include "flamegpu/sim/AgentInterface.h"

namespace flamegpu {

class CUDAScatter;
class CUDAFatAgent;
struct VarOffsetStruct;
class HostAPI;
/**
 * This is the regular CUDAAgent
 * It provides access to the device buffers representing the states of a particular agent
 * However it does not own these buffers, they are owned by it's parent CUDAFatAgent, as buffers are shared with all mapped agents too.
 */
class CUDAAgent : public AgentInterface {
#ifdef VISUALISATION
    friend class visualiser::AgentVis;
#endif  // VISUALISATION

 public:
    /**
     *  map of agent function name to RTC function instance
     */
     typedef std::map<const std::string, std::unique_ptr<jitify::experimental::KernelInstantiation>> CUDARTCFuncMap;
     typedef std::map<const std::string, std::unique_ptr<detail::curve::CurveRTCHost>> CUDARTCHeaderMap;
    /**
     * Element type of CUDARTCFuncMap
     */
    typedef std::pair<const std::string, std::unique_ptr<jitify::experimental::KernelInstantiation>> CUDARTCFuncMapPair;
    /**
     * Normal constructor
     * @param description Agent description of the agent
     * @param _cudaSimulation Parent CUDASimulation of the agent
     */
    CUDAAgent(const AgentData& description, const CUDASimulation &_cudaSimulation);
    /**
     * Subagent form constructor, used when creating a CUDAAgent for a mapped agent within a submodel
     * @param description Agent description of the agent
     * @param _cudaSimulation Parent CUDASimulation of the agent
     * @param master_agent The (parent) agent which this is agent is mapped to
     * @param mapping Mapping definition for how this agent is connected with its master agent.
     */
    CUDAAgent(
        const AgentData &description,
        const CUDASimulation &_cudaSimulation,
        const std::unique_ptr<CUDAAgent> &master_agent,
        const std::shared_ptr<SubAgentData> &mapping);
    /** 
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE
     * library so that can be accessed by name within a n agent function
     * @param func The function.
     * @param instance_id The CUDASimulation instance_id of the parent instance. This is added to the hash, to differentiate instances
     * @note TODO: This could be improved by iterating the variable list within the state_list, rather than individually looking up vars (the two lists should have all the same vars)
     * @note This should probably be addressed when curve is updated to not use individual memcpys
     */
    void mapRuntimeVariables(const AgentFunctionData& func, unsigned int instance_id) const;
    /**
     * Copies population data from the provided host object
     * To the device buffers held by this object (overwriting any existing agent data)
     * Also updates population size, clears disabled agents
     * @param population An AgentVector object with the same internal AgentData description, to provide the input data
     * @param state_name The agent state to add the agents to
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @note Scatter is required for initialising submodel vars
     */
    void setPopulationData(const AgentVector& population, const std::string &state_name, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Copies population data the device buffers held by this object
     * To the hosts object (overwriting any existing agent data)
     * @param population An AgentVector object with the same internal AgentData description, to receive the output data
     * @param state_name The agent state to get the agents from
     */
    void getPopulationData(AgentVector& population, const std::string& state_name) const;
    /**
     * Returns the number of alive and active agents in the named state
     * @param state The state to return information about
     */
    unsigned int getStateSize(const std::string &state) const override;
    /**
     * Returns the number of alive and active agents in the named state
     * @param state The state to return information about
     */
    unsigned int getStateAllocatedSize(const std::string &state) const;
    /**
     * Returns the Agent description which this CUDAAgent represents.
     */
    CAgentDescription getAgentDescription() const override;
    /**
     * Returns the device pointer to the buffer for the associated state and variable
     * @note This returns data_condition, such that the buffer does not include disabled agents
     */
    void *getStateVariablePtr(const std::string &state_name, const std::string &variable_name) override;
    /**
     * Processes agent death, this call is forwarded to the fat agent
     * All disabled agents are scattered to swap
     * Only alive agents with deathflag are scattered
     * @param func The agent function condition being processed
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @see CUDAFatAgent::processDeath(unsigned int, const std::string &, unsigned int)
     */
    void processDeath(const AgentFunctionData& func, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Transitions all active agents from the source state to the destination state
     * @param _src The source state
     * @param _dest The destination state
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @see CUDAFatAgent::transitionState(unsigned int, const std::string &, const std::string &, unsigned int)
     */
    void transitionState(const std::string &_src, const std::string &_dest, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Scatters agents based on their output of the agent function condition
     * Agents which failed the condition are scattered to the front and marked as disabled
     * Agents which pass the condition are scattered to after the disabled agents
     * @param func The agent function being processed
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @see CUDAFatAgent::processFunctionCondition(unsigned int, unsigned int)
     * @note Named state must not already contain disabled agents
     * @note The disabled agents are re-enabled using clearFunctionCondition(const std::string &)
     */
    void processFunctionCondition(const AgentFunctionData& func, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Scatters agents from the provided device buffer, this is used for host agent creation
     * The device buffer must be packed according to the param offsets
     * @param state_name The state agents are scattered into
     * @param newSize The number of new agents
     * @param d_inBuff The device buffer containing the new agents
     * @param offsets This defines how the memory is laid out within d_inBuff
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void scatterHostCreation(const std::string &state_name, unsigned int newSize, char *const d_inBuff, const VarOffsetStruct &offsets, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Sorts all agent variables according to the positions stored inside Message Output scan buffer
     * @param state_name The state agents are scattered into
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @see HostAgentAPI::sort(const std::string &, HostAgentAPI::Order, int, int)
     */
    void scatterSort_async(const std::string &state_name, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Allocates a buffer for storing new agents into and
     * uses the cuRVE runtime to map variables for use with an agent function that has device agent birth
     * @param func_agent The Cuda agent which the agent function belongs to (required so that RTC function instances can be obtained)
     * @param func The agent function being processed
     * @param maxLen The maximum number of new agents (this will be the size of the agent state executing func)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param instance_id The CUDASimulation instance_id of the parent instance. This is added to the hash, to differentiate instances
     * @param stream The CUDA stream used to execute the initialisation of the new agent databuffers
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @note This method is async, the stream used it not synchronised
     */
    void mapNewRuntimeVariables_async(const CUDAAgent& func_agent, const AgentFunctionData& func, unsigned int maxLen, CUDAScatter &scatter, unsigned int instance_id, cudaStream_t stream, unsigned int streamId);
    /**
     * Releases the buffer that was storing new age data
     * @param func The function.
     */
    void releaseNewBuffer(const AgentFunctionData& func);
    /**
     * Scatters agents from the currently assigned device agent birth buffer (see member variable newBuffs)
     * The device buffer must be packed in the same format as mapNewRuntimeVariables(const AgentFunctionData&, unsigned int, unsigned int)
     * @param func The agent function being processed
     * @param newSize The maximum number of new agents (this will be the size of the agent state executing func)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void scatterNew(const AgentFunctionData& func, unsigned int newSize, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Reenables all disabled agents within the named state
     * @param state The named state to enable all agents within
     */
    void clearFunctionCondition(const std::string &state);
    /**
     * Instantiates a RTC Agent function (or agent function condition) from agent function data description containing the source.
     * 
     * Uses Jitify to create an instantiation of the program. Any compilation errors in the user provided agent function will be reported here.
     * @param func The Agent function data structure containing the src for the function
     * @param env Object containing environment properties for the simulation instance
     * @param macro_env Object containing environment macro properties for the simulation instance
     * @param function_condition If true then this function will instantiate a function condition rather than an agent function
     * @throw exception::InvalidAgentFunc thrown if the user supplied agent function has compilation errors
     */
    void addInstantitateRTCFunction(const AgentFunctionData& func, const std::shared_ptr<EnvironmentManager>& env, const CUDAMacroEnvironment& macro_env, bool function_condition = false);
    /**
     * Instantiates the curve instance for an (non-RTC) Agent function (or agent function condition) from agent function data description containing the source.
     *
     * This performs CUDA allocations, so must be performed after CUDA device selection/context creation
     * @param func The Agent function data structure containing the details of the function
     * @param env Object containing environment properties for the simulation instance
     * @param macro_env Object containing environment macro properties for the simulation instance
     * @param function_condition If true then this function will instantiate a function condition rather than an agent function
     */
    void addInstantitateFunction(const AgentFunctionData& func, const std::shared_ptr<EnvironmentManager>& env, const CUDAMacroEnvironment& macro_env, bool function_condition = false);
    /**
     * Returns the jitify kernel instantiation of the agent function.
     * Will throw an exception::InvalidAgentFunc excpetion if the function name does not have a valid instantiation
     * @param function_name the name of the RTC agent function or the agent function name suffixed with condition (if it is a function condition)
     */
    const jitify::experimental::KernelInstantiation& getRTCInstantiation(const std::string &function_name) const;
    detail::curve::CurveRTCHost &getRTCHeader(const std::string &function_name) const;
    /**
     * Returns the host interface for managing the curve instance for the named agent function
     * @param function_name The name of the agent's agent function, to return the curve instance for
     */
    detail::curve::HostCurve &getCurve(const std::string& function_name) const;
    /**
     * Returns the CUDARTCFuncMap
     */
    const CUDARTCFuncMap& getRTCFunctions() const;
    /**
     * Resets the number of agents in any unmapped statelists to 0
     * They count as unmapped if they are not mapped to a master state, sub mappings will be reset
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void initUnmappedVars(CUDAScatter& scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Initialises any agent variables within the CUDAFatAgentStateList of state which are not present in the agent-state's CUDAAgentStateList
     * @param state Affected state
     * @param count Number of variables to init
     * @param offset Offset into the buffer of agents to init
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void initExcludedVars(const std::string& state, unsigned int count, unsigned int offset, CUDAScatter& scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Resets the number of agents in every statelist to 0
     */
    void cullAllStates();
    /**
     * Resets the number of agents in any unmapped statelists to 0
     * They count as unmapped if they are not mapped to a master state, sub mappings will be reset
     */
    void cullUnmappedStates();
    /**
     * Resize the state underlying buffers
     * This is triggered on the CUDAFatAgentStateList, so unmapped agent variables are also resized
     * @param state The minimum number of agents that must be representable
     * @param minSize The minimum number of agents that must be representable
     * @param retainData If true existing buffer data is retained
     * @param stream The stream to be used for memcpys if data is retained
     */
    void resizeState(const std::string &state, unsigned int minSize, bool retainData, cudaStream_t stream);
    /**
     * Updates the number of alive agents, does not affect disabled agents or change agent data
     * @param state The state to affect
     * @param newSize Number of active agents
     * @throw exception::InvalidMemoryCapacity If the new number of disabled + active agents would exceed currently allocated buffer capacity
     */
    void setStateAgentCount(const std::string& state, unsigned int newSize);
    /**
     * Returns a list of variable buffers attached to bound agents, not available in this agent
     * @param state The state affected state
     * @note This access is only intended for DeviceAgentVector's correctly handling of subagents
     */
    std::list<std::shared_ptr<VariableBuffer>> getUnboundVariableBuffers(const std::string& state);
    /**
     * Returns the next free agent id, and increments the ID tracker by the specified count
     * @param count Number that will be added to the return value on next call to this function
     * @return An ID that can be assigned to an agent that wil be stored within this CUDAAgent's CUDAFatAgent
     */
    id_t nextID(unsigned int count = 1) override;
    /**
     * Returns a device pointer to the value returns by nextID(0)
     * If the device value is changed, then the internal ID counter must be updated via CUDAAgent::scatterNew()
     */
    id_t* getDeviceNextID();
    /**
     * Assigns IDs to any agents who's ID has the value ID_NOT_SET
     * @param hostapi HostAPI object, this is used to provide cub temp storage
     * @param scatter Scatter instance and scan arrays to be used
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     */
    void assignIDs(HostAPI &hostapi, CUDAScatter& scatter, cudaStream_t stream, unsigned int streamId);
    /**
     * Used to allow HostAgentAPI to store a persistent DeviceAgentVector
     * @param state_name Agent state to affect
     * @param d_vec The DeviceAgentVector to be stored
     */
    void setPopulationVec(const std::string& state_name, const std::shared_ptr<DeviceAgentVector_impl>& d_vec);
    /**
     * Used to allow HostAgentAPI to retrieve a stored DeviceAgentVector
     * @param state_name Agent state to affect
     */
    std::shared_ptr<DeviceAgentVector_impl> getPopulationVec(const std::string& state_name);
    /**
     * Used to allow HostAgentAPI to clear the stored DeviceAgentVector
     * Any changes will be synchronised first
     */
    void resetPopulationVecs();

 private:
    /**
     * Validates all IDs for contained agents, if any share an ID (which is not ID_NOT_SET) an exception is thrown
     * @throws exception::AgentIDCollision If the contained agent populations contain multiple agents with the same ID
     */
    void validateIDCollisions(cudaStream_t stream) const;
    /**
     * Sums the size required for all variables
     */
    static size_t calcTotalVarSize(const AgentData &agent) {
        size_t rtn = 0;
        for (const auto &v : agent.variables) {
            rtn += v.second.type_size * v.second.elements;
        }
        return rtn;
    }
    /**
     * The fat index of this agent within its CUDAFatAgent
     */
    unsigned int getFatIndex() const { return fat_index; }
    /**
     * Returns this CUDAAgent's fat agent
     */
    std::shared_ptr<CUDAFatAgent> getFatAgent() { return fat_agent; }
    /**
     * The agent description of this agent
     */
    const AgentData &agent_description;
    /**
     * Map of all states held by this agent
     */
    std::unordered_map<std::string, std::shared_ptr<CUDAAgentStateList>> state_map;
    /**
     * The 'fat agent' representation of this agent
     * This acts as a CUDAAgent with access to all variables for all mapped agent-states
     */
    std::shared_ptr<CUDAFatAgent> fat_agent;
    /**
     * Multiple mapped agents may share names/definitions, so instead they are assigned indices
     */
    const unsigned int fat_index;
    /**
     * The parent model
     */
    const CUDASimulation &cudaSimulation;
    /**
     * map between function_name (or function_name_condition) and the jitify instance
     */
    CUDARTCFuncMap rtc_func_map;
    /**
     * map between function name (or function_name_condition) and the rtc header
     * This allows access to the header data cache, for updating curve
     */
    CUDARTCHeaderMap rtc_header_map;
    /**
     * map between function name (or function_name_condition) and the curve interface
     * This allows access for updating curve
     */
    std::unordered_map<std::string, std::unique_ptr<detail::curve::HostCurve>> curve_map;
    /**
     * Used when allocated new buffers
     */
    const size_t TOTAL_AGENT_VARIABLE_SIZE;
    /**
     * Holds currently held new buffs set by mapNewRuntimeVariables, cleared by unmapNewRuntimeVariables
     * key: initial state name, val: allocated buffer
     */
    std::unordered_map<std::string, void*> newBuffs;
    /**
     * Mutex for writing to newBuffs
     */
    std::mutex newBuffsMutex;
    /**
     * Nullptr until getPopulationData() is called, after which it holds the return value
     */
    std::map<std::string, std::shared_ptr<DeviceAgentVector_impl>> population_dvec;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_
