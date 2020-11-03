#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_

#include <memory>
#include <map>
#include <utility>
#include <string>
#include <mutex>
#include <unordered_map>

// include sub classes
#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/model/AgentFunctionData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/sim/AgentInterface.h"

#ifdef _MSC_VER
#pragma warning(push, 2)
#include "jitify/jitify.hpp"
#pragma warning(pop)
#else
#include "jitify/jitify.hpp"
#endif

class CUDAScatter;
class CUDAFatAgent;
struct VarOffsetStruct;
/**
 * This is the regular CUDAAgent
 * It provides access to the device buffers representing the states of a particular agent
 * However it does not own these buffers, they are owned by it's parent CUDAFatAgent, as buffers are shared with all mapped agents too.
 */
class CUDAAgent : public AgentInterface {
    friend class AgentVis;

 public:
    /**
     *  map of agent function name to RTC function instance
     */
    typedef std::map<const std::string, std::unique_ptr<jitify::KernelInstantiation>> CUDARTCFuncMap;
    /**
     * Element type of CUDARTCFuncMap
     */
    typedef std::pair<const std::string, std::unique_ptr<jitify::KernelInstantiation>> CUDARTCFuncMapPair;
    /**
     * Normal constructor
     * @param description Agent description of the agent
     * @param instance_id Instance id of parent CUDASimulation of the agent
     */
    CUDAAgent(const AgentData& description, const unsigned int &instance_id);
    /**
     * Subagent form constructor, used when creating a CUDAAgent for a mapped agent within a submodel
     * @param description Agent description of the agent
     * @param instance_id Instance id of parent CUDASimulation of the agent
     * @param master_agent The (parent) agent which this is agent is mapped to
     * @param mapping Mapping definition for how this agent is connected with its master agent.
     */
    CUDAAgent(
        const AgentData &description,
        const unsigned int &instance_id,
        const std::unique_ptr<CUDAAgent> &master_agent,
        const std::shared_ptr<SubAgentData> &mapping);
    /** 
     * Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE
     * library so that can be accessed by name within a n agent function
     * @param func The function.
     * @note TODO: This could be improved by iterating the variable list within the state_list, rather than individually looking up vars (the two lists should have all the same vars)
     * @note This should probably be addressed when curve is updated to not use individual memcpys
     */
    void mapRuntimeVariables(const AgentFunctionData& func) const;
    /**
     * Uses the cuRVE runtime to unmap the variables used by the agent function to the cuRVE
     * library so that they are unavailable to be accessed by name within an agent function.
     * @param func The function.
     */
    void unmapRuntimeVariables(const AgentFunctionData& func) const;
    /**
     * Copies population data from the provided host object
     * To the device buffers held by this object (overwriting any existing agent data)
     * Also updates population size, clears disabled agents
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The index of the agent function within the current layer
     * @param population An AgentPopulation object with the same internal AgentData description, to provide the input data
     * @note Scatter is required for initialising submodel vars
     */
    void setPopulationData(const AgentPopulation& population, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Copies population data the device buffers held by this object
     * To the hosts object (overwriting any existing agent data)
     * @param population An AgentPopulation object with the same internal AgentData description, to receive the output data
     */
    void getPopulationData(AgentPopulation& population) const;
    /**
     * Returns the number of alive and active agents in the named state
     * @state The state to return information about
     */
    unsigned int getStateSize(const std::string &state) const;
    /**
     * Returns the number of alive and active agents in the named state
     * @state The state to return information about
     */
    unsigned int getStateAllocatedSize(const std::string &state) const;
    /**
     * Returns the Agent description which this CUDAAgent represents.
     */
    const AgentData &getAgentDescription() const;
    /**
     * Returns the device pointer to the buffer for the associated state and variable
     * @note This returns data_condition, such that the buffer does not include disabled agents
     */
    void *getStateVariablePtr(const std::string &state_name, const std::string &variable_name);
    /**
     * Processes agent death, this call is forwarded to the fat agent
     * All disabled agents are scattered to swap
     * Only alive agents with deathflag are scattered
     * @param func The agent function condition being processed
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The index of the agent function within the current layer
     * @see CUDAFatAgent::processDeath(const unsigned int &, const std::string &, const unsigned int &)
     */
    void processDeath(const AgentFunctionData& func, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Transitions all active agents from the source state to the destination state
     * @param _src The source state
     * @param _dest The destination state
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The index of the agent function within the current layer
     * @see CUDAFatAgent::transitionState(const unsigned int &, const std::string &, const std::string &, const unsigned int &)
     */
    void transitionState(const std::string &_src, const std::string &_dest, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Scatters agents based on their output of the agent function condition
     * Agents which failed the condition are scattered to the front and marked as disabled
     * Agents which pass the condition are scattered to after the disabled agents
     * @param func The agent function being processed
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The index of the agent function within the current layer
     * @see CUDAFatAgent::processFunctionCondition(const unsigned int &, const unsigned int &)
     * @note Named state must not already contain disabled agents
     * @note The disabled agents are re-enabled using clearFunctionCondition(const std::string &)
     */
    void processFunctionCondition(const AgentFunctionData& func, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Scatters agents from the provided device buffer, this is used for host agent creation
     * The device buffer must be packed according to the param offsets
     * @param state_name The state agents are scattered into
     * @param newSize The number of new agents
     * @param d_inBuff The device buffer containing the new agents
     * @param offsets This defines how the memory is laid out within d_inBuff
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId This is required for scan compaction arrays and async
     */
    void scatterHostCreation(const std::string &state_name, const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Sorts all agent variables according to the positions stored inside Message Output scan buffer
     * @param state_name The state agents are scattered into
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream in which the corresponding agent function has executed
     * @see HostAgentInstance::sort(const std::string &, HostAgentInstance::Order, int, int)
     */
    void scatterSort(const std::string &state_name, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Allocates a buffer for storing new agents into and
     * uses the cuRVE runtime to map variables for use with an agent function that has device agent birth
     * @param func_agent The Cuda agent which the agent function belongs to (required so that RTC function instances can be obtained)
     * @param func The agent function being processed
     * @param maxLen The maximum number of new agents (this will be the size of the agent state executing func)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId This is required for scan compaction arrays and async
     */
    void mapNewRuntimeVariables(const CUDAAgent& func_agent, const AgentFunctionData& func, const unsigned int &maxLen, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Uses the cuRVE runtime to unmap the variables used by agent birth and
     * releases the buffer that was storing the data
     * @param func The function.
     */
    void unmapNewRuntimeVariables(const AgentFunctionData& func);
    /**
     * Scatters agents from the currently assigned device agent birth buffer (see member variable newBuffs)
     * The device buffer must be packed in the same format as mapNewRuntimeVariables(const AgentFunctionData&, const unsigned int &, const unsigned int &)
     * @param func The agent function being processed
     * @param newSize The maximum number of new agents (this will be the size of the agent state executing func)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId This is required for scan compaction arrays and async
     */
    void scatterNew(const AgentFunctionData& func, const unsigned int &newSize, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Reenables all disabled agents within the named state
     * @param state The named state to enable all agents within
     */
    void clearFunctionCondition(const std::string &state);
    /**
     * Instatiates a RTC Agent function from agent function data description containing the agent function source.
     * If function_condition variable is true then this function will instantiate a function condition rather than an agent function
     * Uses Jitify to create an instantiation of the program. Any compilation errors in the user provided agent function will be reported here.
     * @param kernel_cache The JitCache to use (probably CUDASimulation::rtc_kernel_cache)
     * @throw InvalidAgentFunc thrown if the user supplied agent function has compilation errors
     */
    void addInstantitateRTCFunction(jitify::JitCache &kernel_cache, const AgentFunctionData& func, bool function_condition = false);
    /**
     * Returns the jitify kernel instantiation of the agent function.
     * Will throw an InvalidAgentFunc excpetion if the function name does not have a valid instantiation
     * @param function_name the name of the RTC agent function or the agent function name suffixed with condition (if it is a function condition)
     */
    const jitify::KernelInstantiation& getRTCInstantiation(const std::string &function_name) const;
    /**
     * Returns the CUDARTCFuncMap
     */
    const CUDARTCFuncMap& getRTCFunctions() const;
    /**
     * Resets the number of agents in any unmapped statelists to 0
     * They count as unmapped if they are not mapped to a master state, sub mappings will be reset
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId This is required for scan compaction arrays and async
     */
    void initUnmappedVars(CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Resets the number of agents in every statelist to 0
     */
    void cullAllStates();
    /**
     * Resets the number of agents in any unmapped statelists to 0
     * They count as unmapped if they are not mapped to a master state, sub mappings will be reset
     */
    void cullUnmappedStates();

 private:
    /**
     * Sums the size required for all variables
     */
    static size_t calcTotalVarSize(const AgentData &agent) {
        size_t rtn = 0;
        for (const auto v : agent.variables) {
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
     * The parent model's instance id
     * This is used for building the rtc header
     */
    const unsigned int &instance_id;
    /**
     * map between function_name (or function_name_condition) and the jitify instance
     */
    CUDARTCFuncMap rtc_func_map;
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
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_
