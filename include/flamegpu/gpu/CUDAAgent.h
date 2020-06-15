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

class CUDAFatAgent;
struct VarOffsetStruct;
/**
 * This is the regular CUDAAgent
 */
class CUDAAgent : public AgentInterface {
 public:
    /**
     *  map of state name to CUDAAgentStateList which allocates memory on the device
     */
    typedef std::map<const std::string, std::unique_ptr<jitify::KernelInstantiation>> CUDARTCFuncMap;
    typedef std::pair<const std::string, std::unique_ptr<jitify::KernelInstantiation>> CUDARTCFuncMapPair;
    /**
     * Normal constructor
     * @param description Agent description of the agent
     * @param _cuda_model Parent CUDAAgentModel of the agent
     */
    CUDAAgent(const AgentData& description, const CUDAAgentModel &_cuda_model);
    /**
    * Subagent form constructor
    * @param description Agent description of the agent
    * @param _cuda_model Parent CUDAAgentModel of the agent
    */
    CUDAAgent(
        const AgentData &description,
        const CUDAAgentModel &_cuda_model,
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
     *
     * @param func The function.
     */
    void unmapRuntimeVariables(const AgentFunctionData& func) const;

    /**
     * Copies population data from the provided host object
     * To the device buffers held by this object (overwriting any existing agent data)
     * Also updates population size, clears disabled agents
     * @param population An AgentPopulation object with the same internal AgentData description, to provide the input data
     */
    void setPopulationData(const AgentPopulation& population);
    /**
     * Copies population data the device buffers held by this object
     * To the hosts object (overwriting any existing agent data)
     * @param population An AgentPopulation object with the same internal AgentData description, to receive the output data
     */
    void getPopulationData(AgentPopulation& population) const;
    /**
     * Returns the number of alive and active agents in the named state
     */
    unsigned int getStateSize(const std::string &state) const;
    /**
     * Returns the number of alive and active agents in the named state
     */
    unsigned int getStateAllocatedSize(const std::string &state) const;
    const AgentData &getAgentDescription() const;
    void *getStateVariablePtr(const std::string &state_name, const std::string &variable_name);
    /**
     * Processes agent death, this call is forwarded to the fat agent
     * All disabled agents are scattered to swap
     * Only alive agents with deathflag are scattered
     * @param func The agent function condition being processed
     * @param streamId The index of the agent function within the current layer
     * @see CUDAFatAgent::processDeath(const unsigned int &, const std::string &, const unsigned int &)
     */
    void processDeath(const AgentFunctionData& func, const unsigned int &streamId);
    /**
     * Transitions all active agents from the source state to the destination state
     * @param _src The source state
     * @param _dest The destination state
     * @param streamId The index of the agent function within the current layer
     * @see CUDAFatAgent::transitionState(const unsigned int &, const std::string &, const std::string &, const unsigned int &)
     */
    void transitionState(const std::string &_src, const std::string &_dest, const unsigned int &streamId);
    /**
     * Scatters agents based on their output of the agent function condition
     * Agents which failed the condition are scattered to the front and marked as disabled
     * Agents which pass the condition are scatterd to after the disabled agents
     * @param func The agent function condition being processed
     * @param streamId The index of the agent function within the current layer
     * @see CUDAFatAgent::processFunctionCondition(const unsigned int &, const unsigned int &)
     * @note Named state must not already contain disabled agents
     * @note The disabled agents are re-enabled using clearFunctionCondition(const std::string &)
     */
    void processFunctionCondition(const AgentFunctionData& func, const unsigned int &streamId);
    void scatterHostCreation(const std::string &state_name, const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets);
    void mapNewRuntimeVariables(const AgentFunctionData& func, const unsigned int &maxLen, const unsigned int &streamId);
    void unmapNewRuntimeVariables(const AgentFunctionData& func);
    void scatterNew(const AgentFunctionData& func, const unsigned int &newSize, const unsigned int &streamId);

    /**
     * Reenables all disabled agents within the named state
     * @param state The named state to enable all agents within
     */
    void clearFunctionCondition(const std::string &state);

    /**
     * Instatiates a RTC Agent function from agent function data description containing the agent function source.
     * If function_condition variable is true then this function will instantiate a function condition rather than an agent function
     * Uses Jitify to create an instantiation of the program. Any compilation errors in the user provided agent function will be reported here.
     * @throw InvalidAgentFunc thrown if the user supplied agent function has compilation errors
     */
    void addInstantitateRTCFunction(const AgentFunctionData& func, bool function_condition = false);
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

 private:
    static size_t calcTotalVarSize(const AgentData &agent) {
        size_t rtn = 0;
        for (const auto v : agent.variables) {
            rtn += v.second.type_size * v.second.elements;
        }
        return rtn;
    }
    unsigned int getFatIndex() const { return fat_index; }
    std::shared_ptr<CUDAFatAgent> getFatAgent() { return fat_agent; }
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
     * The agent description of this agent
     */
    const AgentData &agent_description;
    /**
     * The parent model
     */
    const CUDAAgentModel &cuda_model;
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
