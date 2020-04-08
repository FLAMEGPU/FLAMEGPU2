/**
* @file CUDAAgent.h
* @authors Paul
* @date 5 Mar 2014
* @brief
*
* @see
* @warning
*/

#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_

#include <memory>
#include <map>
#include <utility>
#include <string>

// include sub classes
#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/sim/AgentInterface.h"

// forward declare classes from other modules
struct AgentData;
struct AgentFunctionData;
class AgentPopulation;
class Curve;

typedef std::map<const std::string, std::unique_ptr<CUDAAgentStateList>> CUDAStateMap;  // map of state name to CUDAAgentStateList which allocates memory on the device
typedef std::pair<const std::string, std::unique_ptr<CUDAAgentStateList>> CUDAStateMapPair;

/** \brief CUDAAgent class is used as a container for storing the GPU data of all variables in all states
 * The CUDAAgent contains a hash index which maps a variable name to a unique index. Each CUDAAgentStateList
 * will use this hash index to map variable names to unique pointers in GPU memory space. This is required so
 * that at runtime a variable name can be related to a unique array of data on the device. It works like a traditional hashmap however the same hashing is used for all states that an agent can be in (as agents have the same variables regardless of state).
 */
class CUDAAgent : public AgentInterface {
    friend class AgentVis;
 public:
    explicit CUDAAgent(const AgentData& description);
    virtual ~CUDAAgent(void);

    const AgentData& getAgentDescription() const override;
    /**
     * Resizes the internal CUDAAgentStateLists
     * @param newSize The new minimum number of agents required, it may allocate more than requested here
     * @param streamId Internally passed by CUDAAgentModel to identify which scan array to resize
     * @note Currently it is not possible to reduce the allocated size
     * @note Currently it is not possible for internal state lists for the same agent to have different sizes 
     */
    void resize(const unsigned int &newSize, const unsigned int &streamId);
    /* Can be used to override the current population data without reallocating */
    void setPopulationData(const AgentPopulation& population);

    void getPopulationData(AgentPopulation& population);

    unsigned int getMaximumListSize() const;

    /** @brief Uses the cuRVE runtime to map the variables used by the agent function to the cuRVE library so that can be accessed by name within a n agent function
    */
    void mapRuntimeVariables(const AgentFunctionData& func, const std::string &state) const;

    /**
     * @brief    Uses the cuRVE runtime to unmap the variables used by the agent function to the cuRVE
     *             library so that they are unavailable to be accessed by name within an agent function.
     *
     * @param    func    The function.
     */

    void unmapRuntimeVariables(const AgentFunctionData& func) const;

    const std::unique_ptr<CUDAAgentStateList> &getAgentStateList(const std::string &state_name) const;

    void *getStateVariablePtr(const std::string &state_name, const std::string &variable_name) override;
    ModelData::size_type getStateSize(const std::string &state_name) const override;
    /**
     * Uses the agent scan flag for the named stream to sort agents and
     * reduce agent count so that agents which have reported death are removed
     *
     * @param func Agent function being actioned
     * @param streamId The scan_flag stream to use
     */
    void processDeath(const AgentFunctionData& func, const unsigned int &streamId);
    /**
     * Uses the agent scan flag for the named stream to sort agents and 
     * set the agent function condition state for the function's initial state
     * 
     * Agents are sorted so that those which failed the condition are moved to the start of the list
     * and those which failed the condition are move to the end of the list
     * @param func Agent function being actioned
     * @param streamId The scan_flag stream to use
     */
    void processFunctionCondition(const AgentFunctionData& func, const unsigned int &streamId);
    /**
     * Clears the agent function condition state values for the provided agent state
     * @param state The agent state to action
     */
    void clearFunctionConditionState(const std::string &state);
    /**
     * Transitions agents from src state to dest state, appending the existing destiation state list
     * @param src Source state
     * @dest dest Destination state
     * @param streamId The stream being used
     */
    void transitionState(const std::string &src, const std::string &dest, const unsigned int &streamId);

    /**
     * Maps variables within initial_state->new CUDAAgentStateList
     * They are mapped to "_agent_birth"_hash + func_hash + var_hash
     * @param func The agent function which will use the mapped variables
     * @param maxLen The maximum length set within curve
     */
    void mapNewRuntimeVariables(const AgentFunctionData& func, const unsigned int &maxLen) const;
    /**
    * Unmapaps variables within initial_state->new CUDAAgentStateList, freeing up space in CURVE hash table
    * They are mapped to "_agent_birth"_hash + func_hash + var_hash
    * @param func The agent function which used the mapped variables
    */
    void unmapNewRuntimeVariables(const AgentFunctionData& func) const;
    /**
     * Resize the new agent variable state list to the required length
     * This is useful because an agent function attached to a larget agent pop, might output agents of a smaller agent pop
     * Hence is should scale independently of other lists
     * @param func The agent function performing the agent birth
     * @param newSize The new minimum number of agents to be stored in the new list
     * @oaram streamId The stream index for handling any stream safe stuff
     */
    void resizeNew(const AgentFunctionData& func, const unsigned int &newSize, const unsigned int &streamId);
    /**
     * Scatters agents from d_list_new to d_list
     * @param state The CUDAAgentStateList to perform scatter on
     * @param newSize The max possible number of new agents
     * @param streamId Stream index for stream safe operations
     * @note This may resize death scan flag, which will lose it's data, hence always processDeath first
     */
    void scatterNew(const std::string state, const unsigned int &newSize, const unsigned int &streamId);

 protected:
    /** @brief    Zero all state variable data. */
    void zeroAllStateVariableData();

 private:
    const AgentData& agent_description;

    CUDAStateMap state_map;

    unsigned int max_list_size;  // The maximum length of the agent variable arrays based on the maximum population size passed to setPopulationData
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENT_H_
