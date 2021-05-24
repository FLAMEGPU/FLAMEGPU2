#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTDATA_H_

#include <memory>
#include <unordered_map>
#include <set>
#include <string>

#include "flamegpu/model/Variable.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/defines.h"

class AgentDescription;
struct AgentFunctionData;

/**
 * This is the internal data store for AgentDescription
 * Users should only access that data stored within via an instance of AgentDescription
 */
struct AgentData : std::enable_shared_from_this<AgentData> {
    friend class ModelDescription;
    friend struct ModelData;
    friend class DependencyGraph;
    /**
     * Map of name:agent function definition
     * map<string, AgentFunctionData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<AgentFunctionData>> FunctionMap;

    /**
     * Holds all of the agent's function definitions
     */
    FunctionMap functions;
    /**
     * Holds all of the agent's variable definitions
     */
    VariableMap variables;
    /**
     * Holds all of the agent's possible states
     */
    std::set<std::string> states;
    /**
     * The initial state of newly created agents
     * @note must be found within member set states
     */
    std::string initial_state;
    /**
     * The number of functions that have agent output of this agent type
     * This value is modified by AgentFunctionDescription
     */
    unsigned int agent_outputs;
    /**
     * Description class which provides convenient accessors
     * This may be null if the instance has been cloned
     */
    std::shared_ptr<AgentDescription> description;
    /**
     * Name of the agent, used to refer to the agent in many functions
     */
    std::string name;
    /**
     * Internal value used to track whether the user has requested the default state as a state
     */
    bool keepDefaultState;
    /**
     * Check whether any agent functions within the ModelDescription hierarchy output agents of this type
     * @return true if this type of agent is created by any agent functions
     */
    bool isOutputOnDevice() const;
    /**
     * Equality operator, checks whether AgentData hierarchies are functionally the same
     * @param rhs Right hand side
     * @returns True when agents are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const AgentData &rhs) const;
    /**
     * Equality operator, checks whether AgentData hierarchies are functionally different
     * @param rhs Right hand side
     * @returns True when agents are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const AgentData &rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    AgentData(const AgentData &other) = delete;
    /**
     * Returns a constant copy of this agent's hierarchy
     * Does not copy description, sets it to nullptr instead
     * @return A shared ptr to a copy
     */
    std::shared_ptr<const AgentData> clone() const;

 protected:
    /**
     * Copy constructor
     * This is unsafe, should only be used internally, use clone() instead
     * This does not setup functions map 
     * @param model New parent model (we do not copy the other AgentData's parent model)
     * @param other Other AgentData to copy data from
     */
    AgentData(std::shared_ptr<const ModelData> model, const AgentData &other);
    /**
     * Normal constructor, only to be called by ModelDescription
     * @param model Parent model
     * @param agent_name Name of the agent
     */
    AgentData(std::shared_ptr<const ModelData> model, const std::string &agent_name);
};

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTDATA_H_
