#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_

#include <string>
#include <map>
#include <typeinfo>
#include <memory>
#include <vector>
#include <set>

#include "flamegpu/model/Variable.h"
#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/pop/AgentVector.h"
#include "flamegpu/pop/AgentInstance.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/util/type_decode.h"

namespace flamegpu {

class CAgentFunctionDescription;
class AgentFunctionDescription;
class AgentDescription;
namespace visualiser {
struct AgentVisData;
}

class CAgentDescription {
    friend struct AgentData;
    /**
     * AgentVector takes a clone of AgentData
     */
    friend AgentVector::AgentVector(const CAgentDescription& agent_desc, flamegpu::size_type);
    friend AgentInstance::AgentInstance(const CAgentDescription& agent_desc);
    friend bool AgentVector::matchesAgentType(const CAgentDescription& other) const;
    /**
     * Needs internal access to variables
     */
    friend struct visualiser::AgentVisData;

 public:
    /**
     * Constructor, creates an interface to the AgentData
     * @param data Data store of this agent's data
     */
    explicit CAgentDescription(std::shared_ptr<AgentData> data);
    explicit CAgentDescription(std::shared_ptr<const AgentData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same AgentData/ModelData
     */
    CAgentDescription(const CAgentDescription& other_agent) = default;
    CAgentDescription(CAgentDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same AgentData/ModelData
     */
    CAgentDescription& operator=(const CAgentDescription& other_agent) = default;
    CAgentDescription& operator=(CAgentDescription&& other_agent) = default;
    /**
     * Equality operator, checks whether AgentDescription hierarchies are functionally the same
     * @param rhs right hand side
     * @returns True when agents are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CAgentDescription& rhs) const;
    /**
     * Equality operator, checks whether AgentDescription hierarchies are functionally different
     * @param rhs right hand side
     * @returns True when agents are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CAgentDescription& rhs) const;

    /**
     * @return The agent's name
     */
    std::string getName() const;
    /**
     * @return The number of possible states agents of this type can enter
     */
    flamegpu::size_type getStatesCount() const;
    /**
     * @return The state which newly created agents of this type begin in
     */
    std::string getInitialState() const;
    /**
     * @param variable_name Name used to refer to the desired variable
     * @return The type of the named variable
     * @throws exception::InvalidAgentVar If a variable with the name does not exist within the agent
     */
    const std::type_index& getVariableType(const std::string& variable_name) const;
    /**
     * @param variable_name Name used to refer to the desired variable
     * @return The size of the named variable's type
     * @throws exception::InvalidAgentVar If a variable with the name does not exist within the agent
     */
    size_t getVariableSize(const std::string& variable_name) const;
    /**
     * @param variable_name Name used to refer to the desired variable
     * @return The number of elements in the name variable (1 if it isn't an array)
     * @throws exception::InvalidAgentVar If a variable with the name does not exist within the agent
     */
    flamegpu::size_type getVariableLength(const std::string& variable_name) const;
    /**
     * Get the total number of variables this agent has
     * @return The total number of variables within the agent
     * @note This count includes internal variables used to track things such as agent ID
     */
    flamegpu::size_type getVariablesCount() const;
    /**
     * Returns an immutable reference to the named agent function
     * @param function_name Name used to refer to the desired agent function
     * @return An immutable reference to the specified AgentFunctionDescription
     * @throws exception::InvalidAgentFunc If a function with the name does not exist within the agent
     * @see AgentDescription::Function(const std::string &) for the mutable version
     */
    CAgentFunctionDescription getFunction(const std::string& function_name) const;
    /**
     * Get the total number of functions this agent has
     * @return The total number of functions within the agent
     */
    flamegpu::size_type getFunctionsCount() const;
    /**
     * The total number of agent functions, within the ModelDescription hierarchy, which create new agents of this type
     * @return The total number of agent functions within the ModelDescription hierarchy which create new agents of this type
     * @see AgentDescription::isOutputOnDevice()
     */
    flamegpu::size_type getAgentOutputsCount() const;
    /**
     * @param state_name Name of the state to check
     * @return True when a state with the specified name exists within the agent
     */
    bool hasState(const std::string& state_name) const;
    /**
     * @param variable_name Name of the variable to check
     * @return True when a variable with the specified name exists within the agent
     */
    bool hasVariable(const std::string& variable_name) const;
    /**
     * @param function_name Name of the function to check
     * @return True when a function with the specified name exists within the agent
     */
    bool hasFunction(const std::string& function_name) const;
    /**
     * Check whether any agent functions output agents of this type
     * @return True if any agent functions, with the model hierarchy, create new agents of this type
     * @see AgentDescription::getAgentOutputsCount()
     */
    bool isOutputOnDevice() const;
    /**
     * Get the set of possible states for an agent of this type
     * @return An immutable reference to the set of states agents of this type can enter
     */
    const std::set<std::string>& getStates() const;

 protected:
     /**
      * The class which stores all of the agent's data.
      */
     std::shared_ptr<AgentData> agent;
};
/**
 * Within the model hierarchy, this class represents the definition of an agent for a FLAMEGPU model
 * This class is used to configure external elements of agents, such as variables and functions
 * @see AgentData The internal data store for this class
 * @see ModelDescription::newAgent(const std::string&) For creating instances of this class
 * @note To set an agent's id, the agent must be part of a model which has begun (id's are automatically assigned before initialisation functions and can not be manually set by users)
 */
class AgentDescription : public CAgentDescription {
    /**
     * ?
     */
    friend struct AgentFunctionData;
    /**
     * Accesses to check for unattached agent functions
     */
    friend class DependencyGraph;
    /**
     * AgentFunctionData accesses member variable model
     * to check that agent outputs are from the same model instance
     */
    friend class AgentFunctionDescription;

 public:
    /**
     * Constructor, creates an interface to the AgentData
     * @param data Data store of this agent's data
     */
    explicit AgentDescription(std::shared_ptr<AgentData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same AgentData/ModelData
     */
    AgentDescription(const AgentDescription& other_agent) = default;
    AgentDescription(AgentDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same AgentData/ModelData
     */
    AgentDescription& operator=(const AgentDescription& other_agent) = default;
    AgentDescription& operator=(AgentDescription&& other_agent) = default;

    /**
     * Adds a new state to the possible states this agent can enter
     * State's only exist as strings and have no additional configuration
     * @param state_name Name of the state
     * @throws exception::InvalidStateName If the agent already has a state with the same name
     */
    void newState(const std::string &state_name);
    /**
     * Sets the initial state which new agents begin in
     * @param initial_state Name of the desired state
     * @throws exception::InvalidStateName If the named state is not found within the agent
     */
    void setInitialState(const std::string &initial_state);

    /**
     * Adds a new variable array to the agent
     * @param variable_name Name of the variable array
     * @param default_value Default value of variable for new agents if unset, defaults to each element set 0
     * @tparam T Type of the agent variable, this must be an arithmetic type
     * @tparam N The length of the variable array (1 if not an array, must be greater than 0)
     * @throws exception::InvalidAgentVar If a variable already exists within the agent with the same name
     * @throws exception::InvalidAgentVar If N is <= 0
     */
    template<typename T, flamegpu::size_type N>
    void newVariable(const std::string &variable_name, const std::array<T, N> &default_value = {});
#ifndef SWIG
    /**
     * Adds a new variable to the agent
     * @param variable_name Name of the variable
     * @param default_value Default value of variable for new agents if unset, defaults to 0
     * @tparam T Type of the agent variable, this must be an arithmetic type
     * @throws exception::InvalidAgentVar If a variable already exists within the agent with the same name
     */
    template<typename T>
    void newVariable(const std::string& variable_name, T default_value = {});
#else
    /**
     * Adds a new variable to the agent
     * @param variable_name Name of the variable
     * @param default_value Default value of variable for new agents if unset, defaults to 0
     * @tparam T Type of the agent variable, this must be an arithmetic type
     * @throws exception::InvalidAgentVar If a variable already exists within the agent with the same name
     * @note Swig is unable to handle {} default param, however it's required for GLM support
     * Similarly, can't just provide 2 protoypes which overload, Python doesn't support that
     * Hence, easiest to require python users to init GLM types as arrays
     */
    template<typename T>
    void newVariable(const std::string& variable_name, T default_value = 0);
    /**
     * Adds a new variable array to the agent
     * @param variable_name Name of the variable array
     * @param length The length of the variable array (1 if not an array, must be greater than 0)
     * @param default_value Default value of variable for new agents if unset, defaults to each element set 0
     * @tparam T Type of the agent variable, this must be an arithmetic type
     * @throws exception::InvalidAgentVar If a variable already exists within the agent with the same name
     * @throws exception::InvalidAgentVar If length is <= 0
     */
    template<typename T>
    void newVariableArray(const std::string &variable_name, const flamegpu::size_type &length, const std::vector<T>&default_value = {});
#endif

    /**
     * Adds a new (device) function to the agent
     * @param function_name Name of the functions
     * @param a Instance of the agent function, used to implicitly set the template arg
     * Should be declared using FLAMEGPU_AGENT_FUNCTION notation
     * @tparam AgentFunction The agent function's containing struct type
     * @return A mutable reference to the new AgentFunctionDescription
     * @throws exception::InvalidAgentFunc If a variable already exists within the agent with the same name
     * @note The same agent function can be passed to the same agent twice
     */
    template<typename AgentFunction>
    AgentFunctionDescription newFunction(const std::string &function_name, AgentFunction a = AgentFunction());
    /**
     * Adds a new runtime (device) function to the agent from a string containing the source code
     * @param function_name Name of the functions
     * @param func_src representation of an agent function
     * Should be declared using FLAMEGPU_AGENT_FUNCTION notation
     * @return A mutable reference to the new AgentFunctionDescription
     * @throws exception::InvalidAgentFunc If a variable already exists within the agent with the same name
     * @note The same agent function can be passed to the same agent twice
     */
    AgentFunctionDescription newRTCFunction(const std::string& function_name, const std::string& func_src);
    /**
     * Adds a new runtime (device) function to the agent from a file containing the source code
     * @param function_name Name of the functions
     * @param file_path File path to file containing source for the agent function
     * Should be declared using FLAMEGPU_AGENT_FUNCTION notation
     * @return A mutable reference to the new AgentFunctionDescription
     * @throws exception::InvalidFilePath If file_path cannot be read
     * @throws exception::InvalidAgentFunc If a variable already exists within the agent with the same name
     * @note The same agent function can be passed to the same agent twice
     */
    AgentFunctionDescription newRTCFunctionFile(const std::string& function_name, const std::string& file_path);
    /**
     * Returns a mutable reference to the named agent function, which can be used to configure the function
     * @param function_name Name used to refer to the desired agent function
     * @return A mutable reference to the specified AgentFunctionDescription
     * @throws exception::InvalidAgentFunc If a functions with the name does not exist within the agent
     * @see AgentDescription::getFunction(const std::string &) for the immutable version
     */
    AgentFunctionDescription Function(const std::string &function_name);

    /**
     * Set how often this agent is sorted. Default value is 1.
     * @param sortPeriod Sort this agent every sortPeriod steps. A value of 0 means no sorting will take place
     */
    void setSortPeriod(const unsigned int sortPeriod);
};

/**
 * Template implementation
 */
template <typename T, flamegpu::size_type N>
void AgentDescription::newVariable(const std::string &variable_name, const std::array<T, N> &default_value) {
    if (!variable_name.empty() && variable_name[0] == '_') {
        THROW exception::ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
            "in AgentDescription::newVariable().");
    }
    std::string lower_variable_name = variable_name;
    for (auto& c : lower_variable_name)
        c = static_cast<char>(tolower(c));
    if (lower_variable_name == "name" || lower_variable_name == "state") {
        THROW exception::ReservedName("Agent variables cannot be named 'name' or 'state', these are reserved for backwards compatibility reasons, "
            "in AgentDescription::newVariable().");
    }
    if (lower_variable_name == "_auto_sort_bin_index") {
        THROW exception::ReservedName("The variable name '_auto_sort_bin_index' is reserved for internal usage, "
            "in AgentDescription::newVariable().");
    }
    // Array length 0 makes no sense
    static_assert(type_decode<T>::len_t * N > 0, "A variable cannot have 0 elements.");
    if (agent->variables.find(variable_name) == agent->variables.end()) {
        const std::array<typename type_decode<T>::type_t, type_decode<T>::len_t * N> *casted_default =
        reinterpret_cast<const std::array<typename type_decode<T>::type_t, type_decode<T>::len_t* N>*>(&default_value);
        agent->variables.emplace(variable_name, Variable(*casted_default));
        return;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') already contains variable '%s', "
        "in AgentDescription::newVariable().",
        agent->name.c_str(), variable_name.c_str());
}
template <typename T>
void AgentDescription::newVariable(const std::string &variable_name, const T default_value) {
    newVariable<T, 1>(variable_name, { default_value });
}
#ifdef SWIG
template<typename T>
void AgentDescription::newVariableArray(const std::string& variable_name, const flamegpu::size_type& length, const std::vector<T>& default_value) {
    if (!variable_name.empty() && variable_name[0] == '_') {
        THROW exception::ReservedName("Agent variable names cannot begin with '_', this is reserved for internal usage, "
            "in AgentDescription::newVariable().");
    }
    std::string lower_variable_name = variable_name;
    for (auto& c : lower_variable_name)
        c = static_cast<char>(tolower(c));
    if (lower_variable_name == "name" || lower_variable_name == "state") {
        THROW exception::ReservedName("Agent variables cannot be named 'name' or 'state', these are reserved for backwards compatibility reasons, "
            "in AgentDescription::newVariable().");
    }
    if (length == 0) {
        THROW exception::InvalidAgentVar("Agent variable arrays must have a length greater than 0."
            "in AgentDescription::newVariable().");
    }
    if (default_value.size() && default_value.size() != length) {
        THROW exception::InvalidAgentVar("Agent variable array length specified as %d, but default value provided with %llu elements, "
            "in AgentDescription::newVariable().",
            length, static_cast<unsigned int>(default_value.size()));
    }
    if (agent->variables.find(variable_name) == agent->variables.end()) {
        std::vector<typename type_decode<T>::type_t> temp(static_cast<size_t>(type_decode<T>::len_t * length));
        if (default_value.size()) {
            memcpy(temp.data(), default_value.data(), sizeof(typename type_decode<T>::type_t) * type_decode<T>::len_t * length);
        }
        agent->variables.emplace(variable_name, Variable(type_decode<T>::len_t* length, temp));
        return;
    }
    THROW exception::InvalidAgentVar("Agent ('%s') already contains variable '%s', "
        "in AgentDescription::newVariable().",
        agent->name.c_str(), variable_name.c_str());
}
#endif
// Found in "flamegpu/model/AgentFunctionDescription.h"
// template<typename AgentFunction>
// AgentFunctionDescription &AgentDescription::newFunction(const std::string &function_name, AgentFunction)

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_
