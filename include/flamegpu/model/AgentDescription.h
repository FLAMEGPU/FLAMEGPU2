#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_

#include <string>
#include <map>
#include <typeinfo>
#include <memory>
#include <vector>

#include "flamegpu/model/ModelDescription.h"
class AgentFunctionDescription;
class MessageDescription;

class AgentDescription : public std::enable_shared_from_this<AgentDescription> {
    friend class MessageDescription; // I don't like this level of visibility, use common shared storage instead?

    /**
     * Only way to construct an AgentDescription
     */
    friend AgentDescription& ModelDescription::newAgent(const std::string &);

    /**
     * Constructors
     */
    /**
     * @note Can't force ModelDescription shared, so this is passed instead
     */
    AgentDescription();
    AgentDescription(ModelDescription * const parent, const std::string &agent_name);
    // Copy Construct
    AgentDescription(const AgentDescription &other_agent);
    // Move Construct
    AgentDescription(AgentDescription &&other_agent) noexcept;
    // Copy Assign
    AgentDescription& operator=(const AgentDescription &other_agent);
    // Move Assign
    AgentDescription& operator=(AgentDescription &&other_agent) noexcept;

    AgentDescription clone(const std::string &cloned_agent_name) const;

 public:
    /**
     * Typedefs
     */
    typedef unsigned int size_type;
    struct Variable
    {
        /**
         * Cannot explicitly specify template args of constructor, so we take redundant arg for implicit template
         */
        template<typename T>
        Variable(size_type _elements, T)
            : type(typeid(T)), type_size(sizeof(T)), elements(elements) { }
        const std::type_index type;
        const size_t type_size;
        const unsigned int elements;
    };
    typedef std::map<const std::string, Variable> VariableMap;
    typedef std::map<const std::string, std::shared_ptr<AgentFunctionDescription>> FunctionMap;

    /**
     * Accessors
     */
    void newState(const std::string &state_name);
    void setInitialState(const std::string &initial_state);

    template<typename AgentVariable, size_type N = 1>
    void newVariable(const std::string &variable_name);

    template<typename AgentFunction>
    AgentFunctionDescription &newFunction(const std::string &function_name, AgentFunction a = AgentFunction());
    AgentFunctionDescription &Function(const std::string &function_name);
    AgentFunctionDescription &cloneFunction(const AgentFunctionDescription &function);

    /**
     * Const Accessors
     */
    std::string getName() const;
    
    std::type_index getVariableType(const std::string &variable_name) const;
    size_t getVariableSize(const std::string &variable_name) const;
    size_type getVariableLength(const std::string &variable_name) const;
    size_type getVariablesCount() const;
    const AgentFunctionDescription& getFunction(const std::string &function_name) const;

    bool hasState(const std::string &state_name) const;
    bool hasVariable(const std::string &variable_name) const;
    bool hasFunction(const std::string &function_name) const;

    const std::set<std::string> &getStates() const;
    const VariableMap &getVariables() const;
    const FunctionMap &getFunctions() const;

private:
    /**
     * Member vars
     */
    std::string name;

    std::set<std::string> states;
    std::string initial_state = ModelDescription::DEFAULT_STATE;
    VariableMap variables;
    FunctionMap functions;
    ModelDescription * const model;
};

/**
 * Template implementation
 */
template <typename T, AgentDescription::size_type N>
void AgentDescription::newVariable(const std::string &variable_name) {
    if (variables.find(variable_name) == variables.end()) {
        variables.emplace(variable_name, Variable(N, T()));
        return;
    }
    THROW InvalidAgentVar("Agent ('%s') already contains variable '%s', "
        "in AgentDescription::newVariable().",
        name.c_str(), variable_name.c_str());
}

// Found in "flamegpu/model/AgentFunctionDescription.h"
// template<typename AgentFunction>
// AgentFunctionDescription &AgentDescription::newFunction(const std::string &function_name, AgentFunction)

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_
