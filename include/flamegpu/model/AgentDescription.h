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

    /**
     * Only way to construct an AgentDescription
     */
    friend AgentDescription& ModelDescription::newAgent(const std::string &);

    /**
     * Constructors
     */
    AgentDescription(const std::string &agent_name);
    // Copy Construct
    AgentDescription(const AgentDescription &other_agent);
    // Move Construct
    AgentDescription(AgentDescription &&other_agent);
    // Copy Assign
    AgentDescription& operator=(const AgentDescription &other_agent);
    // Move Assign
    AgentDescription& operator=(AgentDescription &&other_agent);

    AgentDescription clone(const std::string &cloned_agent_name) const;

 public:
    /**
     * Typedefs
     */
    typedef unsigned int size_type;
    typedef std::map<const std::string, const std::type_info&> VariableMap;
    typedef std::map<const std::string, AgentFunctionDescription> FunctionMap;

    /**
     * Accessors
     */
    void newState(const std::string &state_name);
    void setInitialState(const std::string &initial_state);

    template <typename T>
    void newVariable(const std::string &variable_name);

    AgentFunctionDescription &newFunction(const std::string &function_name);
    AgentFunctionDescription &Function(const std::string &function_name);
    AgentFunctionDescription &cloneFunction(const AgentFunctionDescription &);

    /**
     * Const Accessors
     */
    std::string getName() const;
    
    const std::type_info& getVariableType(const std::string &variable_name) const;
    size_t getVariableSize(const std::string &variable_name) const;
    size_t getVariableSize(const std::string &variable_name) const;
    size_type getVariablesCount() const;
    const AgentFunctionDescription& getFunction(const std::string &function_name) const;

    const std::vector<std::string> &getStates() const;
    const VariableMap &getVariables() const;
    const FunctionMap& getFunctions() const;

    bool hasState(const std::string &state_name) const;
    bool hasVariable(const std::string &variable_name) const;
    bool hasFunction(const std::string &function_name) const;

 private:
    std::string name;

    std::vector<std::string> states;
    std::string initial_state = ModelDescription::DEFAULT_STATE;
    VariableMap variables;
    FunctionMap functions;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTDESCRIPTION_H_
