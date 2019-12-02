#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_

#include <string>
#include <memory>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/AgentFunction.h"

class MessageDescription;

class AgentFunctionDescription {
    friend class AgentDescription;

    /**
     * Constructors
     */
    template<typename AgentFunction>
    AgentFunctionDescription(ModelDescription *const _model, const std::shared_ptr<AgentDescription> &parent_agent, const std::string &function_name, AgentFunction t = AgentFunction());
    // Copy Construct
    AgentFunctionDescription(const AgentFunctionDescription &other_function);
    // Move Construct
    AgentFunctionDescription(AgentFunctionDescription &&other_function);
    // Copy Assign
    AgentFunctionDescription& operator=(const AgentFunctionDescription &other_function);
    // Move Assign
    AgentFunctionDescription& operator=(AgentFunctionDescription &&other_function);

    AgentFunctionDescription clone(const std::string &cloned_function_name) const;

 public:
    /**
     * Accessors
     */
    void setInitialState(const std::string &initial_state);
    void setEndState(const std::string &end_state);
    void setMessageInput(const std::string &message_name);
    void setMessageInput(MessageDescription &message);
    void setMessageOutput(const std::string &message_name);
    void setMessageOutput(MessageDescription &message);
    void setAgentOutput(const std::string &agent_name);
    void setAgentOutput(AgentDescription &agent);
    void setAllowAgentDeath(const bool &has_death);
    
    MessageDescription &MessageInput();
    MessageDescription &MessageOutput();
    AgentDescription &AgentOutput();
    bool &AllowAgentDeath();
    
    /**
     * Const Accessors
     */
    std::string getName() const;
    std::string getInitialState() const;
    std::string getEndState() const;
    const MessageDescription &getMessageInput() const;
    const MessageDescription &getMessageOutput() const;
    const AgentDescription &getAgentOutput() const;    
    bool getHasAgentDeath() const;
    
    bool hasMessageInput() const;
    bool hasMessageOutput() const;
    bool hasAgentOutput() const;
    
 private:
    std::string name;
    
    AgentFunctionWrapper *const func;
    
    std::string initial_state = ModelDescription::DEFAULT_STATE;
    std::string end_state = ModelDescription::DEFAULT_STATE;
    
    MessageDescription *message_input = nullptr;
    MessageDescription *message_output = nullptr;
    
    AgentDescription *agent_output = nullptr;

    bool has_agent_death = false;

    std::weak_ptr<AgentDescription> parent;
    ModelDescription * const model;
};

/**
 * Template implementation
 */
template<typename AgentFunction>
AgentFunctionDescription::AgentFunctionDescription(ModelDescription *const _model, const std::shared_ptr<AgentDescription> &parent_agent, const std::string &function_name, AgentFunction)
    : name(function_name), func(&agent_function_wrapper<AgentFunction>), parent(parent_agent), model(_model) {
    // Force init initial_state, end_state to a valid state?
}

template<typename AgentFunction>
AgentFunctionDescription &AgentDescription::newFunction(const std::string &function_name, AgentFunction) {
    if (functions.find(function_name) == functions.end()) {
        auto rtn = std::shared_ptr<AgentFunctionDescription>(new AgentFunctionDescription(this->model, this->shared_from_this(), AgentFunction()));
        functions.emplace(function_name, rtn);
        return *rtn;
    }
    THROW InvalidAgentFunc("Agent ('%s') already contains function '%s', "
        "in AgentDescription::newFunction().",
        name.c_str(), function_name.c_str());
}
#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_
