#ifndef INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_

#include <string>
#include <memory>

#include "flamegpu/model/ModelDescription.h"
#include "flamegpu/model/AgentDescription.h"
// include the agent function syntax form the runtime api
// #include "flamegpu/runtime/flamegpu_device_api.h"
typedef void FLAMEGPU_AGENT_FUNCTION_POINTER;

class MessageDescription;

class AgentFunctionDescription {
    friend class AgentDescription;

    /**
     * Constructors
     */
    AgentFunctionDescription(const std::string &function_name, FLAMEGPU_AGENT_FUNCTION_POINTER *p_func);
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
    void setMessageInput(const MessageDescription &message);
    void setMessageOutput(const std::string &message_name);
    void setMessageOutput(const MessageDescription &message);
    void setAgentOutput(const std::string &agent_name);
    void setAgentOutput(const AgentDescription &agent);
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
    
    FLAMEGPU_AGENT_FUNCTION_POINTER *const func;
    
    std::string initial_state = ModelDescription::DEFAULT_STATE;
    std::string end_state = ModelDescription::DEFAULT_STATE;
    
    std::shared_ptr<MessageDescription> message_input = nullptr;
    std::shared_ptr<MessageDescription> message_output = nullptr;
    
    std::shared_ptr<AgentDescription> agent_output = nullptr;
    
    bool has_agent_death;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_AGENTFUNCTIONDESCRIPTION_H_
