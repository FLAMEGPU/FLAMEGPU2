#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/MessageDescription.h"

/**
 * Constructors
 */
// Copy Construct
AgentFunctionDescription::AgentFunctionDescription(const AgentFunctionDescription &other_function)
    : func(other_function.func), model(other_function.model) {
    // TODO
}
// Move Construct
AgentFunctionDescription::AgentFunctionDescription(AgentFunctionDescription &&other_function)
    : func(other_function.func), model(other_function.model) {
    // TODO
}
// Copy Assign
AgentFunctionDescription& AgentFunctionDescription::operator=(const AgentFunctionDescription &other_function) {
    // TODO
    return *this;
}
// Move Assign
AgentFunctionDescription& AgentFunctionDescription::operator=(AgentFunctionDescription &&other_function) {
    // TODO
    return *this;
}

AgentFunctionDescription AgentFunctionDescription::clone(const std::string &cloned_function_name) const {
    // TODO
    return *this;
}


/**
 * Accessors
 */
void AgentFunctionDescription::setInitialState(const std::string &init_state) {
    if(auto p = parent.lock()) {
        if(p->hasState(initial_state)) {
            this->initial_state = init_state;
        } else {
            THROW InvalidStateName("Agent ('%s') does not contain state '%s', "
                "in AgentFunctionDescription::setInitialState()\n",
                p->getName().c_str(), init_state.c_str());
        }
    } else {
        THROW InvalidParent("Agent parent has expired, "
            "in AgentFunctionDescription::setInitialState()\n");
    }
}
void AgentFunctionDescription::setEndState(const std::string &exit_state) {
    if(auto p = parent.lock()) {
        if(p->hasState(initial_state)) {
            this->end_state = exit_state;
        } else {
            THROW InvalidStateName("Agent ('%s') does not contain state '%s', "
                "in AgentFunctionDescription::setEndState()\n",
                p->getName().c_str(), exit_state.c_str());
        }
    } else {
        THROW InvalidParent("Agent parent has expired, "
            "in AgentFunctionDescription::setEndState()\n");
    }
}
void AgentFunctionDescription::setMessageInput(const std::string &message_name) {
    if (model->hasMessage(message_name)) {
        this->message_input = &model->Message(message_name);
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageInput()\n",
            model->getName().c_str(), message_name.c_str());
    }
}
void AgentFunctionDescription::setMessageInput(MessageDescription &message) {
    if (model->hasMessage(message.getName())) {
        if(&model->getMessage(message.getName())==&message) {
            this->message_input = &message;
        } else {
            THROW InvalidMessage("Message '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setMessageInput()\n",
                message.getName().c_str(), model->getName().c_str());
        }
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageInput()\n",
            model->getName().c_str(), message.getName().c_str());
    }
}
void AgentFunctionDescription::setMessageOutput(const std::string &message_name) {
    if (model->hasMessage(message_name)) {
        this->message_output = &model->Message(message_name);
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageOutput()\n",
            model->getName().c_str(), message_name.c_str());
    }
}
void AgentFunctionDescription::setMessageOutput(MessageDescription &message) {
    if (model->hasMessage(message.getName())) {
        if (&model->getMessage(message.getName()) == &message) {
            this->message_output = &message;
        }
        else {
            THROW InvalidMessage("Message '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setMessageOutput()\n",
                message.getName().c_str(), model->getName().c_str());
        }
    }
    else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageOutput()\n",
            model->getName().c_str(), message.getName().c_str());
    }
}
void AgentFunctionDescription::setAgentOutput(const std::string &agent_name) {
    if (model->hasAgent(agent_name)) {
        this->agent_output = &model->Agent(agent_name);
    }
    else {
        THROW InvalidAgentName("Model ('%s') does not contain agent '%s', "
            "in AgentFunctionDescription::setAgentOutput()\n",
            model->getName().c_str(), agent_name.c_str());
    }
}
void AgentFunctionDescription::setAgentOutput(AgentDescription &agent) {
    if (model->hasAgent(agent.getName())) {
        if (&model->getAgent(agent.getName()) == &agent) {
            this->agent_output = &agent;
        }
        else {
            THROW InvalidMessage("Agent '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setAgentOutput()\n",
                agent.getName().c_str(), model->getName().c_str());
        }
    }
    else {
        THROW InvalidMessageName("Model ('%s') does not contain agent '%s', "
            "in AgentFunctionDescription::setAgentOutput()\n",
            model->getName().c_str(), agent.getName().c_str());
    }
}
void AgentFunctionDescription::setAllowAgentDeath(const bool &has_death) {
    has_agent_death = has_death;
}
    
MessageDescription &AgentFunctionDescription::MessageInput() {
    if (message_input)
        return *message_input;
    THROW OutOfBoundsException("Message input has not been set, "
        "in AgentFunctionDescription::MessageInput()\n");
}
MessageDescription &AgentFunctionDescription::MessageOutput() {
    if(message_output)
        return *message_output;
    THROW OutOfBoundsException("Message output has not been set, "
        "in AgentFunctionDescription::MessageOutput()\n");
}
AgentDescription &AgentFunctionDescription::AgentOutput() {
    if (agent_output)
        return *agent_output;
    THROW OutOfBoundsException("Agent output has not been set, "
        "in AgentFunctionDescription::AgentOutput()\n");
}
bool &AgentFunctionDescription::AllowAgentDeath() {
    return has_agent_death;
}
    
/**
 * Const Accessors
 */
std::string AgentFunctionDescription::getName() const {
    return name;
}
std::string AgentFunctionDescription::getInitialState() const {
    return initial_state;
}
std::string AgentFunctionDescription::getEndState() const {
    return end_state;
}
const MessageDescription &AgentFunctionDescription::getMessageInput() const {
    if (message_input)
        return *message_input;
    THROW OutOfBoundsException("Message input has not been set, "
        "in AgentFunctionDescription::getMessageInput()\n");
}
const MessageDescription &AgentFunctionDescription::getMessageOutput() const {
    if (message_output)
        return *message_output;
    THROW OutOfBoundsException("Message output has not been set, "
        "in AgentFunctionDescription::getMessageOutput()\n");
}
const AgentDescription &AgentFunctionDescription::getAgentOutput() const {
    if (agent_output)
        return *agent_output;
    THROW OutOfBoundsException("Agent output has not been set, "
        "in AgentFunctionDescription::getAgentOutput()\n");
}
bool AgentFunctionDescription::getHasAgentDeath() const {
    return has_agent_death;
}
    
bool AgentFunctionDescription::hasMessageInput() const {
    return message_input != nullptr;
}
bool AgentFunctionDescription::hasMessageOutput() const {
    return message_output != nullptr;
}
bool AgentFunctionDescription::hasAgentOutput() const {
    return agent_output != nullptr;
}