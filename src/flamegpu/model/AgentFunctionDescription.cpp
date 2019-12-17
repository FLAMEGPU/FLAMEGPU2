#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/MessageDescription.h"

/**
 * Constructors
 */

AgentFunctionDescription::AgentFunctionDescription(ModelData *const _model, AgentFunctionData *const description)
    : model(_model)
    , function(description) { }

bool AgentFunctionDescription::operator==(const AgentFunctionDescription& rhs) const {
    return *this->function == *rhs.function;  // Compare content is functionally the same
}
bool AgentFunctionDescription::operator!=(const AgentFunctionDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Accessors
 */
void AgentFunctionDescription::setInitialState(const std::string &init_state) {
    if (auto p = function->parent.lock()) {
        if (p->description->hasState(init_state)) {
            this->function->initial_state = init_state;
        } else {
            THROW InvalidStateName("Agent ('%s') does not contain state '%s', "
                "in AgentFunctionDescription::setInitialState()\n",
                p->name.c_str(), init_state.c_str());
        }
    } else {
        THROW InvalidParent("Agent parent has expired, "
            "in AgentFunctionDescription::setInitialState()\n");
    }
}
void AgentFunctionDescription::setEndState(const std::string &exit_state) {
    if (auto p = function->parent.lock()) {
        if (p->description->hasState(exit_state)) {
            this->function->end_state = exit_state;
        } else {
            THROW InvalidStateName("Agent ('%s') does not contain state '%s', "
                "in AgentFunctionDescription::setEndState()\n",
                p->name.c_str(), exit_state.c_str());
        }
    } else {
        THROW InvalidParent("Agent parent has expired, "
            "in AgentFunctionDescription::setEndState()\n");
    }
}
void AgentFunctionDescription::setMessageInput(const std::string &message_name) {
    if (auto other = function->message_output.lock()) {
        if (message_name == other->name) {
            THROW InvalidMessageName("Message '%s' is already bound as message output in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageInput()\n",
                message_name.c_str(), function->name.c_str());
        }
    }
    auto a = model->messages.find(message_name);
    if (a != model->messages.end()) {
        this->function->message_input = a->second;
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageInput()\n",
            model->name.c_str(), message_name.c_str());
    }
}
void AgentFunctionDescription::setMessageInput(MessageDescription &message) {
    if (message.model != function->description->model) {
        THROW DifferentModel("Attempted to use agent description from a different model, "
            "in AgentFunctionDescription::setAgentOutput()\n");
    }
    if (auto other = function->message_output.lock()) {
        if (message.getName() == other->name) {
            THROW InvalidMessageName("Message '%s' is already bound as message output in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageInput()\n",
                message.getName().c_str(), function->name.c_str());
        }
    }
    auto a = model->messages.find(message.getName());
    if (a != model->messages.end()) {
        if (a->second->description.get() == &message) {
            this->function->message_input = a->second;
        } else {
            THROW InvalidMessage("Message '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setMessageInput()\n",
                message.getName().c_str(), model->name.c_str());
        }
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageInput()\n",
            model->name.c_str(), message.getName().c_str());
    }
}
void AgentFunctionDescription::setMessageOutput(const std::string &message_name) {
    if (auto other = function->message_input.lock()) {
        if (message_name == other->name) {
            THROW InvalidMessageName("Message '%s' is already bound as message input in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageOutput()\n",
                message_name.c_str(), function->name.c_str());
        }
    }
    auto a = model->messages.find(message_name);
    if (a != model->messages.end()) {
        this->function->message_output = a->second;
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageOutput()\n",
            model->name.c_str(), message_name.c_str());
    }
}
void AgentFunctionDescription::setMessageOutput(MessageDescription &message) {
    if (message.model != function->description->model) {
        THROW DifferentModel("Attempted to use agent description from a different model, "
            "in AgentFunctionDescription::setAgentOutput()\n");
    }
    if (auto other = function->message_input.lock()) {
        if (message.getName() == other->name) {
            THROW InvalidMessageName("Message '%s' is already bound as message input in agent function %s, "
                "the same message cannot be input and output by the same function, "
                "in AgentFunctionDescription::setMessageOutput()\n",
                message.getName().c_str(), function->name.c_str());
        }
    }
    auto a = model->messages.find(message.getName());
    if (a != model->messages.end()) {
        if (a->second->description.get() == &message) {
            this->function->message_output = a->second;
        } else {
            THROW InvalidMessage("Message '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setMessageOutput()\n",
                message.getName().c_str(), model->name.c_str());
        }
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain message '%s', "
            "in AgentFunctionDescription::setMessageOutput()\n",
            model->name.c_str(), message.getName().c_str());
    }
}
void AgentFunctionDescription::setMessageOutputOptional(const bool &output_is_optional) {
    this->function->message_output_optional = output_is_optional;
}
void AgentFunctionDescription::setAgentOutput(const std::string &agent_name) {
    // Clear old value
    if (auto b = this->function->agent_output.lock()) {
        b->agent_outputs--;
    }
    // Set new
    auto a = model->agents.find(agent_name);
    if (a != model->agents.end()) {
        this->function->agent_output = a->second;
        a->second->agent_outputs++;  // Mark inside agent that we are using it as an output
    } else {
        THROW InvalidAgentName("Model ('%s') does not contain agent '%s', "
            "in AgentFunctionDescription::setAgentOutput()\n",
            model->name.c_str(), agent_name.c_str());
    }
}
void AgentFunctionDescription::setAgentOutput(AgentDescription &agent) {
    if (agent.model != function->description->model) {
        THROW DifferentModel("Attempted to use agent description from a different model, "
            "in AgentFunctionDescription::setAgentOutput()\n");
    }
    // Clear old value
    if (auto b = this->function->agent_output.lock()) {
        b->agent_outputs--;
    }
    // Set new
    auto a = model->agents.find(agent.getName());
    if (a != model->agents.end()) {
        if (a->second->description.get() == &agent) {
            this->function->agent_output = a->second;
            a->second->agent_outputs++;  // Mark inside agent that we are using it as an output
        } else {
            THROW InvalidMessage("Agent '%s' is not from Model '%s', "
                "in AgentFunctionDescription::setAgentOutput()\n",
                agent.getName().c_str(), model->name.c_str());
        }
    } else {
        THROW InvalidMessageName("Model ('%s') does not contain agent '%s', "
            "in AgentFunctionDescription::setAgentOutput()\n",
            model->name.c_str(), agent.getName().c_str());
    }
}
void AgentFunctionDescription::setAllowAgentDeath(const bool &has_death) {
    function->has_agent_death = has_death;
}

MessageDescription &AgentFunctionDescription::MessageInput() {
    if (auto m = function->message_input.lock())
        return *m->description;
    THROW OutOfBoundsException("Message input has not been set, "
        "in AgentFunctionDescription::MessageInput()\n");
}
MessageDescription &AgentFunctionDescription::MessageOutput() {
    if (auto m = function->message_output.lock())
        return *m->description;
    THROW OutOfBoundsException("Message output has not been set, "
        "in AgentFunctionDescription::MessageOutput()\n");
}
AgentDescription &AgentFunctionDescription::AgentOutput() {
    if (auto a = function->agent_output.lock())
        return *a->description;
    THROW OutOfBoundsException("Agent output has not been set, "
        "in AgentFunctionDescription::AgentOutput()\n");
}
bool &AgentFunctionDescription::MessageOutputOptional() {
    return function->message_output_optional;
}
bool &AgentFunctionDescription::AllowAgentDeath() {
    return function->has_agent_death;
}

/**
 * Const Accessors
 */
std::string AgentFunctionDescription::getName() const {
    return function->name;
}
std::string AgentFunctionDescription::getInitialState() const {
    return function->initial_state;
}
std::string AgentFunctionDescription::getEndState() const {
    return function->end_state;
}
const MessageDescription &AgentFunctionDescription::getMessageInput() const {
    if (auto m = function->message_input.lock())
        return *m->description;
    THROW OutOfBoundsException("Message input has not been set, "
        "in AgentFunctionDescription::getMessageInput()\n");
}
const MessageDescription &AgentFunctionDescription::getMessageOutput() const {
    if (auto m = function->message_output.lock())
        return *m->description;
    THROW OutOfBoundsException("Message output has not been set, "
        "in AgentFunctionDescription::getMessageOutput()\n");
}
bool AgentFunctionDescription::getMessageOutputOptional() const {
    return this->function->message_output_optional;
}
const AgentDescription &AgentFunctionDescription::getAgentOutput() const {
    if (auto a = function->agent_output.lock())
        return *a->description;
    THROW OutOfBoundsException("Agent output has not been set, "
        "in AgentFunctionDescription::getAgentOutput()\n");
}
bool AgentFunctionDescription::getAllowAgentDeath() const {
    return function->has_agent_death;
}

bool AgentFunctionDescription::hasMessageInput() const {
    return function->message_input.lock() != nullptr;
}
bool AgentFunctionDescription::hasMessageOutput() const {
    return function->message_output.lock() != nullptr;
}
bool AgentFunctionDescription::hasAgentOutput() const {
    return function->agent_output.lock() != nullptr;
}
AgentFunctionWrapper *AgentFunctionDescription::getFunctionPtr() const {
    return function->func;
}
