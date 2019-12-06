#include "flamegpu/model/ModelDescription.h"

#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/exception/FGPUException.h"

/**
* Constructors
*/
ModelDescription::ModelDescription(const std::string &model_name)
    : model(new ModelData(model_name)) { }
// Copy Construct
ModelDescription::ModelDescription(const ModelDescription &other_model) {
    // TODO
}
// Move Construct
ModelDescription::ModelDescription(ModelDescription &&other_model) {
    // TODO
}
// Copy Assign
ModelDescription& ModelDescription::operator=(const ModelDescription &other_model) {
    // TODO
    return *this;
}
// Move Assign
ModelDescription& ModelDescription::operator=(ModelDescription &&other_model) {
    // TODO
    return *this;
}

/**
* Accessors
*/
AgentDescription& ModelDescription::newAgent(const std::string &agent_name) {
    if(!hasAgent(agent_name)) {
        auto rtn = std::shared_ptr<AgentDescription>(new AgentDescription(this, agent_name));
        agents.emplace(agent_name, rtn);
        return *rtn;
    }
    THROW InvalidAgentName("Agent with name '%s' already exists, "
        "in ModelDescription::newAgent().",
        agent_name.c_str());
}
AgentDescription& ModelDescription::Agent(const std::string &agent_name) {
    auto rtn = agents.find(agent_name);
    if (rtn != agents.end())
        return *rtn->second;
    THROW InvalidAgentName("Agent ('%s') was not found, "
        "in ModelDescription::getAgent().",
        agent_name.c_str());
}
AgentDescription& ModelDescription::cloneAgent(const AgentDescription &agent) {
    // TODO
    return *((*agents.begin()).second);
}

MessageDescription& ModelDescription::newMessage(const std::string &message_name) {
    if (!hasMessage(message_name)) {
        auto rtn = std::shared_ptr<MessageDescription>(new MessageDescription(this, message_name));
        messages.emplace(message_name, rtn);
        return *rtn;
    }
    THROW InvalidMessageName("Message with name '%s' already exists, "
        "in ModelDescription::newAgent().",
        message_name.c_str());
}
MessageDescription& ModelDescription::Message(const std::string &message_name) {
    auto rtn = messages.find(message_name);
    if (rtn != messages.end())
        return *rtn->second;
    THROW InvalidMessageName("Message ('%s') was not found, "
        "in ModelDescription::Message().",
        message_name.c_str());
}
MessageDescription& ModelDescription::cloneMessage(const MessageDescription &message) {
    // TODO
    return *((*messages.begin()).second);
}

EnvironmentDescription& ModelDescription::Environment() {
    return environment;
}
EnvironmentDescription& ModelDescription::cloneEnvironment(const EnvironmentDescription &env) {
    // TODO
    return this->environment;
}

/**
* Const Accessors
*/
std::string ModelDescription::getName() const {
    return name;
}

const AgentDescription& ModelDescription::getAgent(const std::string &agent_name) const {
    auto rtn = agents.find(agent_name);
    if (rtn != agents.end())
        return *rtn->second;
    THROW InvalidAgentVar("Agent ('%s') was not found, "
        "in ModelDescription::getAgent().",
        agent_name.c_str());
}
const MessageDescription& ModelDescription::getMessage(const std::string &message_name) const {
    auto rtn = messages.find(message_name);
    if (rtn != messages.end())
        return *rtn->second;
    THROW InvalidMessageVar("Message ('%s') was not found, "
        "in ModelDescription::getMessage().",
        message_name.c_str());
}
const EnvironmentDescription& ModelDescription::getEnvironment() const {
    return environment;
}

const ModelDescription::AgentMap& ModelDescription::getAgents() const {
    return agents;
}
const ModelDescription::MessageMap& ModelDescription::getMessages() const {
    return messages;
}

bool ModelDescription::hasAgent(const std::string &agent_name) const {
    return agents.find(agent_name) != agents.end();
}
bool ModelDescription::hasMessage(const std::string &message_name) const {
    return messages.find(message_name) != messages.end();
}

ModelDescription ModelDescription::clone(const std::string &cloned_model_name) const {
    // TODO
    return ModelDescription(cloned_model_name);
}