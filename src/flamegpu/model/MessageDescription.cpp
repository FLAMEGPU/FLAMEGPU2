#include "flamegpu/model/MessageDescription.h"

/**
 * Constructors
 */
MessageDescription::MessageDescription(ModelDescription * const _model, const std::string &message_name)
    : name(message_name), model(_model) { }
// Copy Construct
MessageDescription::MessageDescription(const MessageDescription &other_message)
    : model(other_message.model) {
    // TODO
}
// Move Construct
MessageDescription::MessageDescription(MessageDescription &&other_message)
    : model(other_message.model) {
    // TODO
}
// Copy Assign
MessageDescription& MessageDescription::operator=(const MessageDescription &other_message) {
    // TODO
    return *this;
}
// Move Assign
MessageDescription& MessageDescription::operator=(MessageDescription &&other_message) {
    // TODO
    return *this;
}

MessageDescription MessageDescription::clone(const std::string &cloned_message_name) const {
    // TODO
    return MessageDescription(model, cloned_message_name);
}

/**
 * Const Accessors
 */
std::string MessageDescription::getName() const {
    return name;
}

std::type_index MessageDescription::getVariableType(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return f->second.type;
    }
    THROW InvalidAgentVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableType().",
        name.c_str(), variable_name.c_str());
}
size_t MessageDescription::getVariableSize(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return f->second.type_size;
    }
    THROW InvalidAgentVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableSize().",
        name.c_str(), variable_name.c_str());
}
MessageDescription::size_type MessageDescription::getVariableLength(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return f->second.elements;
    }
    THROW InvalidAgentVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableLength().",
        name.c_str(), variable_name.c_str());
}
MessageDescription::size_type MessageDescription::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX variables
    return static_cast<size_type>(variables.size());
}
const MessageDescription::VariableMap &MessageDescription::getVariables() const {
    return variables;
}
bool MessageDescription::hasVariable(const std::string &variable_name) const {
    return variables.find(variable_name) != variables.end();
}