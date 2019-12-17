#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/model/AgentDescription.h"  // Used by Move-Assign

/**
 * Constructors
 */
MessageDescription::MessageDescription(ModelData *const _model, MessageData *const description)
    : model(_model)
    , message(description) { }

bool MessageDescription::operator==(const MessageDescription& rhs) const {
    return *this->message == *rhs.message;  // Compare content is functionally the same
}
bool MessageDescription::operator!=(const MessageDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
std::string MessageDescription::getName() const {
    return message->name;
}

std::type_index MessageDescription::getVariableType(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableType().",
        message->name.c_str(), variable_name.c_str());
}
size_t MessageDescription::getVariableSize(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.type_size;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableSize().",
        message->name.c_str(), variable_name.c_str());
}
ModelData::size_type MessageDescription::getVariableLength(const std::string &variable_name) const {
    auto f = message->variables.find(variable_name);
    if (f != message->variables.end()) {
        return f->second.elements;
    }
    THROW InvalidMessageVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableLength().",
        message->name.c_str(), variable_name.c_str());
}
ModelData::size_type MessageDescription::getVariablesCount() const {
    // Downcast, will never have more than UINT_MAX variables
    return static_cast<ModelData::size_type>(message->variables.size());
}
bool MessageDescription::hasVariable(const std::string &variable_name) const {
    return message->variables.find(variable_name) != message->variables.end();
}
