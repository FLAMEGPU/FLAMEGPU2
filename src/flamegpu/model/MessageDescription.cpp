#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/model/AgentDescription.h" // Used by Move-Assign
/**
 * Constructors
 */
MessageDescription::MessageDescription(ModelData *const _model, MessageData *const description)
    : model(_model)
    , message(description) { }
// Copy Construct
MessageDescription::MessageDescription(const MessageDescription &other_message)
    : model(other_message.model)
    , message(other_message.message) {
    // TODO
}
// Move Construct
MessageDescription::MessageDescription(MessageDescription &&other_message) noexcept
    : model(move(other_message.model))
    , message(other_message.message) { }
// Copy Assign
MessageDescription& MessageDescription::operator=(const MessageDescription &other_message) {
    // TODO
    return *this;
}
// Move Assign
MessageDescription& MessageDescription::operator=(MessageDescription &&other_message) {
    //std::string old_name = this->message->name;
    //this->message->name = move(other_message.message->name);
    //this->message->variables = move(other_message.message->variables);
    //if (old_name != this->message->name) {
    //    // TODO: if name has changed, update references?
    //    // Rename inside model
    //    auto it = model->messages.find(old_name);
    //    if(it != model->messages.end()) {
    //        model->messages.emplace(this->message->name, it->second);
    //        model->messages.erase(it);
    //    } else {
    //        // This should never happen
    //    }
    //}
    return *this;
}

MessageDescription MessageDescription::clone(const std::string &cloned_message_name) const {
    // TODO
    return MessageDescription(model, message);
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