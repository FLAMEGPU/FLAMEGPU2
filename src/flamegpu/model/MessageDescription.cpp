#include "flamegpu/model/MessageDescription.h"

/**
 * Constructors
 */
MessageDescription::MessageDescription(const std::string &message_name)
    : name(message_name) { }
// Copy Construct
MessageDescription::MessageDescription(const MessageDescription &other_message) {

}
// Move Construct
MessageDescription::MessageDescription(MessageDescription &&other_message) {

}
// Copy Assign
MessageDescription& MessageDescription::operator=(const MessageDescription &other_message) {

}
// Move Assign
MessageDescription& MessageDescription::operator=(MessageDescription &&other_message) {

}

MessageDescription MessageDescription::clone(const std::string &cloned_message_name) const {

}



/**
 * Accessors
 */
template<typename T, MessageDescription::size_type N = 1>
void MessageDescription::newVariable(const std::string &variable_name) {
    if (variables.find(variable_name) == variables.end()) {
        variables.emplace(variable_name, std::make_tuple(typeid(T), sizeof(T), N));
        return;
    }
    THROW InvalidAgentVar("Message ('%s') already contains variable '%s', "
        "in MessageDescription::newVariable().",
        name.c_str(), variable_name.c_str());
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
        return std::get<0>(f->second);
    }
    THROW InvalidAgentVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableType().",
        name.c_str(), variable_name.c_str());
}
size_t MessageDescription::getVariableSize(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return std::get<1>(f->second);
    }
    THROW InvalidAgentVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableSize().",
        name.c_str(), variable_name.c_str());
}
MessageDescription::size_type MessageDescription::getVariableLength(const std::string &variable_name) const {
    auto f = variables.find(variable_name);
    if (f != variables.end()) {
        return std::get<2>(f->second);
    }
    THROW InvalidAgentVar("Message ('%s') does not contain variable '%s', "
        "in MessageDescription::getVariableLength().",
        name.c_str(), variable_name.c_str());
}
MessageDescription::size_type MessageDescription::getVariablesCount() const {
    return variables.size();
}
const MessageDescription::VariableMap &MessageDescription::getVariables() const {

}
bool MessageDescription::hasVariable(const std::string &variable_name) const {
    return variables.find(variable_name) != variables.end();
}