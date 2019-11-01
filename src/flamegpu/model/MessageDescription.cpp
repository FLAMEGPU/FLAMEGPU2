/**
 * @file MessageDescription.cpp
 * @authors
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "flamegpu/model/MessageDescription.h"
#include "flamegpu/exception/FGPUException.h"

MessageDescription::MessageDescription(const std::string message_name, unsigned int initial_size) : name(message_name), variables(), maximum_size(initial_size) { }

MessageDescription::~MessageDescription() {}

const std::string MessageDescription::getName() const {
    return name;
}

VariableMap& MessageDescription::getVariableMap() {
    return variables;
}

const VariableMap& MessageDescription::getVariableMap() const {
    return variables;
}

size_t MessageDescription::getMemorySize() const {
    size_t size = 0;
    for (VarTypeSizeMap::const_iterator it = sizes.begin(); it != sizes.end(); it++) {
        size += it->second;
    }
    return size;
}

unsigned int MessageDescription::getNumberMessageVariables() const {
    return static_cast<unsigned int>(variables.size());
}

size_t MessageDescription::getMessageVariableSize(const std::string variable_name) const {
    // get the variable name type
    VariableMap::const_iterator mm = variables.find(variable_name);
    if (mm == variables.end())
        throw InvalidMessageVar("Invalid message memory variable");
    const std::type_info *t = &(mm->second);
    // get the type size
    VarTypeSizeMap::const_iterator tsm = sizes.find(t);
    if (tsm == sizes.end())
        throw InvalidMapEntry("Missing entry in type sizes map");
    return tsm->second;
}

const std::type_info& MessageDescription::getVariableType(const std::string variable_name) const {
    VariableMap::const_iterator iter;
    iter = variables.find(variable_name);

    if (iter == variables.end())
        throw InvalidMessageVar("Invalid message memory variable");

    return iter->second;
}

unsigned int MessageDescription::getMaximumMessageListCapacity() const {
    return maximum_size;
}
