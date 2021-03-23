#include <iostream>
#include <string>

#include "flamegpu/model/HostFunctionDescription.h"


/**
 * Constructors
 */

HostFunctionDescription::HostFunctionDescription(std::string name, FLAMEGPU_HOST_FUNCTION_POINTER host_function) {
    this->name = name;
    this->function = host_function;
}

bool HostFunctionDescription::operator==(const HostFunctionDescription& rhs) const {
    return *this->function == *rhs.function;  // Compare content is functionally the same
}
bool HostFunctionDescription::operator!=(const HostFunctionDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Accessors
 */

std::string HostFunctionDescription::getName() { 
    return name;
}

FLAMEGPU_HOST_FUNCTION_POINTER HostFunctionDescription::getFunctionPtr() const {
    return function;
}
