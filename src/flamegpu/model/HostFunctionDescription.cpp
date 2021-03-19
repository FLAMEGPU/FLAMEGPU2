#include <iostream>
#include <string>

#include "flamegpu/model/HostFunctionDescription.h"


/**
 * Constructors
 */

HostFunctionDescription::HostFunctionDescription(FLAMEGPU_HOST_FUNCTION_POINTER host_function) {
    this->function = host_function;
}

bool HostFunctionDescription::operator==(const HostFunctionDescription& rhs) const {
    return *this->function == *rhs.function;  // Compare content is functionally the same
}
bool HostFunctionDescription::operator!=(const HostFunctionDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */

FLAMEGPU_HOST_FUNCTION_POINTER HostFunctionDescription::getFunctionPtr() const {
    return function;
}
