#include <iostream>
#include <string>

#include "flamegpu/model/HostFunctionDescription.h"

namespace flamegpu {


/**
 * Constructors
 */

HostFunctionDescription::HostFunctionDescription(std::string name, FLAMEGPU_HOST_FUNCTION_POINTER host_function) {
    this->name = name;
    this->function = host_function;
}

HostFunctionDescription::HostFunctionDescription(std::string name, HostFunctionCallback *func_callback) {
    this->name = name;
    this->callbackObject = func_callback;
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

HostFunctionCallback* HostFunctionDescription::getCallbackObject() {
    return callbackObject;
}

}  // namespace flamegpu
