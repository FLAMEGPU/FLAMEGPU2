 /**
 * @file AgentInstance.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_
#define INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_

#include <vector>
#include <string>

// include dependant class agent state memory (required in template functions)
#include "flamegpu/pop/AgentStateMemory.h"

class AgentInstance {
 public:
    AgentInstance(AgentStateMemory& state_memory, unsigned int i);

    virtual ~AgentInstance();

    template <typename T> void setVariable(std::string variable_name, const T value);

    template <typename T>  T getVariable(std::string variable_name);

 private:
    const unsigned int index;
    AgentStateMemory& agent_state_memory;
};

template <typename T> void AgentInstance::setVariable(std::string variable_name, const T value) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    if (v.getType() != typeid(T)) {
        THROW InvalidVarType("Wrong variable type passed to GenericMemoryVector::getVector(). "
            "This agent data vector expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    // do the replace
    reinterpret_cast<T*>(v.getDataPtr())[index] = value;
}

template <typename T> T AgentInstance::getVariable(std::string variable_name) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    if (v.getType() != typeid(T)) {
        THROW InvalidVarType("Wrong variable type passed to GenericMemoryVector::getVector(). "
            "This agent data vector expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    // todo error handling around the cast to check for exceptions
    return reinterpret_cast<const T*>(v.getReadOnlyDataPtr())[index];
}

#endif  // INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_
