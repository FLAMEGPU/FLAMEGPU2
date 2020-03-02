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
#include <utility>

// include dependant class agent state memory (required in template functions)
#include "flamegpu/pop/AgentStateMemory.h"

class AgentInstance {
 public:
    AgentInstance(AgentStateMemory& state_memory, unsigned int i);

    virtual ~AgentInstance();

    template <typename T> void setVariable(const std::string &variable_name, const T &value);
    template <typename T, unsigned int N>  void setVariable(const std::string &variable_name, const std::array<T, N> &value);
    template <typename T> void setVariable(const std::string &variable_name, const unsigned int &index, const T &value);

    template <typename T>  T getVariable(const std::string &variable_name);
    template <typename T, unsigned int N>  std::array<T, N> getVariable(const std::string &variable_name);
    template <typename T>  T getVariable(const std::string &variable_name, const unsigned int &index);

 private:
    const unsigned int index;
    AgentStateMemory& agent_state_memory;
};

template <typename T>
void AgentInstance::setVariable(const std::string &variable_name, const T &value) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    const Variable &v_desc = agent_state_memory.getVariableDescription(variable_name);
    if (v_desc.elements != 1) {
        THROW InvalidAgentVar("This method is not suitable for agent array variables, "
            " variable '%s' was passed, "
            "in AgentInstance::getVariable().",
            variable_name.c_str());
    }
    if (v.getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Wrong variable type passed to AgentInstance::setVariable(). "
            "This expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    // do the replace
    reinterpret_cast<T*>(v.getDataPtr())[index] = value;
}
template <typename T, unsigned int N>
void AgentInstance::setVariable(const std::string &variable_name, const std::array<T, N> &value) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    const Variable &v_desc = agent_state_memory.getVariableDescription(variable_name);
    // if (v_desc.elements < 2) {
    //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
    //         "in AgentInstance::setVariable().",
    //         variable_name.c_str());
    // }
    if (v.getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Wrong variable type passed to AgentInstance::setVariable(). "
            "This expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    if (v_desc.elements != N) {
        THROW InvalidVarArrayLen("Wrong variable array length passed to AgentInstance::setVariable(). "
            "This expects %u, but %u was requested.",
            v_desc.elements, N);
    }
    memcpy(reinterpret_cast<T*>(v.getDataPtr()) + (index * N), value.data(), sizeof(T) * N);
}
template <typename T>
void AgentInstance::setVariable(const std::string &variable_name, const unsigned int &array_index, const T &value) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    const Variable &v_desc = agent_state_memory.getVariableDescription(variable_name);
    // if (v_desc.elements < 2) {
    //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
    //         "in AgentInstance::setVariable().",
    //         variable_name.c_str());
    // }
    if (v.getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Wrong variable type passed to AgentInstance::setVariable(). "
            "This expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    if (v_desc.elements <= array_index) {
        THROW OutOfRangeVarArray("Index %d is out range %d in AgentInstance::setVariable(). ",
            array_index, v_desc.elements);
    }
    reinterpret_cast<T*>(v.getDataPtr())[(index * v_desc.elements) + array_index] = value;
}

template <typename T>
T AgentInstance::getVariable(const std::string &variable_name) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    const Variable &v_desc = agent_state_memory.getVariableDescription(variable_name);
    if (v_desc.elements != 1) {
        THROW InvalidAgentVar("This method is not suitable for agent array variables, "
            " variable '%s' was passed, "
            "in AgentInstance::getVariable().",
            variable_name.c_str());
    }
    if (v.getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Wrong variable type passed to AgentInstance::getVariable(). "
            "This expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    // todo error handling around the cast to check for exceptions
    return reinterpret_cast<const T*>(v.getReadOnlyDataPtr())[index];
}
template <typename T, unsigned int N>
std::array<T, N> AgentInstance::getVariable(const std::string &variable_name) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    const Variable &v_desc = agent_state_memory.getVariableDescription(variable_name);
    // if (v_desc.elements < 2) {
    //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
    //         "in AgentInstance::getVariable().",
    //         variable_name.c_str());
    // }
    if (v.getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Wrong variable type passed to AgentInstance::getVariable(). "
            "This expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    if (v_desc.elements != N) {
        THROW InvalidVarArrayLen("Wrong variable array length passed to AgentInstance::getVariable(). "
            "This expects %u, but %u was requested.",
            v_desc.elements, N);
    }
    std::array<T, N> rtn;
    memcpy(rtn.data(), reinterpret_cast<T*>(v.getDataPtr()) + (index * N), sizeof(T) * N);
    return std::move(rtn);  // I think std::move will transfer the std::array rather than duplicating it
}
template <typename T>
T AgentInstance::getVariable(const std::string &variable_name, const unsigned int &array_index) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);
    const Variable &v_desc = agent_state_memory.getVariableDescription(variable_name);
    // if (v_desc.elements < 2) {
    //     THROW InvalidAgentVar("Agent variable '%s' in not an array variable, "
    //         "in AgentInstance::getVariable().",
    //         variable_name.c_str());
    // }
    if (v.getType() != std::type_index(typeid(T))) {
        THROW InvalidVarType("Wrong variable type passed to AgentInstance::getVariable(). "
            "This expects '%s', but '%s' was requested.",
            v.getType().name(), typeid(T).name());
    }
    if (v_desc.elements <= array_index) {
        THROW OutOfRangeVarArray("Index %d is out range %d in AgentInstance::getVariable(). ",
            array_index, v_desc.elements);
    }
    return reinterpret_cast<T*>(v.getDataPtr())[(index * v_desc.elements) + array_index];
}

#endif  // INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_
