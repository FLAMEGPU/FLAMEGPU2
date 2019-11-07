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

    template <typename T>  const T getVariable(std::string variable_name);

 private:
    const unsigned int index;
    AgentStateMemory& agent_state_memory;
};

template <typename T> void AgentInstance::setVariable(std::string variable_name, const T value) {
    // todo check that the variable exists

    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);

    // get the vector of correct type
    std::vector<T> &t_v = v.getVector<T>();
    typename std::vector<T>::iterator it = t_v.begin() + index;

    // do the insert
    t_v.insert(it, value);
}

template <typename T>  const T AgentInstance::getVariable(std::string variable_name) {
    // todo check that the variable exists
    GenericMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);

    // get the vector of correct type
    std::vector<T> &t_v = v.getVector<T>();
    // typename std::vector<T>::iterator it = t_v.begin() + index;

    // todo error handling around the cast to check for exceptions
    return t_v.at(index);
}

#endif  // INCLUDE_FLAMEGPU_POP_AGENTINSTANCE_H_
