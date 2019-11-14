 /**
 * @file AgentMemory.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <iostream>

#include "flamegpu/pop/AgentStateMemory.h"

#include "flamegpu/pop/AgentPopulation.h"
#include "flamegpu/model/AgentDescription.h"

AgentStateMemory::AgentStateMemory(const AgentPopulation &p, unsigned int initial_capacity) : population(p) {
    // state memory map is cloned from the agent description
    population.getAgentDescription().initEmptyStateMemoryMap(state_memory);

    // set current size to 0 (no agents in this state yet)
    current_size = 0;

    // if there is an initial population size then resize the memory vectors
    if (initial_capacity > 0)
        resizeMemoryVectors(initial_capacity);
}

unsigned int AgentStateMemory::incrementSize() {
    // add one to current size (returns old size)
    return current_size++;
}

GenericMemoryVector& AgentStateMemory::getMemoryVector(const std::string variable_name) {
    StateMemoryMap::iterator iter;
    iter = state_memory.find(variable_name);

    if (iter == state_memory.end()) {
        THROW InvalidAgentVar("Agent ('%s') variable ('%s') was not found, "
            "in AgentStateMemory::getMemoryVector().",
            population.getAgentName().c_str(), variable_name.c_str());
    }
    return *(iter->second);
}

const GenericMemoryVector& AgentStateMemory::getReadOnlyMemoryVector(const std::string variable_name) const {
    StateMemoryMap::const_iterator iter;
    iter = state_memory.find(variable_name);

    if (iter == state_memory.end()) {
        THROW InvalidAgentVar("Agent ('%s') variable ('%s') was not found, "
            "in AgentStateMemory::getReadOnlyMemoryVector().",
            population.getAgentName().c_str(), variable_name.c_str());
    }

    return *(iter->second);
}

const std::type_info& AgentStateMemory::getVariableType(std::string variable_name) {
    return population.getAgentDescription().getVariableType(variable_name);
}

bool AgentStateMemory::isSameDescription(const AgentDescription& description) const {
    return (&description == &population.getAgentDescription());
}

void AgentStateMemory::resizeMemoryVectors(unsigned int s) {
    // all size checking is done by the population

    MemoryMap::const_iterator iter;
    const MemoryMap &m = population.getAgentDescription().getMemoryMap();

    for (iter = m.begin(); iter != m.end(); iter++) {
        const std::string variable_name = iter->first;

        GenericMemoryVector &v = getMemoryVector(variable_name);
        v.resize(s);
    }
}

unsigned int AgentStateMemory::getPopulationCapacity() const {
    return population.getMaximumStateListCapacity();
}

unsigned int AgentStateMemory::getStateListSize() const {
    return current_size;
}

void AgentStateMemory::overrideStateListSize(unsigned int size) {
    current_size = size;
}
