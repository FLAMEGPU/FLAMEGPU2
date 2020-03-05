/**
* @file AgentPopulation.cpp
* @authors Paul
* @date
* @brief
*
* @see
* @warning
*/

#include <exception>

#include "flamegpu/pop/AgentPopulation.h"

#include "flamegpu/pop/AgentInstance.h"
#include "flamegpu/model/AgentDescription.h"


const unsigned int AgentPopulation::DEFAULT_POPULATION_SIZE = 10;  // 1024

AgentPopulation::AgentPopulation(const AgentDescription &agent_description, unsigned int initial_size)
    : agent(agent_description.agent->clone())  // Set parent to nullptr, shouldn't need to refer upwards
    , states_map()
    , maximum_size(initial_size) {
    // init the state maps
    // loop through states in agent description and create memory for each
    for (const auto &state : agent->states) {
        // add a new state memory object to the states map
        states_map.insert(AgentStatesMap::value_type(state, std::make_unique<AgentStateMemory>(*this, initial_size)));
    }
}

AgentPopulation::~AgentPopulation() {}

AgentInstance AgentPopulation::getNextInstance(const std::string agent_state) {
    // get correct state memory
    AgentStatesMap::iterator sm;
    sm = states_map.find(agent_state);
    if (sm == states_map.end()) {
        THROW InvalidPopulationData("Agent ('%s') state ('%s') not found, "
            "in AgentPopulation::getNextInstance()",
            agent->name.c_str(), agent_state.c_str());
    }

    // increment size and initialise new agent, returns old size
    unsigned int index = sm->second->incrementSize();
    if (index >= getMaximumStateListCapacity()) {
        THROW InvalidMemoryCapacity("Agent ('%s') state ('%s') size would be execeed, "
            "in AgentPopulation::getNextInstance()",
            agent->name.c_str(), agent_state.c_str());
    }
    // return new instance from state memory with index of current size (then increment)
    return AgentInstance(*sm->second, index);
}

AgentInstance AgentPopulation::getInstanceAt(unsigned int index, const std::string agent_state) {
    // get correct state memory
    AgentStatesMap::iterator sm;
    sm = states_map.find(agent_state);
    if (sm == states_map.end()) {
        THROW InvalidPopulationData("Agent state ('%s') was not found, "
            "in AgentPopulation::getInstanceAt().",
            agent_state.c_str());
    }
    // check the index does not exceed current size
    if (index >= sm->second->getStateListSize()) {
        THROW InvalidMemoryCapacity("Index '%u' exceeds Agent ('%s') state ('%s') current size, "
            "in AgentPopulation::getInstanceAt().",
            index, agent->name.c_str(), agent_state.c_str());
    }

    // return new instance from state memory with index of current size (then increment)
    return AgentInstance(*sm->second, index);
}

AgentStateMemory& AgentPopulation::getStateMemory(const std::string agent_state) {
    // check if the state map exists
    AgentStatesMap::const_iterator iter = states_map.find(agent_state);

    if (iter == states_map.end()) {
        THROW InvalidStateName("Agent ('%s') state name ('%s') was not found, "
            "in AgentPopulation::getStateMemory().",
            agent->name.c_str(), agent_state.c_str());
    }

    return *iter->second;
}

/* this is the current size */
unsigned int AgentPopulation::getCurrentListSize(const std::string agent_state) const {
    // check if the state map exists
    AgentStatesMap::const_iterator iter = states_map.find(agent_state);

    if (iter == states_map.end()) {
        THROW InvalidStateName("Agent ('%s') state name ('%s') was not found, "
            "in AgentPopulation::getCurrentListSize().",
            agent->name.c_str(), agent_state.c_str());
    }
    return iter->second->getStateListSize();
}

const AgentStateMemory& AgentPopulation::getReadOnlyStateMemory(const std::string agent_state) const {
    // check if the state map exists
    AgentStatesMap::const_iterator iter;
    iter = states_map.find(agent_state);

    if (iter == states_map.end()) {
        THROW InvalidAgentFunc("Agent ('%s') state name ('%s') was not found, "
            "in AgentPopulation::getReadOnlyStateMemory().",
            agent->name.c_str(), agent_state.c_str());
    }

    return *iter->second;
}

std::string AgentPopulation::getAgentName() const {
    return agent->name;
}

const AgentData& AgentPopulation::getAgentDescription() const {
    return *agent;
}

unsigned int AgentPopulation::getMaximumStateListCapacity() const {
    return maximum_size;
}

void AgentPopulation::setStateListCapacity(unsigned int size) {
    if (size < maximum_size) {
        THROW InvalidPopulationData("Agent population state list sizes cannot be reduced (%u -> %u attempted), "
            "in AgentPopulation::setStateListCapacity().",
            maximum_size, size);
    }

    // set the maximum size
    maximum_size = size;

    for (auto &smp : states_map) {
        smp.second->resizeMemoryVectors(maximum_size);
    }
}
