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

#include <flamegpu/pop/AgentPopulation.h>

#include <flamegpu/pop/AgentInstance.h>
#include <flamegpu/model/AgentDescription.h>

AgentPopulation::AgentPopulation(const AgentDescription &agent_description, unsigned int initial_size):
    agent(agent_description),
    states_map(),
    maximum_size(initial_size) {
    // init the state maps
    const StateMap& sm = agent.getStateMap();

    // loop through states in agent description and create memory for each
    for (const StateMapPair &smp : sm) {

        // add a new state memory object to the states map
        states_map.insert(AgentStatesMap::value_type(smp.first, std::unique_ptr<AgentStateMemory>(new AgentStateMemory(*this, initial_size))));
    }
}

AgentPopulation::~AgentPopulation() {}

AgentInstance AgentPopulation::getNextInstance(const std::string agent_state) {

    // get correct state memory
    AgentStatesMap::iterator sm;
    sm = states_map.find(agent_state);
    if (sm == states_map.end())

        throw InvalidPopulationData("Agent state not found when pushing back instance");

    // increment size gives old size
    unsigned int index = sm->second->incrementSize();
    if (index >= getMaximumStateListCapacity())
        throw InvalidMemoryCapacity("Agent state size will be exceeded");

    // return new instance from state memory with index of current size (then increment)
    return AgentInstance(*sm->second, index);

}

AgentInstance AgentPopulation::getInstanceAt(unsigned int index, const std::string agent_state) {

    // get correct state memory
    AgentStatesMap::iterator sm;
    sm = states_map.find(agent_state);
    if (sm == states_map.end())

        throw InvalidPopulationData("Agent state not found when pushing back instance");

    // check the index does not exceed current size
    if (index >= sm->second->getStateListSize())
        throw InvalidMemoryCapacity("Can not get Instance. Index exceeds current size.");

    // return new instance from state memory with index of current size (then increment)
    return AgentInstance(*sm->second, index);

}

AgentStateMemory& AgentPopulation::getStateMemory(const std::string agent_state) {

    // check if the state map exists
    AgentStatesMap::const_iterator iter;
    iter = states_map.find(agent_state);

    if (iter == states_map.end()) {

        // throw std::runtime_error("Invalid agent state name");
        throw InvalidStateName();
    }

    return *iter->second;
}

/* this is the current size */
unsigned int AgentPopulation::getCurrentListSize(const std::string agent_state) {

    return getStateMemory(agent_state).getStateListSize();
}

const AgentStateMemory& AgentPopulation::getReadOnlyStateMemory(const std::string agent_state) const {

    // check if the state map exists
    AgentStatesMap::const_iterator iter;
    iter = states_map.find(agent_state);

    if (iter == states_map.end()) {

        // throw std::runtime_error("Invalid agent state name");
        throw InvalidStateName();
    }

    return *iter->second;
}

const std::string AgentPopulation::getAgentName() const {

    return agent.getName();
}

const AgentDescription& AgentPopulation::getAgentDescription() const {

    return agent;
}

unsigned int AgentPopulation::getMaximumStateListCapacity() const {

    return maximum_size;
}

void AgentPopulation::setStateListCapacity(unsigned int size) {

    if (size < maximum_size)

        throw InvalidPopulationData("Can not reduce size of agent population state list!");

    // set the maximum size
    maximum_size = size;

    for (AgentStatesMapPair &smp : states_map) {

        smp.second->resizeMemoryVectors(maximum_size);
    }
}
