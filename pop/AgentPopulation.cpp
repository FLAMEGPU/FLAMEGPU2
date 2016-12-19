 /**
 * @file AgentPopulation.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "AgentPopulation.h"

AgentPopulation::AgentPopulation(const AgentDescription &agent_description, unsigned int initial_size):
	agent(agent_description),
    maximum_size(initial_size),
    states_map()
{
	//init the state maps
	const StateMap& sm = agent.getStateMap();

	//loop through states in agent description and create memory for each
	for (const StateMapPair &smp : sm){
		//add a new state memory object to the states map
		states_map.insert(AgentStatesMap::value_type(smp.first, std::unique_ptr<AgentStateMemory>(new AgentStateMemory(*this, initial_size))));
	}
}

AgentPopulation::~AgentPopulation() {}

AgentInstance AgentPopulation::pushBackInstance(const std::string agent_state)
{
	//increment all state memory vectors
	for (AgentStatesMapPair &smp : states_map){
		smp.second->incrementSize();
	}
	
	//get correct state memory
	AgentStatesMap::iterator sm;
	sm = states_map.find(agent_state);
	if (sm == states_map.end())
		throw std::exception("Agent state not found when pushing back instance");

	//return new instance from state memory with index of current size (then increment)
	return AgentInstance(*sm->second, maximum_size++);

}

AgentInstance AgentPopulation::getInstanceAt(unsigned int index, const std::string agent_state)
{
	//check the index does not exceed current size
	if (index <= maximum_size)
		throw std::exception("Can not get instance. Try increasing size of pushing new instance.");

	//get correct state memory
	AgentStatesMap::iterator sm;
	sm = states_map.find(agent_state);
	if (sm == states_map.end())
		throw std::exception("Agent state not found when pushing back instance");

	//return new instance from state memory with index of current size (then increment)
	return AgentInstance(*sm->second, index);

}

const AgentStateMemory& AgentPopulation::getStateMemory(const std::string agent_state) const
{
    //check if the state map exists
    AgentStatesMap::const_iterator iter;
	iter = states_map.find(agent_state);

    if (iter == states_map.end())
    {
        //throw std::runtime_error("Invalid agent state name");
        throw InvalidStateName();
    }

    return *iter->second;
}

const std::string AgentPopulation::getAgentName() const
{
    return agent.getName();
}

const AgentDescription& AgentPopulation::getAgentDescription() const
{
	return agent;
}

unsigned int AgentPopulation::getMaximumStateListSize() const
{
	return maximum_size;
}

void AgentPopulation::setStateListSize(unsigned int size)
{
	if (size < maximum_size){
		throw std::exception("Can not reduce size of agent population state list!");
	}

	//set the maximum size
	maximum_size = size;

	for (AgentStatesMapPair &smp : states_map){
		smp.second->resizeMemoryVectors(maximum_size);
	}
}
