/*
 * AgentPopulation.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include "AgentPopulation.h"

AgentInstance AgentPopulation::addInstance(std::string agent_name, std::string agent_state) {

	//boost::tuple<std::string, std::string> k(agent_name, agent_state);
	std::string k = agent_name + "_" + agent_state;

	//check if the state map exists
	AgentStatesMap::iterator iter;
	iter = states_map.find(k);

	//existing state map
	if (iter != states_map.end())
	{
		return AgentInstance(*iter->second);
	}
	//create new state map
	else
	{
		AgentDescription& agent_description = model.getAgentDescription(agent_name);
		iter = states_map.insert(k, new AgentStateMemory(agent_description, agent_state)).first;
		return AgentInstance(*iter->second);
	}
}
