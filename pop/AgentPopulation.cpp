/*
 * AgentPopulation.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include "AgentPopulation.h"

AgentPopulation::AgentPopulation(const ModelDescription &model_description): model(model_description), states_map() {}

AgentPopulation::~AgentPopulation() {}

AgentInstance AgentPopulation::addInstance(const std::string agent_name, const std::string agent_state) {

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
		const AgentDescription& agent_description = model.getAgentDescription(agent_name);
		iter = states_map.insert(AgentStatesMap::value_type(k, std::unique_ptr<AgentStateMemory>(new AgentStateMemory(agent_description, agent_state)))).first;
		return AgentInstance(*iter->second);
	}
}
