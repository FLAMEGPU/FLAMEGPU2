/*
 * AgentPopulation.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include "AgentPopulation.h"

AgentPopulation::AgentPopulation(const ModelDescription &model_description, const std::string name, unsigned int max_size): 
	model(model_description), 
	agent_name(name), 
	maximum_size(max_size), 
	states_map() {

}

AgentPopulation::~AgentPopulation() {}

AgentInstance AgentPopulation::addInstance(const std::string agent_state) {

	//boost::tuple<std::string, std::string> k(agent_name, agent_state);
	std::string k = agent_name + "_" + agent_state;

	//check if the state map exists
	AgentStatesMap::iterator iter;
	iter = states_map.find(k);

	//check max size is not exceeded if it is then increase it
	if (iter->second->getSize() == maximum_size){
		maximum_size += POPULATION_SIZE_INCREMENT;
	}

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

const AgentStateMemory& AgentPopulation::getStateMemory(const std::string agent_state) const {
	std::string k = agent_name + "_" + agent_state;

	//check if the state map exists
	AgentStatesMap::const_iterator iter;
	iter = states_map.find(k);

	if (iter == states_map.end()){
		throw std::runtime_error("Invalid agent state name");
	}

	return *iter->second;
}

const std::string AgentPopulation::getAgentName() const {
	return agent_name;
}

unsigned int AgentPopulation::getMaximumPopulationSize() const{
	return maximum_size;
}