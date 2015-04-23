/*
 * AgentPopulation.h
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#ifndef AGENTPOPULATION_H_
#define AGENTPOPULATION_H_

#include <memory>
#include <map>

#include "../model/ModelDescription.h"
#include "AgentStateMemory.h"
#include "AgentInstance.h"

#define POPULATION_SIZE_INCREMENT 1024
#define DEFAULT_POPULATION_SIZE 1024

typedef std::map<const std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMap;	//key is concat of agent and state name!

class AgentPopulation {
public:
	AgentPopulation(const ModelDescription &model_description, const std::string agent_name, unsigned int size_hint=DEFAULT_POPULATION_SIZE);

	virtual ~AgentPopulation();

	AgentInstance addInstance(const std::string agent_state = "default");

	const AgentStateMemory& getStateMemory(const std::string agent_state = "default") const;

	const std::string getAgentName() const;

	unsigned int getMaximumPopulationSize() const;

private:

	const ModelDescription &model;
	const std::string agent_name;
	AgentStatesMap states_map;
	unsigned int maximum_size; //size is maximum size for agents in any single state (same for all states of same agent type)


};

#endif /* AGENTPOPULATION_H_ */
