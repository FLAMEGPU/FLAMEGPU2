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

typedef std::map<const std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMap;	//key is concat of agent and state name!

class AgentPopulation {
public:
	AgentPopulation(const ModelDescription &model_description);

	virtual ~AgentPopulation();

	AgentInstance addInstance(const std::string agent_name, const std::string agent_state);

private:

	const ModelDescription &model;
	AgentStatesMap states_map;


};

#endif /* AGENTPOPULATION_H_ */
