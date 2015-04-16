/*
 * AgentPopulation.h
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#ifndef AGENTPOPULATION_H_
#define AGENTPOPULATION_H_

#include <boost/shared_ptr.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include "boost/tuple/tuple.hpp"

#include "../model/ModelDescription.h"
#include "AgentStateMemory.h"
#include "AgentInstance.h"

typedef boost::ptr_map<std::string, AgentStateMemory> AgentStatesMap;	//key is concat of agent and state name!

class AgentPopulation {
public:
	AgentPopulation(ModelDescription &model_description): model(model_description), states_map() {}
	virtual ~AgentPopulation() {}

	AgentInstance addInstance(std::string agent_name, std::string agent_state);

private:

	ModelDescription &model;
	AgentStatesMap states_map;


};

#endif /* AGENTPOPULATION_H_ */
