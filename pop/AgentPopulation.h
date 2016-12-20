 /**
 * @file AgentPopulation.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef AGENTPOPULATION_H_
#define AGENTPOPULATION_H_

#include <memory>
#include <map>

class AgentStateMemory;	//forward declaration

#include "../model/AgentDescription.h"
//#include "AgentStateMemory.h"
//#include "AgentInstance.h"

class AgentInstance;

#define DEFAULT_POPULATION_SIZE 1024

typedef std::map<const std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMap;	//key is concat of agent and state name!
typedef std::pair<const std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMapPair;

class AgentPopulation {
public:
	AgentPopulation(const AgentDescription &agent_description, unsigned int initial_size = DEFAULT_POPULATION_SIZE);

	virtual ~AgentPopulation();

	AgentInstance pushBackInstance(const std::string agent_state = "default");

	AgentInstance getInstanceAt(unsigned int index, const std::string agent_state = "default");

	const AgentStateMemory& getStateMemory(const std::string agent_state = "default") const;

	const std::string getAgentName() const;

	const AgentDescription& getAgentDescription() const;

	/* This is the maximum size of any single state list. */
	unsigned int getMaximumStateListSize() const;

	void setStateListSize(unsigned int);

private:

	const AgentDescription &agent;
	AgentStatesMap states_map;
	unsigned int maximum_size; //size is maximum size for agents in any single state (same for all states of same agent type)


};

#endif /* AGENTPOPULATION_H_ */
