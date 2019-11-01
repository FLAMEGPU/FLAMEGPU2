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

// include sub dependency AgentInstance which includes AgentStateMemory
#include <flamegpu/pop/AgentInstance.h>

// forward declarations
class AgentDescription;


#define DEFAULT_POPULATION_SIZE 10// 1024

typedef std::map<const std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMap;    // key is concat of agent and state name!
typedef std::pair<const std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMapPair;

class AgentPopulation {
public:
    AgentPopulation(const AgentDescription &agent_description, unsigned int initial_size = DEFAULT_POPULATION_SIZE);

    virtual ~AgentPopulation();

    AgentInstance getNextInstance(const std::string agent_state = "default");

    AgentInstance getInstanceAt(unsigned int index, const std::string agent_state = "default");

    AgentStateMemory& getStateMemory(const std::string agent_state = "default");

    unsigned int getCurrentListSize(const std::string agent_state = "default");

    const AgentStateMemory& getReadOnlyStateMemory(const std::string agent_state = "default") const;

    const std::string getAgentName() const;

    const AgentDescription& getAgentDescription() const;

    /* This is the maximum size of any single state list. */
    unsigned int getMaximumStateListCapacity() const;

    void setStateListCapacity(unsigned int);

private:

    const AgentDescription &agent;
    AgentStatesMap states_map;
    unsigned int maximum_size; // size is maximum size for agents in any single state (same for all states of same agent type)


};

#endif /* AGENTPOPULATION_H_ */
