 /**
 * @file AgentPopulation.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef INCLUDE_FLAMEGPU_POP_AGENTPOPULATION_H_
#define INCLUDE_FLAMEGPU_POP_AGENTPOPULATION_H_

#include <memory>
#include <map>
#include <utility>
#include <string>

// include sub dependency AgentInstance which includes AgentStateMemory
#include "flamegpu/pop/AgentInstance.h"
#include "flamegpu/model/ModelData.h"

// forward declarations
class AgentDescription;
struct AgentData;


class AgentPopulation {
    typedef std::map<std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMap;    // key is concat of agent and state name!
    typedef std::pair<std::string, std::unique_ptr<AgentStateMemory>> AgentStatesMapPair;

 public:
    static const unsigned int DEFAULT_POPULATION_SIZE;
    explicit AgentPopulation(const AgentDescription &agent_description, unsigned int initial_size = DEFAULT_POPULATION_SIZE);

    virtual ~AgentPopulation();

    AgentInstance getNextInstance(const std::string agent_state = ModelData::DEFAULT_STATE);

    AgentInstance getInstanceAt(unsigned int index, const std::string agent_state = ModelData::DEFAULT_STATE);

    AgentStateMemory& getStateMemory(const std::string agent_state = ModelData::DEFAULT_STATE);

    unsigned int getCurrentListSize(const std::string agent_state = ModelData::DEFAULT_STATE) const;

    const AgentStateMemory& getReadOnlyStateMemory(const std::string agent_state = ModelData::DEFAULT_STATE) const;

    std::string getAgentName() const;

    const AgentData& getAgentDescription() const;

    /* This is the maximum size of any single state list. */
    unsigned int getMaximumStateListCapacity() const;

    void setStateListCapacity(unsigned int);

 private:
    const std::shared_ptr<const AgentData> agent;
    AgentStatesMap states_map;
    unsigned int maximum_size;  // size is maximum size for agents in any single state (same for all states of same agent type)
};

#endif  // INCLUDE_FLAMEGPU_POP_AGENTPOPULATION_H_
