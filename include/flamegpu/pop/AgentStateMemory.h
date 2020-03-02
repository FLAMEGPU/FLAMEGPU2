 /**
 * @file AgentStateMemory.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef INCLUDE_FLAMEGPU_POP_AGENTSTATEMEMORY_H_
#define INCLUDE_FLAMEGPU_POP_AGENTSTATEMEMORY_H_

#include <string>
#include <vector>
#include <map>
#include <typeindex>

// include generic memory vectors
#include "flamegpu/pop/MemoryVector.h"
#include "flamegpu/model/Variable.h"

struct AgentData;
class AgentPopulation;

class AgentStateMemory {  // agent_list
 public:
    explicit AgentStateMemory(const AgentPopulation &population, unsigned int initial_capacity = 0);
    virtual ~AgentStateMemory() {}

    /**
     * Increments 'current_size' (the number of agents, independent of the allocated memory size)
     * Initialises the new agent with default values
     * @return The new index of the new agent
     */
    unsigned int incrementSize();

    GenericMemoryVector& getMemoryVector(const std::string &variable_name);

    const GenericMemoryVector& getReadOnlyMemoryVector(const std::string &variable_name) const;

    const std::type_index& getVariableType(const std::string &variable_name) const;  // const

    bool isSameDescription(const AgentData& description) const;

    void resizeMemoryVectors(unsigned int size);

    // the maximum number of possible agents in this population (same for all state lists)
    unsigned int getPopulationCapacity() const;

    // the actual number of agents in this state
    unsigned int getStateListSize() const;

    void overrideStateListSize(unsigned int size);

    const Variable &getVariableDescription(const std::string &variable_name);

 protected:
    const AgentPopulation &population;
    const std::string agent_state;
    StateMemoryMap state_memory;
    unsigned int current_size;
};

#endif  // INCLUDE_FLAMEGPU_POP_AGENTSTATEMEMORY_H_
