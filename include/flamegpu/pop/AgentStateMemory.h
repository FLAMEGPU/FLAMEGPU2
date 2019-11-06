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
#include <typeinfo>

// include generic memory vectors
#include "flamegpu/pop/MemoryVector.h"

class AgentPopulation;
class AgentDescription;

class AgentStateMemory {  // agent_list
 public:
    explicit AgentStateMemory(const AgentPopulation &population, unsigned int initial_capacity = 0);
    virtual ~AgentStateMemory() {}

    unsigned int incrementSize();

    GenericMemoryVector& getMemoryVector(const std::string variable_name);

    const GenericMemoryVector& getReadOnlyMemoryVector(const std::string variable_name) const;

    const std::type_info& getVariableType(std::string variable_name);  // const

    bool isSameDescription(const AgentDescription& description) const;

    void resizeMemoryVectors(unsigned int size);

    // the maximum number of possible agents in this population (same for all state lists)
    unsigned int getPopulationCapacity() const;

    // the actual number of agents in this state
    unsigned int getStateListSize() const;

    void overrideStateListSize(unsigned int size);

 protected:
    const AgentPopulation &population;
    const std::string agent_state;
    StateMemoryMap state_memory;
    unsigned int current_size;
};

#endif  // INCLUDE_FLAMEGPU_POP_AGENTSTATEMEMORY_H_
