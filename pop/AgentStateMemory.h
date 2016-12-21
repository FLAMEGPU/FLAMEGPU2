 /**
 * @file AgentStateMemory.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#ifndef AGENTSTATEMEMORY_H_
#define AGENTSTATEMEMORY_H_


#include <string>
#include <vector>
#include <map>
#include <typeinfo>
#include <memory>
#include <boost/any.hpp>
#include <typeinfo>

//include generic memory vectors
#include "MemoryVector.h"

class AgentPopulation;
class AgentDescription;


class AgentStateMemory  // agent_list
{
public:
	AgentStateMemory(const AgentPopulation &population, unsigned int initial_size = 0);
    virtual ~AgentStateMemory() {}

    void incrementSize();

	GenericMemoryVector& getMemoryVector(const std::string variable_name);

	const GenericMemoryVector& getReadOnlyMemoryVector(const std::string variable_name) const;

    const std::type_info& getVariableType(std::string variable_name); //const

    bool isSameDescription(const AgentDescription& description) const;

	void resizeMemoryVectors(unsigned int size);

	unsigned int getPopulationSize() const;

protected:
	const AgentPopulation &population;
    const std::string agent_state;
    StateMemoryMap state_memory;

};

#endif /* AGENTSTATEMEMORY_H_ */
