 /**
 * @file AgentMemory.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <iostream>

#include "AgentStateMemory.h"

#include "AgentPopulation.h"
#include "../model/AgentDescription.h"



AgentStateMemory::AgentStateMemory(const AgentPopulation &p, unsigned int initial_size) : population(p)
{
	//state memory map is cloned from the agent description
	population.getAgentDescription().initEmptyStateMemoryMap(state_memory);

	//if there is an initial population size then resize the memory vectors
	if (initial_size > 0)
		resizeMemoryVectors(initial_size);
}


void AgentStateMemory::incrementSize()
{
    //loop through the memory maps and increment the vector size by 1
	const MemoryMap &m = population.getAgentDescription().getMemoryMap();
	for (const MemoryMapPair &mmp : m){
		GenericMemoryVector &v = getMemoryVector(mmp.first);
		v.incrementVector();
	}

}


GenericMemoryVector& AgentStateMemory::getMemoryVector(const std::string variable_name)
{
    StateMemoryMap::iterator iter;
    iter = state_memory.find(variable_name);

    if (iter == state_memory.end())
        throw InvalidAgentVar();


    return *(iter->second);
}

const GenericMemoryVector& AgentStateMemory::getReadOnlyMemoryVector(const std::string variable_name) const
{
    StateMemoryMap::const_iterator iter;
    iter = state_memory.find(variable_name);

    if (iter == state_memory.end())
        throw InvalidAgentVar();

    return *(iter->second);
}

const std::type_info& AgentStateMemory::getVariableType(std::string variable_name)
{
	return population.getAgentDescription().getVariableType(variable_name);
}

bool AgentStateMemory::isSameDescription(const AgentDescription& description) const
{
	return (&description == &population.getAgentDescription());
}

void AgentStateMemory::resizeMemoryVectors(unsigned int s)
{
	//all size checking is done by the population

	MemoryMap::const_iterator iter;
	const MemoryMap &m = population.getAgentDescription().getMemoryMap();

	for (iter = m.begin(); iter != m.end(); iter++)
	{
		const std::string variable_name = iter->first;

		GenericMemoryVector &v = getMemoryVector(variable_name);
		v.resize(s);
	}
}

unsigned int AgentStateMemory::getPopulationSize() const
{
	return population.getMaximumStateListSize();
}
