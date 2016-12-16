 /**
 * @file AgentMemory.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "AgentStateMemory.h"
#include <iostream>

AgentStateMemory::AgentStateMemory(const AgentDescription& description, const std::string state) : agent_description(description), agent_state(state), size(0)
{
	//state memory map is cloned from the agent description
	description.initEmptyStateMemoryMap(state_memory);
}

unsigned int AgentStateMemory::getSize() const
{
    return size;
}


unsigned int AgentStateMemory::creatNewInstance()
{

    //loop through the memory maps
    MemoryMap::const_iterator iter;
    const MemoryMap &m = agent_description.getMemoryMap();

    for (iter = m.begin(); iter != m.end(); iter++)
    {
        const std::string variable_name = iter->first;

		GenericAgentMemoryVector &v = getMemoryVector(variable_name);
		v.incrementVector();
    }
    return size++;


}




GenericAgentMemoryVector& AgentStateMemory::getMemoryVector(const std::string variable_name)
{
    StateMemoryMap::iterator iter;
    iter = state_memory.find(variable_name);

    if (iter == state_memory.end())
        throw InvalidAgentVar();


    return *(iter->second);
}

const GenericAgentMemoryVector& AgentStateMemory::getReadOnlyMemoryVector(const std::string variable_name) const
{
    StateMemoryMap::const_iterator iter;
    iter = state_memory.find(variable_name);

    if (iter == state_memory.end())
        throw InvalidAgentVar();

    return *(iter->second);
}

const std::type_info& AgentStateMemory::getVariableType(std::string variable_name)
{
    return agent_description.getVariableType(variable_name);
}

bool AgentStateMemory::isSameDescription(const AgentDescription& description) const
{
    return (&description == &agent_description);
}
