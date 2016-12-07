/*
 * AgentMemory.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include "AgentStateMemory.h"

AgentStateMemory::AgentStateMemory(const AgentDescription& description, const std::string state): agent_description(description), state_memory(), agent_state(state), size(0) {
	MemoryMap::const_iterator iter;
	const MemoryMap &m = description.getMemoryMap();

	for (iter = m.begin(); iter != m.end(); iter++){
		const std::string variable_name = iter->first;
		const std::type_info& type = iter->second;

		state_memory.insert(StateMemoryMap::value_type(variable_name, std::unique_ptr<std::vector<boost::any>> (new std::vector<boost::any>())));
	}

}

unsigned int AgentStateMemory::getSize() const{
	return size;

}

void AgentStateMemory::incrementSize() { // loop through state map and set the values to zero
	size++;


	for (iter = m.begin(); iter != m.end(); iter++){
		const std::string variable_name = iter->first;
        std::vector<boost::any>& sm = getMemoryVector(variable_name);
        //sm.insert(size, 0); // check

	}

}

std::vector<boost::any>& AgentStateMemory::getMemoryVector(const std::string variable_name) {
	StateMemoryMap::iterator iter;
	iter = state_memory.find(variable_name);

	if (iter == state_memory.end())
		throw std::runtime_error("Invalid agent memory variable");

	return *iter->second;
}

const std::vector<boost::any>& AgentStateMemory::getReadOnlyMemoryVector(const std::string variable_name) const {
	StateMemoryMap::const_iterator iter;
	iter = state_memory.find(variable_name);

	if (iter == state_memory.end())
		throw std::runtime_error("Invalid agent memory variable");

	return *iter->second;
}

const std::type_info& AgentStateMemory::getVariableType(std::string variable_name) {
		return agent_description.getVariableType(variable_name);
}

bool AgentStateMemory::isSameDescription(const AgentDescription& description) const{
	return (&description == &agent_description);
}
