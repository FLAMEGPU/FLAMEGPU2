/*
 * AgentMemory.cpp
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#include <boost/container/map.hpp>

#include "AgentMemory.h"

AgentStateMemory::AgentStateMemory(AgentDescription& description, std::string state): agent_description(description), state_memory(), agent_name(description.getName()), agent_state(state), size(0) {
	MemoryMap::const_iterator iter;
	MemoryMap &m = description.getMemoryMap();

	for (iter = m.begin(); iter != m.end(); iter++){
		std::string variable_name = iter->first;
		const std::type_info& type = iter->second;

		state_memory.insert(variable_name, new std::vector<boost::any>());
	}
}

unsigned int AgentStateMemory::getSize() {
	return size;
}

void AgentStateMemory::incrementSize() {
	size++;
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
