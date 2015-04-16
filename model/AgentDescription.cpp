/*
 * AgentDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "AgentDescription.h"


void AgentDescription::setName(std::string name) {
}

std::string AgentDescription::getName() const{
	return name;
}

void AgentDescription::addState(std::string state_name, AgentStateDescription *state, bool initial_state) {

	//check if this is a stateless system
	if (stateless){
		states.release();
		stateless = false;
	}

	states.insert(state_name, state);
	if (initial_state)
		this->initial_state = state_name;
}

void AgentDescription::setInitialState(std::string state_name) {
	initial_state = state_name;
}

void AgentDescription::addAgentFunction(const AgentFunctionDescription& function) {

	functions.insert(function.getName(), new AgentFunctionDescription(function));
}

MemoryMap& AgentDescription::getMemoryMap() {
	return memory;
}


const MemoryMap& AgentDescription::getMemoryMap() const {
	return memory;
}


unsigned int AgentDescription::getMemorySize() const {
	unsigned int size = 0;
	for (TypeSizeMap::const_iterator it = sizes.begin(); it != sizes.end(); it++){
		size += it->second;
	}
	return size;
}

unsigned int AgentDescription::getNumberAgentVariables() const{
	return memory.size();
}

const unsigned int AgentDescription::getAgentVariableSize(const std::string variable_name) const{
	//get the variable name type
	MemoryMap::const_iterator mm = memory.find(variable_name);
	if (mm == memory.end())
		throw std::runtime_error("Invalid agent memory variable");
	const std::type_info *t = &(mm->second);
	//get the type size
	TypeSizeMap::const_iterator tsm = sizes.find(t);
	if (tsm == sizes.end())
		throw std::runtime_error("Missing entry in type sizes map. Something went bad.");
	return tsm->second;
}

bool AgentDescription::requiresAgentCreation() const{
	
	//needs to search entire model for any functions with an agent output for this agent
	for (FunctionMap::const_iterator it= functions.begin(); it != functions.end(); it++){
		//if (*it->second()->)
	}

	return false;
}

const std::type_info& AgentDescription::getVariableType(std::string variable_name) {
	MemoryMap::iterator iter;
	iter = memory.find(variable_name);

	if (iter == memory.end())
		throw std::runtime_error("Invalid agent memory variable");

	return iter->second;

}
