/*
 * AgentDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "AgentDescription.h"

AgentDescription::AgentDescription(std::string name) : states(), functions(), memory(), sizes(), default_state(new AgentStateDescription("default")){
	stateless = true;
	this->name = name;
	addState(*default_state, true);
}

AgentDescription::~AgentDescription() {

}


void AgentDescription::setName(std::string name) {
}

std::string AgentDescription::getName() const{
	return name;
}

void AgentDescription::addState(const AgentStateDescription& state, bool initial_state) {

	//check if this is a stateless system
	if (stateless){
		stateless = false;
	}

	states.insert(StateMap::value_type(state.getName(), state));
	if (initial_state)
		setInitialState(state.getName());
}

void AgentDescription::setInitialState(const std::string initial_state) {
	this->initial_state = initial_state;
}

void AgentDescription::addAgentFunction(const AgentFunctionDescription& function) {

	functions.insert(FunctionMap::value_type(function.getName(), function));
}

MemoryMap& AgentDescription::getMemoryMap() {
	return memory;
}


const MemoryMap& AgentDescription::getMemoryMap() const {
	return memory;
}

const StateMap& AgentDescription::getStateMap() const {
	return states;
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

const std::type_info& AgentDescription::getVariableType(const std::string variable_name) const{
	MemoryMap::const_iterator iter;
	iter = memory.find(variable_name);

	if (iter == memory.end())
		throw std::runtime_error("Invalid agent memory variable");

	return iter->second;

}
