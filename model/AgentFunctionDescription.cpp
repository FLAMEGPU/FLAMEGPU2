/*
 * AgentFunctionDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "AgentFunctionDescription.h"

std::string AgentFunctionDescription::getEndState() {
	return end_state;
}

void AgentFunctionDescription::setEndState(std::string end_state) {
	this->end_state = end_state;
}

std::string AgentFunctionDescription::getIntialState() {
	return initial_state;
}

void AgentFunctionDescription::setIntialState(std::string intial_state) {
	this->initial_state = initial_state;
}

std::string AgentFunctionDescription::getName() const{
	return function_name;
}

void AgentFunctionDescription::setName(std::string name) {
	this->function_name = name;
}

void AgentFunctionDescription::addInput(const AgentFunctionInput &input) {
	if (inputs.size() == 0)
		inputs.insert(input.getMessageName(), new AgentFunctionInput(input));
	//else TODO: raise error
}

void AgentFunctionDescription::addOutput(const AgentFunctionOutput &output) {
	if (inputs.size() == 0)
		outputs.insert(output.getMessageName(), new AgentFunctionOutput(output));
	//else TODO: raise error
}
