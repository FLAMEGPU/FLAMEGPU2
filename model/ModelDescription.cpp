/*
 * ModelDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 *  Last modified : 28 Nov 2016
 */

#include "ModelDescription.h"

ModelDescription::ModelDescription(const std::string model_name) : agents(), messages(), name(model_name) {}

ModelDescription::~ModelDescription() {}

const std::string ModelDescription::getName() const {
	return name;
}

void ModelDescription::addAgent(const AgentDescription &agent) {
	agents.insert(AgentMap::value_type(agent.getName(), agent));
}

void ModelDescription::addMessage(const MessageDescription &message) {
	messages.insert(MessageMap::value_type(message.getName(), message));
}

const AgentDescription& ModelDescription::getAgentDescription(const std::string agent_name) const{
	AgentMap::const_iterator iter;
	iter = agents.find(agent_name);
	if (iter == agents.end())
		throw std::runtime_error("Invalid agent memory variable");
	return iter->second;
}

const AgentMap& ModelDescription::getAgentMap() const {
	return agents;
}

/*Moz*/
const MessageMap& ModelDescription::getMessageMap() const {
	return messages;

/*Moz*/
const FunctionMap& ModelDescription::getFunctionMap() const {
	return functions;
}