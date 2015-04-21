/*
 * ModelDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
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
