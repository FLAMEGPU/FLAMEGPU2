/*
 * ModelDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "ModelDescription.h"

std::string ModelDescription::getName() const {
	return name;
}

void ModelDescription::setName(std::string name) {
	this->name = name;
}

void ModelDescription::addAgent(const AgentDescription &agent) {
	agents.insert(agent.getName(), new AgentDescription(agent));
}

void ModelDescription::addMessage(const MessageDescription &message) {
	messages.insert(message.getName(), new MessageDescription(message));
}

AgentDescription& ModelDescription::getAgentDescription(std::string agent_name){
	AgentMap::iterator iter;
	iter = agents.find(agent_name);
	if (iter == agents.end())
		throw std::runtime_error("Invalid agent memory variable");
	return *iter->second;
}
