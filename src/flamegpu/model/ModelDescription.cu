 /**
 * @file ModelDescription.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <flamegpu/model/ModelDescription.h>

ModelDescription::ModelDescription(const std::string model_name) : name(model_name), agents(), messages(), population() {}

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

void ModelDescription::addPopulation(AgentPopulation &pop)
{
 	population.insert(PopulationMap::value_type(pop.getAgentName(), pop));
}

const AgentDescription& ModelDescription::getAgentDescription(const std::string agent_name) const{
	AgentMap::const_iterator iter;
	iter = agents.find(agent_name);
	if (iter == agents.end())
		throw InvalidAgentVar();
	return iter->second;
}

const MessageDescription& ModelDescription::getMessageDescription(const std::string message_name) const{
	MessageMap::const_iterator iter;
	iter = messages.find(message_name);
	if (iter == messages.end())
		throw InvalidMessageVar();
	return iter->second;
}

AgentPopulation& ModelDescription::getAgentPopulation(const std::string agent_name) const{
	PopulationMap::const_iterator iter;
	iter = population.find(agent_name);
	if (iter == population.end())
		throw InvalidAgentVar();
	return iter->second;
}

const AgentMap& ModelDescription::getAgentMap() const {
	return agents;
}

const MessageMap& ModelDescription::getMessageMap() const {
	return messages;
}
