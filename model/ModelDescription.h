/*
 * ModelDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef MODELDESCRIPTION_H_
#define MODELDESCRIPTION_H_


#include <string>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/container/map.hpp>
#include <typeinfo>

#include "AgentDescription.h"
#include "MessageDescription.h"

typedef boost::ptr_map<std::string, AgentDescription> AgentMap;
typedef boost::ptr_map<std::string, MessageDescription> MessageMap;

class ModelDescription {
public:
	ModelDescription(std::string name) : agents(), messages() { this->name = name;}

	virtual ~ModelDescription() {}

	std::string getName() const;

	void setName(std::string name);

	void addAgent(const AgentDescription &agent);

	void addMessage(const MessageDescription &message);

	AgentDescription& getAgentDescription(std::string agent_name);

private:
	std::string name;
	AgentMap agents;
	MessageMap messages;
};


#endif /* MODELDESCRIPTION_H_ */
