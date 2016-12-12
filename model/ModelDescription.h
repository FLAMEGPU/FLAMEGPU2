/*
 * ModelDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 *  Last modified : 28 Nov 2016
 */

#ifndef MODELDESCRIPTION_H_
#define MODELDESCRIPTION_H_


#include <string>
#include <map>
#include <memory>
#include <vector>
#include <typeinfo>

#include "AgentDescription.h"
#include "MessageDescription.h"


typedef std::map<const std::string, const AgentDescription&> AgentMap;
typedef std::map<const std::string, const MessageDescription&> MessageMap;
typedef std::map<const std::string, const AgentFunctionDescription&> FunctionMap; /*Moz*/

class ModelDescription {
public:
	ModelDescription(const std::string model_name);

	virtual ~ModelDescription();

	const std::string getName() const;

	void addAgent(const AgentDescription &agent);

	void addMessage(const MessageDescription &message);

	const AgentDescription& getAgentDescription(const std::string agent_name) const;

	const AgentMap& getAgentMap() const;

	// Moz
	const MessageMap& getMessageMap() const;
	// Moz
	const FunctionMap& getFunctionMap() const;


private:
	std::string name;
	AgentMap agents;
	MessageMap messages;

	/*Moz*/
	FunctionMap functions;
};


#endif /* MODELDESCRIPTION_H_ */
