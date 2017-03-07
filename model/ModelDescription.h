
#ifndef MODELDESCRIPTION_H_
#define MODELDESCRIPTION_H_

/**
 * @file ModelDescription.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
 */

#include <string>
#include <map>
#include <memory>
#include <vector>
#include <typeinfo>

//include class dependencies
#include "AgentDescription.h"
#include "MessageDescription.h"



typedef std::map<const std::string, const AgentDescription&> AgentMap;
typedef std::map<const std::string, const MessageDescription&> MessageMap;


class ModelDescription {
public:
	ModelDescription(const std::string model_name);

	virtual ~ModelDescription();

	const std::string getName() const;

	void addAgent(const AgentDescription &agent);

	void addMessage(const MessageDescription &message);

	const AgentDescription& getAgentDescription(const std::string agent_name) const;

	const AgentMap& getAgentMap() const;

	const MessageMap& getMessageMap() const;


private:
	std::string name;
	AgentMap agents;
	MessageMap messages;

	//function map removed. This belongs to agents.
};


#endif /* MODELDESCRIPTION_H_ */
