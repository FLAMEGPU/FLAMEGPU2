
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
#include "../pop/AgentPopulation.h"

typedef std::map<const std::string, const AgentDescription&> AgentMap;
typedef std::map<const std::string, const MessageDescription&> MessageMap;
typedef std::map<const std::string, AgentPopulation&> PopulationMap;


class ModelDescription {
public:
	ModelDescription(const std::string model_name);

	virtual ~ModelDescription();

	const std::string getName() const;

	void addAgent(const AgentDescription &agent);

	void addMessage(const MessageDescription &message);

	void addPopulation(AgentPopulation &population);
	void initialise(const ModelDescription &model, char * input);
	void output(const ModelDescription &model, char * output);


	const AgentDescription& getAgentDescription(const std::string agent_name) const;
	const MessageDescription& getMessageDescription(const std::string message_name) const;
	AgentPopulation& getAgentPopulation(const std::string agent_name) const;

	const AgentMap& getAgentMap() const;

	const MessageMap& getMessageMap() const;

	//const PopulationMap& getPopulationMap() const;

private:
	std::string name;
	AgentMap agents;
	MessageMap messages;
	PopulationMap population;

	//function map removed. This belongs to agents.
};


#endif /* MODELDESCRIPTION_H_ */
