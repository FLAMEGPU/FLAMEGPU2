/*
 * AgentDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef AGENTDESCRIPTION_H_
#define AGENTDESCRIPTION_H_

#include <string>
#include <map>
#include <typeinfo>

#include "AgentStateDescription.h"
#include "AgentFunctionDescription.h"

typedef std::map<const std::string, const AgentStateDescription&> StateMap;

typedef std::map<const std::string, const AgentFunctionDescription&> FunctionMap;

typedef std::map<const std::string, const std::type_info&> MemoryMap;

typedef std::map<const std::type_info*, std::size_t> TypeSizeMap;

class AgentDescription {
public:
	AgentDescription(std::string name);

	virtual ~AgentDescription();

	void setName(std::string name);

	std::string getName() const;

	void addState(const AgentStateDescription& state, bool initial_state=false);

	void setInitialState(const std::string initial_state);

	void addAgentFunction(const AgentFunctionDescription &function);

	template <typename T> void addAgentVariable(const std::string variable_name){
		memory.insert(MemoryMap::value_type(variable_name, typeid(T)));
		sizes.insert(TypeSizeMap::value_type(&typeid(T), (unsigned int)sizeof(T)));
	}

	MemoryMap& getMemoryMap(); //TODO should be shared pointer

	const MemoryMap& getMemoryMap() const;

	const unsigned int getAgentVariableSize(const std::string variable_name) const;

	unsigned int getMemorySize() const;

	unsigned int getNumberAgentVariables() const;

	bool requiresAgentCreation() const;

	const std::type_info& getVariableType(const std::string variable_name) const;



private:
	std::string name;
	bool stateless;			//system does not use states (i.e. only has a default state)
	std::string initial_state;
	std::unique_ptr<AgentStateDescription> default_state;

	StateMap states;
	FunctionMap functions;
	MemoryMap memory;
	TypeSizeMap sizes;
};

#endif /* AGENTDESCRIPTION_H_ */
