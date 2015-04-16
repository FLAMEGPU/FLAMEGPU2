/*
 * AgentDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef AGENTDESCRIPTION_H_
#define AGENTDESCRIPTION_H_

#include <string>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/container/map.hpp>
#include <typeinfo>

#include "AgentStateDescription.h"
#include "AgentFunctionDescription.h"

typedef boost::ptr_map<std::string, AgentStateDescription> StateMap;
typedef boost::ptr_map<std::string, AgentFunctionDescription> FunctionMap;
typedef std::map<std::string, const std::type_info&> MemoryMap;
typedef std::map<const std::type_info*, std::size_t> TypeSizeMap;

class AgentDescription {
public:
	AgentDescription(std::string name) : states(), functions(), memory(), sizes(){
		stateless = true;
		this->name = name;
		addState("default", new AgentStateDescription("default"));
	}

	virtual ~AgentDescription() {}

	void setName(std::string name);

	std::string getName() const;

	void addState(std::string state_name, AgentStateDescription *state, bool initial_state=false);

	void setInitialState(std::string state_name);

	void addAgentFunction(const AgentFunctionDescription &function);

	template <typename T> void addAgentVariable(std::string variable_name){
		memory.insert(memory.end(), MemoryMap::value_type(variable_name, typeid(T)));
		sizes.insert(TypeSizeMap::value_type(&typeid(T), (unsigned int)sizeof(T)));
	}

	MemoryMap& getMemoryMap(); //TODO should be shared pointer

	const MemoryMap& getMemoryMap() const;

	const unsigned int getAgentVariableSize(const std::string variable_name) const;

	unsigned int getMemorySize() const;

	unsigned int getNumberAgentVariables() const;

	bool requiresAgentCreation() const;

	const std::type_info& getVariableType(std::string variable_name);



private:
	std::string name;
	bool stateless;			//system does not use states (i.e. only has a default state)
	StateMap states;
	FunctionMap functions;
	MemoryMap memory;
	TypeSizeMap sizes;
	std::string initial_state;
};

#endif /* AGENTDESCRIPTION_H_ */
