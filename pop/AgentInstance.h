/*
 * AgentInstance.h
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#ifndef AGENTINSTANCE_H_
#define AGENTINSTANCE_H_

#include <string>
#include <vector>
#include <typeinfo>
#include <boost/any.hpp>

#include "AgentStateMemory.h"

class AgentInstance; //forward declaration


class AgentInstance {
public:
	AgentInstance(AgentStateMemory& state_memory);

	virtual ~AgentInstance();

	template <typename T> void setVariable(std::string variable_name, const T value){
		std::vector<boost::any>& v = agent_state_memory.getMemoryVector(variable_name);
		std::vector<boost::any>::iterator it = v.begin() + index;

		//type check
		const std::type_info& v_type = agent_state_memory.getVariableType(variable_name);
		if (v_type != typeid(T))
			throw std::runtime_error("Bad variable type in agent instance");
		
		//do the insert TODO
		v.insert(it, value);




	}

private:
	const unsigned int index;
	AgentStateMemory& agent_state_memory;
};

#endif /* AGENTINSTANCE_H_ */
