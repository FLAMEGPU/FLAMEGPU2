/*
 * AgentInstance.h
 *
 *  Created on: 5 Mar 2014
 *      Author: paul
 */

#ifndef AGENTINSTANCE_H_
#define AGENTINSTANCE_H_

#include <iostream>
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
		//todo check that the variable exists

		GenericAgentMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);

		//type check
		const std::type_info& v_type = v.getType();
		if (v_type != typeid(T))
			//throw std::runtime_error("Bad variable type in agent instance set variable");
			throw InvalidVarType();

		//must cast the vector as th correct type (not sure if this is legal)
		std::vector<T> *t_v = static_cast<std::vector<T>*>(v.getVectorPtr());
		std::vector<T>::iterator it = t_v->begin() + index;

		//do the insert
		t_v->insert(it, value);
	}

	template <typename T>  const T getVariable(std::string variable_name){

		//todo check that the variable exists
		GenericAgentMemoryVector& v = agent_state_memory.getMemoryVector(variable_name);

		//type check
		const std::type_info& v_type = agent_state_memory.getVariableType(variable_name);
		if (v_type != typeid(T))
			//throw std::runtime_error("Bad variable type in agent instance get variable");
			throw InvalidVarType();

		//must cast the vector as th correct type (not sure if this is legal)
		std::vector<T> *t_v = static_cast<std::vector<T>*>(v.getVectorPtr());
		std::vector<T>::iterator it = t_v->begin() + index;

		//todo error handling around the cast to check for exceptions
		return t_v->at(index);
	}

private:
	const unsigned int index;
	AgentStateMemory& agent_state_memory;
};

#endif /* AGENTINSTANCE_H_ */
