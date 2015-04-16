/*
 * AgentFunctionDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef AGENTFUNCTIONDESCRIPTION_H_
#define AGENTFUNCTIONDESCRIPTION_H_

#include <string>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/container/map.hpp>

#include "AgentFunctionInput.h"
#include "AgentFunctionOutput.h"

typedef boost::ptr_map<std::string, AgentFunctionInput> InputsMap;
typedef boost::ptr_map<std::string, AgentFunctionOutput> OutputsMap;

class AgentFunctionDescription {
public:

	AgentFunctionDescription() {}

	AgentFunctionDescription(std::string function_name) : function_name(function_name), initial_state("default"), end_state("default") {  }


	virtual ~AgentFunctionDescription() {}

	std::string getEndState();

	void setEndState(std::string end_state);

	std::string getIntialState();

	void setIntialState(std::string intial_state);

	std::string getName() const;

	void setName(std::string name);

	void addInput(const AgentFunctionInput &input);

	void addOutput(const AgentFunctionOutput &output);

	//todo: add agent output

public:

	std::string function_name;
	std::string initial_state;
	std::string end_state;
	InputsMap inputs;
	OutputsMap outputs;

};

#endif /* AGENTFUNCTIONDESCRIPTION_H_ */
