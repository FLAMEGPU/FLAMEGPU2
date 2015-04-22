/*
 * AgentFunctionDescription.h
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#ifndef AGENTFUNCTIONDESCRIPTION_H_
#define AGENTFUNCTIONDESCRIPTION_H_

#include <string>
#include <map>
#include <memory>

#include "AgentFunctionInput.h"
#include "AgentFunctionOutput.h"

typedef std::map<std::string, const AgentFunctionInput&> InputsMap;
typedef std::map<std::string, const AgentFunctionOutput&> OutputsMap;

class AgentFunctionDescription {
public:

	AgentFunctionDescription(const std::string function_name);

	virtual ~AgentFunctionDescription();

	const std::string getEndState() const;

	void setEndState(const std::string end_state);

	const std::string getIntialState() const;

	void setIntialState(const std::string intial_state);

	const std::string getName() const;

	void addInput(const AgentFunctionInput &input);

	void addOutput(const AgentFunctionOutput &output);

	//todo: add agent output

public:

	const std::string function_name;
	std::string initial_state;
	std::string end_state;
	InputsMap inputs;
	OutputsMap outputs;

};

#endif /* AGENTFUNCTIONDESCRIPTION_H_ */
