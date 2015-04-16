/*
 * AgentFunctionOutput.h
 *
 *  Created on: 4 Mar 2014
 *      Author: paul
 */

#ifndef AGENTFUNCTIONOUTPUT_H_
#define AGENTFUNCTIONOUTPUT_H_

#include <string>

typedef enum{
	SINGLE_MESSAGE,
	OPTIONAL_MESSAGE
} FunctionOutputType;

class AgentFunctionOutput {
public:
	AgentFunctionOutput(std::string message_name) { this->message_name = message_name; type = SINGLE_MESSAGE; }
	virtual ~AgentFunctionOutput() {}

	void setMessageName(std::string message_name);

	std::string getMessageName() const;

	void setFunctionOutputType(FunctionOutputType type);

	FunctionOutputType getFunctionOutoutType();

private:
	std::string message_name;
	FunctionOutputType type;
};

#endif /* AGENTFUNCTIONOUTPUT_H_ */
