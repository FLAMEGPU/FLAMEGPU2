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
	AgentFunctionOutput(const std::string output_message_name);

	virtual ~AgentFunctionOutput();

	const std::string getMessageName() const;

	void setFunctionOutputType(FunctionOutputType type);

	FunctionOutputType getFunctionOutoutType();

private:
	const std::string message_name;
	FunctionOutputType type;
};

#endif /* AGENTFUNCTIONOUTPUT_H_ */
