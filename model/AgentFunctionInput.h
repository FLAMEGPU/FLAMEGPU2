/*
 * AgentFunctionInput.h
 *
 *  Created on: 4 Mar 2014
 *      Author: paul
 */

#ifndef AGENTFUNCTIONINPUT_H_
#define AGENTFUNCTIONINPUT_H_

#include <string>

class AgentFunctionInput {
public:
	AgentFunctionInput(const std::string input_message_name);
	virtual ~AgentFunctionInput();

	const std::string getMessageName() const;

private:
	const std::string message_name;

//input type (single or optional)
};

#endif /* AGENTFUNCTIONINPUT_H_ */
