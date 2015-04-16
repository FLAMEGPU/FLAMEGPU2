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
	AgentFunctionInput(std::string message_name) { this->message_name = message_name; }
	virtual ~AgentFunctionInput() {}

	std::string getMessageName() const;

	void setMessageName(std::string message_name);

private:
	std::string message_name;

//input type (single or optional)
};

#endif /* AGENTFUNCTIONINPUT_H_ */
