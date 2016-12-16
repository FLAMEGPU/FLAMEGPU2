/**
 * @file AgentFunctionInput.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
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
