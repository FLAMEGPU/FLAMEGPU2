 /**
 * @file  AgentFunctionInput.h
 * @author  Paul Richmond, Mozhgan Kabiri Chimeh
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
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
