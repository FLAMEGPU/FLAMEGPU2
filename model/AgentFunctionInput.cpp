/*
 * AgentFunctionInput.cpp
 *
 *  Created on: 4 Mar 2014
 *      Author: paul
 */

#include "AgentFunctionInput.h"

AgentFunctionInput::AgentFunctionInput(const std::string input_message_name) :  message_name(input_message_name){

}

AgentFunctionInput::~AgentFunctionInput() {}

const std::string AgentFunctionInput::getMessageName() const{
	return message_name;
}
