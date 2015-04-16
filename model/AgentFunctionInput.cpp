/*
 * AgentFunctionInput.cpp
 *
 *  Created on: 4 Mar 2014
 *      Author: paul
 */

#include "AgentFunctionInput.h"


std::string AgentFunctionInput::getMessageName() const{
	return message_name;
}

void AgentFunctionInput::setMessageName(std::string message_name) {
	this->message_name = message_name;
}
