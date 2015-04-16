/*
 * AgentFunctionOutput.cpp
 *
 *  Created on: 4 Mar 2014
 *      Author: paul
 */

#include "AgentFunctionOutput.h"



void AgentFunctionOutput::setMessageName(std::string message_name) {
	this->message_name = message_name;
}

std::string AgentFunctionOutput::getMessageName() const{
	return message_name;
}

void AgentFunctionOutput::setFunctionOutputType(FunctionOutputType type) {
	this->type = type;
}

FunctionOutputType AgentFunctionOutput::getFunctionOutoutType() {
	return type;
}
