/*
 * AgentFunctionOutput.cpp
 *
 *  Created on: 4 Mar 2014
 *      Author: paul
 */

#include "AgentFunctionOutput.h"


AgentFunctionOutput::AgentFunctionOutput(const std::string output_message_name): message_name(output_message_name) { 
	setFunctionOutputType(SINGLE_MESSAGE); 
}

AgentFunctionOutput::~AgentFunctionOutput() {

}

const std::string AgentFunctionOutput::getMessageName() const{
	return message_name;
}

void AgentFunctionOutput::setFunctionOutputType(FunctionOutputType type) {
	this->type = type;
}

FunctionOutputType AgentFunctionOutput::getFunctionOutoutType() {
	return type;
}
