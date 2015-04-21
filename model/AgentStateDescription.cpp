/*
 * AgentStateDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "AgentStateDescription.h"


AgentStateDescription::AgentStateDescription(const std::string state_name) : name(state_name){ 
	
}

AgentStateDescription::~AgentStateDescription() {

}

const std::string AgentStateDescription::getName() const {
	return name;
}
