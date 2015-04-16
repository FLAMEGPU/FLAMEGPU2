/*
 * AgentStateDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "AgentStateDescription.h"


std::string AgentStateDescription::getName() {
	return name;
}

void AgentStateDescription::setName(std::string name) {
	this->name = name;
}
