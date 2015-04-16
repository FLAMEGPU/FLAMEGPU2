/*
 * MessageDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "MessageDescription.h"

std::string MessageDescription::getName() const{
	return name;
}

void MessageDescription::setName(std::string name) {
	this->name = name;
}
