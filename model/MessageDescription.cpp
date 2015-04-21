/*
 * MessageDescription.cpp
 *
 *  Created on: 20 Feb 2014
 *      Author: paul
 */

#include "MessageDescription.h"

MessageDescription::MessageDescription(const std::string message_name) : variables(), name(message_name) { }

MessageDescription::~MessageDescription() {}

const std::string MessageDescription::getName() const{
	return name;
}
