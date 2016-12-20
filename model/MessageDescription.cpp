/**
 * @file MessageDescription.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "MessageDescription.h"

MessageDescription::MessageDescription(const std::string message_name) : variables(), name(message_name) { }

MessageDescription::~MessageDescription() {}

const std::string MessageDescription::getName() const{
	return name;
}

