/**
 * @file AgentFunctionInput.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include "flamegpu/model/AgentFunctionInput.h"

AgentFunctionInput::AgentFunctionInput(const std::string input_message_name) :  message_name(input_message_name) {
}

AgentFunctionInput::~AgentFunctionInput() {}

const std::string AgentFunctionInput::getMessageName() const {
    return message_name;
}
