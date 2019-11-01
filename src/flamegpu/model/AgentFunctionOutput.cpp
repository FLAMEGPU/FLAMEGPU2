 /**
 * @file AgentFunctionOutput.cpp
 * @authors Paul
 * @date
 * @brief
 *
 * @see
 * @warning
 */

#include <flamegpu/model/AgentFunctionOutput.h>


AgentFunctionOutput::AgentFunctionOutput(const std::string output_message_name): message_name(output_message_name) {
    setFunctionOutputType(SINGLE_MESSAGE);
}

AgentFunctionOutput::~AgentFunctionOutput() {

}

const std::string AgentFunctionOutput::getMessageName() const{
    return message_name;
}

void AgentFunctionOutput::setFunctionOutputType(FunctionOutputType _type) {
    this->type = _type;
}

FunctionOutputType AgentFunctionOutput::getFunctionOutoutType() {
    return type;
}
