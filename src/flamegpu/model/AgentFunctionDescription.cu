/**
* @file AgentFunctionDescription.cpp
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/runtime/cuRVE/curve.h"

AgentFunctionDescription::AgentFunctionDescription(const std::string function_name) : function_name(function_name), initial_state("default"), end_state("default") {
}

AgentFunctionDescription::~AgentFunctionDescription() {}

const std::string AgentFunctionDescription::getEndState() const {
    return end_state;
}

void AgentFunctionDescription::setEndState(const std::string _end_state) {
    this->end_state = _end_state;
}

const std::string AgentFunctionDescription::getInitialState() const {
    return initial_state;
}

void AgentFunctionDescription::setInitialState(const std::string _initial_state) {
    this->initial_state = _initial_state;
}

const std::string AgentFunctionDescription::getName() const {
    return function_name;
}

void AgentFunctionDescription::setInput(const AgentFunctionInput &input) {
    if (inputs.size() == 0) {
        inputs.insert(InputsMap::value_type(input.getMessageName(), input));
    } else {
        THROW InvalidOperation("Only one message input allowed per agent function. "
            "Agent Function ('%s') already has input ('%s'), "
            "in AgentFunctionDescription::setInput().",
            function_name.c_str(), inputs.begin()->second.getMessageName().c_str());
    }
}

void AgentFunctionDescription::setOutput(const AgentFunctionOutput &output) {
    if (outputs.size() == 0) {
        outputs.insert(OutputsMap::value_type(output.getMessageName(), output));
    } else {
        THROW InvalidOperation("Only one message output allowed per agent function. "
            "Agent Function ('%s') already has input ('%s'), "
            "in AgentFunctionDescription::setOutput().",
            function_name.c_str(), outputs.begin()->second.getMessageName().c_str());
    }
}

const InputsMap & AgentFunctionDescription::getInput() {
    // if (inputs != nullptr)
        return inputs;
}

const OutputsMap & AgentFunctionDescription::getOutput() {
    // if (outputs != nullptr)
        return outputs;
}

const std::string AgentFunctionDescription::getInputMessageName() const {
    InputsMap::const_iterator iter;
    iter = inputs.begin();

    return iter->first;
}

const std::string AgentFunctionDescription::getOutputMessageName() const {
    OutputsMap::const_iterator iter;
    iter = outputs.begin();

    return iter->first;
}

/**
* sets the function pointer
* @param function pointer
* @todo raise error
*/
void AgentFunctionDescription::setFunction(FLAMEGPU_AGENT_FUNCTION_POINTER *func_p) {
    func = func_p;
}

/**
* gets the function pointer
* @return the function pointer
*/
FLAMEGPU_AGENT_FUNCTION_POINTER* AgentFunctionDescription::getFunction() const {
    return func;
}

void AgentFunctionDescription::setParent(AgentDescription& agent) {
    // check to make sure it is not owened by any other object
    if (parent != nullptr) {
        THROW InvalidOperation("Agent function can only belong to one agent. "
            "Agent Function ('%s') already attached to agent ('%s'), "
            "in AgentFunctionDescription::setParent().",
            function_name.c_str(), parent->getName().c_str());
    }
    parent = &agent;
}

const AgentDescription& AgentFunctionDescription::getParent() const {
    if (parent == nullptr) {
        THROW InvalidOperation("Agent Function ('%s') does not belong to an agent, "
            "in AgentFunctionDescription::getParent().",
            function_name.c_str());
    }
    return *parent;  // &(*parent);
}

void AgentFunctionDescription::setInputChild(AgentFunctionInput& input) {
    // check to make sure it is not owened by any other object
    if (inputChild != nullptr) {
        THROW InvalidOperation("Only one message input allowed per agent function. "
            "Agent Function ('%s') already has input ('%s'), "
            "in AgentFunctionDescription::setInputChild().",
            function_name.c_str(), inputChild->getMessageName().c_str());
    }
    inputChild = &input;
}

const AgentFunctionInput& AgentFunctionDescription::getInputChild() const {
    if (inputChild == nullptr) {
        THROW InvalidOperation("Agent Function ('%s') does not have message input, "
            "in AgentFunctionDescription::getInputChild().",
            function_name.c_str());
    }
    return *inputChild;
}

void AgentFunctionDescription::setOutputChild(AgentFunctionOutput& output) {
    // check to make sure it is not owened by any other object
    if (outputChild != nullptr) {
        THROW InvalidOperation("Only one message output allowed per agent function. "
            "Agent Function ('%s') already has input ('%s'), "
            "in AgentFunctionDescription::setOutputChild().",
            function_name.c_str(), outputChild->getMessageName().c_str());
    }
    outputChild = &output;
}
const AgentFunctionOutput& AgentFunctionDescription::getOutputChild() const {
    if (outputChild == nullptr) {
        THROW InvalidOperation("Agent Function ('%s') does not have message output, "
            "in AgentFunctionDescription::getOutputChild().",
            function_name.c_str());
    }
    return *outputChild;
}

bool AgentFunctionDescription::hasInputMessage() const {
    return !(inputs.empty());
}
bool AgentFunctionDescription::hasOutputMessage() const {
    return !(outputs.empty());
}

