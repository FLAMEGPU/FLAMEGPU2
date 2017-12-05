/**
* @file AgentFunctionDescription.cpp
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#include "AgentFunctionDescription.h"
#include "AgentDescription.h"
#include "../runtime/cuRVE/curve.h"


AgentFunctionDescription::AgentFunctionDescription(const std::string function_name) : function_name(function_name), initial_state("default"), end_state("default")
{

}

AgentFunctionDescription::~AgentFunctionDescription() {}

const std::string AgentFunctionDescription::getEndState() const
{
    return end_state;
}

void AgentFunctionDescription::setEndState(const std::string end_state)
{
    this->end_state = end_state;
}

const std::string AgentFunctionDescription::getIntialState() const
{
    return initial_state;
}

void AgentFunctionDescription::setIntialState(const std::string intial_state)
{
    this->initial_state = initial_state;
}

const std::string AgentFunctionDescription::getName() const
{
    return function_name;
}

void AgentFunctionDescription::addInput(const AgentFunctionInput &input)
{
    if (inputs.size() == 0)
        inputs.insert(InputsMap::value_type(input.getMessageName(), input));
    else
        throw InvalidOperation("Agent function inpput is not empty");
}

void AgentFunctionDescription::addOutput(const AgentFunctionOutput &output)
{
    if (outputs.size() == 0)
        outputs.insert(OutputsMap::value_type(output.getMessageName(), output));
    else
        throw InvalidOperation("Agent function output is not empty");

}

const InputsMap & AgentFunctionDescription::getInput()
{
   // if (inputs != NULL)
        return inputs;

}

const OutputsMap & AgentFunctionDescription::getOutput()
{
    //if (outputs != NULL)
        return outputs;
}

/**
* sets the function pointer
* @param function pointer
* @todo raise error
*/
void AgentFunctionDescription::setFunction(FLAMEGPU_AGENT_FUNCTION_POINTER *func_p)
{
    func = func_p;
}


/**
* gets the function pointer
* @return the function pointer
*/
FLAMEGPU_AGENT_FUNCTION_POINTER* AgentFunctionDescription::getFunction() const
{
    return func;
}

void AgentFunctionDescription::setParent(AgentDescription& agent)
{
    //check to make sure it is not owened by any other object
    if (parent != NULL)
        throw InvalidOperation("This object is already owned by another agent description");

    parent = &agent;
}

const AgentDescription& AgentFunctionDescription::getParent() const
{
    if (parent==NULL)
        throw InvalidOperation("Does not belong to a model object");
    return *parent;//&(*parent);
}

void AgentFunctionDescription::setInputChild(AgentFunctionInput& input)
{
    //check to make sure it is not owened by any other object
    if (inputChild != NULL)
        throw InvalidOperation("This object is already owned by another agent description");

    inputChild = &input;
}

const AgentFunctionInput& AgentFunctionDescription::getInputChild() const
{
    if (inputChild==NULL)
        throw InvalidOperation("Does not belong to a model object");
    return *inputChild;
}

void AgentFunctionDescription::setOutputChild(AgentFunctionOutput& output)
{
    //check to make sure it is not owened by any other object
    if (outputChild != NULL)
        throw InvalidOperation("This object is already owned by another agent description");

    outputChild = &output;
}
const AgentFunctionOutput& AgentFunctionDescription::getOutputChild() const
{
    if (outputChild==NULL)
        throw InvalidOperation("Does not belong to a model object");
    return *outputChild;
}
