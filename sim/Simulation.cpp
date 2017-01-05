/**
* @file Simulation.cpp
* @authors Paul
* @date
* @brief
*
* @see
* @warning
*/

#include <exception>

#include "Simulation.h"
#include "../model/ModelDescription.h"


Simulation::Simulation(const ModelDescription& model) : model_description(model), layers()
{
    simulation_steps = 1;
}


Simulation::~Simulation(void)
{
}

void Simulation::addFunctionToLayer(int layer, std::string function_name) // we haven't used it yet

{

    try
    {
        layers.at(layer)->addAgentFunction(function_name);
    }
    catch (const std::out_of_range&)
    {
        throw std::runtime_error("Agent layer index out of bounds!");
    }

}

/**
* @param layer number (integer type)
* @return functions' name (string vector) at a specific layer
*/
const std::vector<std::string> Simulation::getFunctionAtLayer(int layer)

{
    if (layer>=layers.size())
        throw InvalidMemoryCapacity("Function layer doesn't exists!"); // out of bound index
    else
        return layers.at(layer)->getAgentFunctions();
}

unsigned int Simulation::addSimulationLayer(SimulationLayer& layer)
{
    layers.push_back(&layer);
    return static_cast<unsigned int>(layers.size())-1;
}

void Simulation::setSimulationSteps(unsigned int steps)
{
    simulation_steps = steps;
}

const ModelDescription& Simulation::getModelDescritpion() const
{
    return model_description;
}
