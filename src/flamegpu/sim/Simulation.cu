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

#include <flamegpu/sim/Simulation.h>
#include <flamegpu/model/ModelDescription.h>


Simulation::Simulation(const ModelDescription& model) : model_description(model), layers()
{
    simulation_steps = 1;
}


Simulation::~Simulation(void)
{
}




const FunctionDescriptionVector& Simulation::getFunctionsAtLayer(unsigned int layer) const
{
    if (layer>=layers.size())
        throw InvalidMemoryCapacity("Function layer doesn't exists!"); // out of bound index
    else
    {
        return layers.at(layer).get().getAgentFunctions(); 
    }
}


unsigned int Simulation::addSimulationLayer(SimulationLayer& layer)
{
    layers.push_back(layer);
    return static_cast<unsigned int>(layers.size())-1;
}

void Simulation::setSimulationSteps(unsigned int steps)
{
    simulation_steps = steps;
}

unsigned int Simulation::getSimulationSteps() const
{
	return simulation_steps;
}

unsigned int Simulation::getLayerCount() const
{
    return (unsigned int) layers.size();
}

const ModelDescription& Simulation::getModelDescritpion() const
{
    return model_description;
}
