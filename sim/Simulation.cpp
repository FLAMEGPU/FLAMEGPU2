#include "Simulation.h"


Simulation::Simulation(const ModelDescription& model) : model_description(model), layers()
{
    simulation_steps = 1;
}


Simulation::~Simulation(void)
{
}

void Simulation::addFunctionToLayer(int layer, std::string function_name)
{

    layers.at(layer)->addAgentFunction(function_name);
}

unsigned int Simulation::addSimulationLayer(SimulationLayer& layer)
{
    layers.push_back(&layer);
    return layers.size()-1;
}

void Simulation::setSimulationSteps(unsigned int steps)
{
    simulation_steps = steps;
}

const ModelDescription& Simulation::getModelDescritpion() const
{
    return model_description;
}
