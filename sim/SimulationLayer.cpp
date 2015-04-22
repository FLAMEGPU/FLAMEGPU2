#include "SimulationLayer.h"


SimulationLayer::SimulationLayer(void)
{
}


SimulationLayer::~SimulationLayer(void)
{
}

void SimulationLayer::addAgentFunction(const std::string function_name){
	layers.push_back(function_name);
}
