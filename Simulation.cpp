#include "Simulation.h"


Simulation::Simulation(void)
{
}


Simulation::~Simulation(void)
{
}

void Simulation::addFunctionToLayer(int layer, std::string function_name){
	try{
		layers.at(layer).addAgentFunction(function_name);
	}catch (const std::out_of_range&) {
		throw std::runtime_error("Agent layer index out of bounds!");
  }
}

int Simulation::addFunctionLayer(){
	layers.push_back(SimulationLayer());
	return layers.size()-1;
}
