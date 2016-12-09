#include "Simulation.h"
#include "SimulationLayer.h"


SimulationLayer::SimulationLayer(Simulation& sim, const std::string name) : simulation(sim), layer_name(name)
{
}


SimulationLayer::~SimulationLayer(void)
{
}

void SimulationLayer::addAgentFunction(const std::string function_name){
	bool found = false;
	AgentMap::const_iterator it;
	const AgentMap& agents = simulation.getModelDescritpion().getAgentMap();

	//check agent function exists
	for (it = agents.begin(); it != agents.end(); it++){
		if (it->second.hasAgentFunction(function_name))
			found = true;
	}

	if (found)
		functions.push_back(function_name);
	else
		//throw std::runtime_error("Unknown agent function!");
		throw InvalidAgentFunc();
}
