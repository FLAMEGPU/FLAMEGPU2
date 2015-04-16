#pragma once

#include <vector>

#include "SimulationLayer.h"

class Simulation
{
public:
	Simulation(void);
	~Simulation(void);

	void addSimulationLayer(SimulationLayer layer);
	void addFunctionToLayer(int layer, std::string function_name);
	int addFunctionLayer();

private:
	std::vector<SimulationLayer> layers;
};

