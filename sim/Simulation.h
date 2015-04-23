#pragma once

#include <vector>
#include <string>

#include "SimulationLayer.h"
#include "Simulation.h"
#include "../model/ModelDescription.h"

class Simulation
{
public:
	Simulation(const ModelDescription& model);
	~Simulation(void);

	unsigned int addSimulationLayer(SimulationLayer& layer);
	void addFunctionToLayer(int layer, std::string function_name);
	void setSimulationSteps(unsigned int steps);
	const ModelDescription& getModelDescritpion() const;

private:
	std::vector<SimulationLayer*> layers;
	const ModelDescription& model_description;
	unsigned int simulation_steps;
};

