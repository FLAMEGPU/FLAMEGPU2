#pragma once

#include <string>
#include <vector>

class Simulation;

class SimulationLayer
{
public:
	SimulationLayer(Simulation& sim, const std::string name = "none");
	~SimulationLayer(void);

	void addAgentFunction(const std::string function_name);

private:
	const std::string layer_name; //not required
	Simulation &simulation;
	std::vector<const std::string> functions; //function names
};

