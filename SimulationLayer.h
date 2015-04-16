#pragma once

#include <string>
#include <vector>

class SimulationLayer
{
public:
	SimulationLayer(void);
	~SimulationLayer(void);

	void addAgentFunction(const std::string function_name);

private:
	std::vector<const std::string> layers; //function names
};

