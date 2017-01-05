#pragma once

#include <string>
#include <vector>

//forward declare dependencies from higher up hierarchy
class Simulation;

class SimulationLayer
{
public:
    SimulationLayer(Simulation& sim, const std::string name = "none");
    ~SimulationLayer(void);

    void addAgentFunction(const std::string function_name);

    const std::vector<std::string> getAgentFunctions();

private:
    const std::string layer_name; //not required
    Simulation &simulation;
    std::vector<std::string> functions; //function names
};

