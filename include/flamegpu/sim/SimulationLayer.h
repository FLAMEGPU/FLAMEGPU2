#pragma once

#include <string>
#include <vector>
#include <map>

// include class dependencies
// #include "../model// ModelDescription.h"
// #include "../model/AgentFunctionDescription.h"

// forward declare dependencies from higher up hierarchy
class Simulation;
class AgentFunctionDescription;



typedef std::vector<std::reference_wrapper<const AgentFunctionDescription>> FunctionDescriptionVector;

class SimulationLayer
{
public:
    SimulationLayer(Simulation& sim, const std::string name = "none");
    ~SimulationLayer(void);

    /** @ brief addAgentFunction adds a function of given name to the simulation layer
     * Will check on the model description that the agent function exists (i.e. is a function defined by some agent). Assuming it is it is added to the vector of functions for this layer.
     */
    void addAgentFunction(const std::string function_name);

    const FunctionDescriptionVector& getAgentFunctions() const; // returns a reference to a vector of agent function descriptions


private:
    const std::string layer_name; // not required: TODO: Remove
    Simulation &simulation;
    FunctionDescriptionVector functions;

};

