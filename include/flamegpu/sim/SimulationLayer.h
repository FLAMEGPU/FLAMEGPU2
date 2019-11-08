#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>

// include class dependencies
// #include "flamegpu/model/ModelDescription.h"
// #include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/runtime/flamegpu_host_api.h"

// forward declare dependencies from higher up hierarchy
class Simulation;
class AgentFunctionDescription;


class SimulationLayer {
 public:
    typedef std::vector<std::reference_wrapper<const AgentFunctionDescription>> FunctionDescriptionVector;
    typedef std::set<FLAMEGPU_HOST_FUNCTION_POINTER> HostFunctionSet;

    explicit SimulationLayer(Simulation& sim, const std::string name = "none");
    ~SimulationLayer(void);

    /** @ brief addAgentFunction adds a function of given name to the simulation layer
     * Will check on the model description that the agent function exists (i.e. is a function defined by some agent). Assuming it is it is added to the vector of functions for this layer.
     */
    void addAgentFunction(const std::string function_name);
    /**
     * Adds a host function to the layer
     * @param func_p Pointer to the desired host function
     */
    void addHostFunction(const FLAMEGPU_HOST_FUNCTION_POINTER *func_p);
    /**
     * returns a reference to a vector of agent function descriptions
     */
    const FunctionDescriptionVector& getAgentFunctions() const;
    /**
    * Returns the set of Host functions attached to the layer
    */
    const HostFunctionSet& getHostFunctions() const;

 private:
    const std::string layer_name;  // not required: TODO: Remove
    Simulation &simulation;
    FunctionDescriptionVector functions;
    HostFunctionSet hostFunctions;
};

