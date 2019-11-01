/**
* @file SimulationLayer.cpp
* @authors Paul
* @date
* @brief
*
* @see
* @warning
*/

#include "flamegpu/sim/SimulationLayer.h"
#include "flamegpu/sim/Simulation.h"
#include "flamegpu/model/ModelDescription.h"

SimulationLayer::SimulationLayer(Simulation& sim, const std::string name) : layer_name(name), simulation(sim) {

}

SimulationLayer::~SimulationLayer(void) {
}

// /**
// * @return find and execute the function
// * @note assuming we know the return type
// * example of usage: callFunc<void>("move"); <-- executes move
// */
// template<typename T,typename... Args>
// T SimulationLayer::callFunc(std::string name, Args&&... args) {
//
//    auto iter = functionPointer.find(name);
//    if (iter == functionPointer.end()) {
//        throw InvalidAgentFunc("Agent Function name not valid ");
//    }
//
//    auto v = iter->second;
//
//    auto typeCast = (T(*)(Args ...))(iter->second);
//
//    /** @todo checking if the func types are the same? */
//    return typeCast(std::forward<Args>(args)...);
// }

/**
* @param function name of type string
*/
void SimulationLayer::addAgentFunction(const std::string name) {
    bool found = false;
    AgentMap::const_iterator it;
    const AgentMap& agents = simulation.getModelDescritpion().getAgentMap();

    // check agent function exists
    for (it = agents.begin(); it != agents.end(); it++) {
        if (it->second.hasAgentFunction(name)) {
            // Search the function map for current agent to see if the agent function exists (it should do the above function has confirmed this)
            const FunctionMap& funcs = it->second.getFunctionMap();
            FunctionMap::const_iterator pos = funcs.find(name);
            // If found then add function the AgentFunctionDescription to the function vector for this layer
            if (pos != funcs.end())
                functions.push_back(pos->second);
            found = true;
            break;
        }
    }

    if (!found)
        throw InvalidAgentFunc("Unknown agent function can not be added to the Function Layer!");
}

/**
* @return FunctionDescMap type that contains a string name and AgentFunctionDescription object
* @note  may change this to add an arg indicating the layer number
*/
const FunctionDescriptionVector& SimulationLayer::getAgentFunctions() const {
    return functions;
}


