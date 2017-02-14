/**
* @file SimulationLayer.cpp
* @authors Paul
* @date
* @brief
*
* @see
* @warning
*/

#include "SimulationLayer.h"

#include "Simulation.h"
#include "../model//ModelDescription.h"


SimulationLayer::SimulationLayer(Simulation& sim, const std::string name) : simulation(sim), layer_name(name)
{
}


SimulationLayer::~SimulationLayer(void)
{
}

//void SimulationLayer::addAgentFunction(const std::string function_name)
//{
//    bool found = false;
//    AgentMap::const_iterator it;
//    const AgentMap& agents = simulation.getModelDescritpion().getAgentMap();
//
//    //check agent function exists
//    for (it = agents.begin(); it != agents.end(); it++)
//    {
//        if (it->second.hasAgentFunction(function_name))
//            found = true;
//    }
//
//    if (found)
//        functions.push_back(function_name);
//    else
//        //throw std::runtime_error("Unknown agent function!");
//        throw InvalidAgentFunc();
//}

///**
//* @return agent functions name vector (string type)
//*/
//const std::vector<std::string> SimulationLayer::getAgentFunctions()
//{
//    return functions;
//}


///**
//* @return find and execute the function
//* @note assuming we know the return type
//* example of usage: callFunc<void>("move"); <-- executes move
//*/
//template<typename T,typename... Args>
//T SimulationLayer::callFunc(std::string name, Args&&... args)
//{
//
//    auto iter = functionPointer.find(name);
//    if(iter == functionPointer.end())
//    {
//        throw InvalidAgentFunc("Agent Function name not valid ");
//    }
//
//    auto v = iter->second;
//
//    auto typeCast = (T(*)(Args ...))(iter->second);
//
//    /** @todo checking if the func types are the same? */
//    return typeCast(std::forward<Args>(args)...);
//}


/**
* @param function name of type string
*/
void SimulationLayer::addAgentFunction(const std::string name)
{
    bool found = false;
    AgentMap::const_iterator it;
    const AgentMap& agents = simulation.getModelDescritpion().getAgentMap();

    //check agent function exists
    for (it = agents.begin(); it != agents.end(); it++)
    {
        if (it->second.hasAgentFunction(name))
        {

            const FunctionMap& funcs = it->second.getFunctionMap();
            auto temp = funcs.find(name);
            if (temp != funcs.end())
                funcMap.insert(FunctionDesMap::value_type(name, temp->second ));
            found = true;
            // break;
        }
    }

    if (!found)
        throw InvalidAgentFunc("Unknown agent function!");
}


/**
* @return FunctionDescMap type that contains a string name and AgentFunctionDescription object
* @note  may change this to add an arg indicating the layer number
*/
const FunctionDesMap& SimulationLayer::getAgentFunctions()
{
    return funcMap;
}


