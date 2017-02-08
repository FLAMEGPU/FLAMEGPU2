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
//
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
* @param function name of type string and a function pointer
* @note example of usage: addAgentFunction("move",move_func);
* @warning may have errors
*/
void SimulationLayer::addAgentFunction(const std::string name, FLAMEGPU_AGENT_FUNCTION funcp)
{
    bool found = false;
    AgentMap::const_iterator it;
    const AgentMap& agents = simulation.getModelDescritpion().getAgentMap();

    //const FunctionMap& funcs = agents->second.getFunctionMap();

    //check agent function exists
    for (it = agents.begin(); it != agents.end(); it++)
    {
        if (it->second.hasAgentFunction(name))
        {
            functionPointer.insert(std::make_pair(name, (FLAMEGPU_AGENT_FUNCTION)funcp));

            const FunctionMap& funcs = it->second.getFunctionMap();
            auto temp = funcs.find(name);
            if (temp != funcs.end())
                addAgentFunctionP(temp->second,funcp);
            found = true;
            // break;
        }
    }

    if (!found)
        //throw std::runtime_error("Unknown agent function!");
        throw InvalidAgentFunc();
}

// To me
// get functionmap that returns a map of func name and agent function desc
// use above in addAgentFunction

/**
* @param AgentFunctionDescription object and a function pointer
* @note we can even not have this function and insert the values to the map in the prev function
* @note Alternatively, we could have a map of map which maps a string to the pair of function pointer and an object
* @note Or we could just use next function instead which map a string name to the AgentFunctionDescription object
* @warning no error handling
*/
void SimulationLayer::addAgentFunctionP(const AgentFunctionDescription& func, FLAMEGPU_AGENT_FUNCTION funcp)
{

    agentfpMap.insert(std::make_pair((FLAMEGPU_AGENT_FUNCTION)funcp,func));
}


///**
//* @param function name of type string and AgentFunctionDescription object
//* @note hasn't been used yet
//* @warning no error handling
//*/
//void SimulationLayer::addAgentFunctionDesc(const AgentFunctionDescription& func, const std::string name)
//{
//
//    funcMap.insert(std::make_pair(name,func));
//}

/**
* @return fpMap type that contains a string name and a function pointer
*/
const fpMap& SimulationLayer::getAgentFuncPointers()
{
    return functionPointer;
}

///**
//* @return function pointer
//* @note hasn't been used yet
//*/
//const FLAMEGPU_AGENT_FUNCTION& SimulationLayer::getfunctionPointer(){}


/**
* @return AgentFunctionMap type that contains AgentFunctionDescription object and a function pointer
*/
const AgentFunctionMap& SimulationLayer::getAgentFunctions()
{
    return agentfpMap;
}



/**
* @note hasn't been used yet
* @warning no error handling
*/
void SimulationLayer::executeAll()
{

}
