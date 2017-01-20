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

void SimulationLayer::addAgentFunction(const std::string function_name)
{
    bool found = false;
    AgentMap::const_iterator it;
    const AgentMap& agents = simulation.getModelDescritpion().getAgentMap();

    //check agent function exists
    for (it = agents.begin(); it != agents.end(); it++)
    {
        if (it->second.hasAgentFunction(function_name))
            found = true;
    }

    if (found)
        functions.push_back(function_name);
    else
        //throw std::runtime_error("Unknown agent function!");
        throw InvalidAgentFunc();
}

/**
* @return agent functions name vector (string type)
*/

const std::vector<std::string> SimulationLayer::getAgentFunctions()
{
    return functions;
}

/**
* @param function name of type strinng and a function pointer
* @note do we need to make a pair of its type to check against it later? To ensure that the function is of the correct type? (e.g: using std::type_index(typeid(func));)
* @warning no error handling
* example of usage: addAgentFunctionP("move",move); <-- executes move
*/
//template<typename T>
//void SimulationLayer::addAgentFunctionP(const std::string name, T func)
//{
//    functionPointer.insert(std::make_pair(name, (fp)func));
//}

/**
* @return fpMap type that contains a string name and a function pointer
*/
const fpMap& SimulationLayer::getAgentFuncPointers()
{
    return functionPointer;
}

/**
* @return find and execute the function
* @note assuming we know the return type
* example of usage: callFunc<void>("move"); <-- executes move
*/
template<typename T,typename... Args>
T SimulationLayer::callFunc(std::string name, Args&&... args)
{

    auto iter = functionPointer.find(name);
    if(iter == functionPointer.end())
    {
        throw InvalidAgentFunc("Agent Function name not valid ");
    }

    auto v = iter->second;

    auto typeCast = (T(*)(Args ...))(iter->second);

    /** @todo checking if the func types are the same? */
    return typeCast(std::forward<Args>(args)...);
}

void SimulationLayer::addAgentFunction(const std::string name, FLAMEGPU_AGENT_FUNCTION funcp)
{
    bool found = false;
    AgentMap::const_iterator it;
    const AgentMap& agents = simulation.getModelDescritpion().getAgentMap();

    //check agent function exists
    for (it = agents.begin(); it != agents.end(); it++)
    {
        if (it->second.hasAgentFunction(name)){
            functionPointer.insert(std::make_pair(name, (FLAMEGPU_AGENT_FUNCTION)funcp));
            addAgentFunctionP(it->second,funcp);
            found = true;
            break;
            }
    }

    if (!found)
        //throw std::runtime_error("Unknown agent function!");
        throw InvalidAgentFunc();
}

void SimulationLayer::addAgentFunctionP(const &AgentFunctionDescription func, FLAMEGPU_AGENT_FUNCTION funcp){

    funcpMap.insert(std::make_pair(func,(FLAMEGPU_AGENT_FUNCTION)funcp);

    //1) add a reference to the function descritption (from model agentdescription object) to the FunctionMap
    //2) and add function pointer to the fpMap
}

void SimulationLayer::addAgentFunctionDesc(const &AgentFunctionDescription func, const std::string name){

    funcMap.insert(std::make_pair(name,func));

    //1) add a reference to the function descritption (from model agentdescription object) to the FunctionMap
    //2) and add function pointer to the fpMap
}

void SimulationLayer::executeAll()
{


}
