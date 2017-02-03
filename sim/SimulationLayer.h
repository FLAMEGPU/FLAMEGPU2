#pragma once

#include <string>
#include <vector>
#include <map>

//include class dependencies
//#include "../model//ModelDescription.h"
//#include "../model/AgentFunctionDescription.h"

//forward declare dependencies from higher up hierarchy
class Simulation;
class AgentFunctionDescription;



//This maps a string function name to a description of the function which also includes a pointer to the actually execute
typedef std::map<const std::string, const AgentFunctionDescription&> FunctionDesMap; //maps function name to function description

class SimulationLayer
{
public:
    SimulationLayer(Simulation& sim, const std::string name = "none");
    ~SimulationLayer(void);

    //Option 1
    void addAgentFunction(const std::string function_name);
    //using the function name look through the Model description to examine each agent and see if the function name belongs to any of the agents
    //if we dont find the function name then error
    //if we do find the function name then add to funcMap a map between the name and the AgentFunctionDescription (we should first check that the function pointer is valid (not null))

    //const std::vector<std::string> getAgentFunctions();


	//
	const FunctionDesMap& getAgentFunctionsInLayer();

    // template<typename T,typename... Args>
    //T callFunc(std::string s1, Args&&... args);

    //void executeAll();

private:
    const std::string layer_name; //not required
    Simulation &simulation;
    FunctionDesMap funcMap;

};

