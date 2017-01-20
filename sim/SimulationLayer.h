#pragma once

#include <string>
#include <vector>

//forward declare dependencies from higher up hierarchy
class Simulation;

/*! function pointer type */
typedef enum FLAME_GPU_AGENT_STATUS{ALIVE, DEAD};

typedef FLAME_GPU_AGENT_STATUS (*FLAMEGPU_AGENT_FUNCTION)(void);

//Option 1
/*! mapping a string (function name) to a function pointer */
typedef std::map<const std::string, FLAMEGPU_AGENT_FUNCTION> fpMap; // or use a vector of pairs --> std::vector<pair<const std::string, fp>> fpPair;
/*! mapping a string (function name) to AgentFunctionDescription object */
typedef std::map<const std::string, const AgentFunctionDescription&> FunctionMap; //maps function name to function description

//Option 2
/*! mapping AgentFunctionDescription object to a function pointer (FLAMEGPU_AGENT_FUNCTION) */
typedef std::map<const AgentFunctionDescription&, FLAMEGPU_AGENT_FUNCTION> FunctionPMap;

class SimulationLayer
{
public:
    SimulationLayer(Simulation& sim, const std::string name = "none");
    ~SimulationLayer(void);

    //void addAgentFunction(const std::string function_name); // replaced by below

    //Option 1
    void addAgentFunction(const std::string function_name, FLAMEGPU_AGENT_FUNCTION funcp);
    //using the function name look through the Model description to examine each agent and see if the function name belongs to any of the agents
    //if we dont find the function name then error
    //if we do find the function name then call the function below with correct arguments

    //Option 2
    void addAgentFunction(const &AgentFunctionDescription func, FLAMEGPU_AGENT_FUNCTION funcp);
    //1) add a reference to the function descritption (from model agentdescription object) to the FunctionMap
    //2) and add function pointer to the fpMap

    //const std::vector<std::string> getAgentFunctions();
    const fpMap& getAgentFuncPointers();
    const FLAMEGPU_AGENT_FUNCTION& getAgentFuncPointers();

   // template<typename T,typename... Args>
    //T callFunc(std::string s1, Args&&... args);

    void executeAll();

private:
    const std::string layer_name; //not required
    Simulation &simulation;
    std::vector<std::string> functions; //function names
    fpMap functionPointer; //function names,func pointer
    FunctionMap funcMap;
    FunctionPMap funcpMap;
    //const AgentDescription &agent; //required so that we can get the function description for the agent
};

