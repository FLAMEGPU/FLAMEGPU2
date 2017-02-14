/*
* flame_api.h
*
*  Created on: 19 Feb 2014
*      Author: paul
*/

#ifndef FLAME_FUNCTIONS_API_H_
#define FLAME_FUNCTIONS_API_H_
/*
singleton class FLAME
{
GetMessageIterator("messagename");
FLAME.getMem<type>("variablename");

};
*/

//TODO: Some example code of the handle class and an example function
//singleton class (only one of them)

class FLAMEGPU_API;  // Forward declaration (class defined below)

//! Datatype for argument of all agent transition functions
typedef FLAMEGPU_API& FLAMEGPU_AgentFunctionParamType;
//! Datatype for return value for all agent transition functions
typedef FLAME_GPU_AGENT_STATUS FLAMEGPU_AgentFunctionReturnType;

// re-defined in AgentFunctionDescription
//! FLAMEGPU function return type
enum FLAME_GPU_AGENT_STATUS { ALIVE, DEAD }; // can be 0 or 1 or same as flame, just define alive 0 and 1 dead
//#define FLAMEGPU_AGENT_ALIVE 0
//#define FLAMEGPU_AGENT_DEAD  1



//! Macro for defining agent transition functions with the correct input
//! argument and return type
#define FLAMEGPU_AGENT_FUNCTION(funcName) \
          FLAMEGPU_AgentFunctionReturnType \
          funcName(FLAMEGPU_AgentFunctionParamType FLAMEGPU)
/**
* @note Example Usage:

FLAMEGPU_AGENT_FUNCTION(move_func) {
  int x = FLAMEGPU.getVariable<int>("x");
  FLAMEGPU.setVariable<int>("x", x*10);
  return ALIVE;
}
*/



class FLAMEGPU_API
{
public:
    FLAMEGPU_API():
        //use the has list to get the right piece of memory
        template<typename T>
        T getVariable(std::string name)
    {
        agent->getVariable<T>(agent.getName(), "x")
    }

    template<typenae T>
    T setVariable(std::string name, T value)
    {
        agent->setVariable<T>(name, value)
    }

private:
    //data structures for hash list

    // from CUDAAgent
    const AgentDescription& agent_description;
    CUDAStateMap state_map;

    unsigned int* h_hashes; //host hash index table //USE SHARED POINTER??
    unsigned int* d_hashes; //device hash index table (used by runtime)

    unsigned int max_list_size; //The maximum length of the agent variable arrays based on the maximum population size passed to setPopulationData

    // from CUDAAgentStateList
    CUDAAgentMemoryHashMap d_list;
    CUDAAgentMemoryHashMap d_swap_list;
    CUDAAgentMemoryHashMap d_new_list;

    unsigned int current_list_size; //???

    CUDAAgent& agent;

};



/**
//before this function would be called the GPU simulation engine would configure the FLAMEGPU_API object.
//This would map any variables to the hash table that the function has access to (but only these variables)

*
* @note Example Usage:
FLAME_GPU_AGENT_STATUS temp_func(FLAMEGPU_API *api){

	int x = api->getVariable<int>("x");
	x++;
	api->setVariable<int>("x", x);
}
*/


#endif /* FLAME_FUNCTIONS_API_H_ */
