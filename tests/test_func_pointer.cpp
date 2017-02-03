/**
* @file test_func_pointer.cpp
* @authors
* @date
* @brief
*
* @see
* @warning
*/

#include "../flame_api.h"
#include "test_func_pointer.h"

// Problem
FLAME_GPU_AGENT_STATUS output_func()
{

//    for (int i = 0; i< population.getStateMemory().getStateListSize(); i++)
//    {
//        // we should be able to print all the variable values from here
//        AgentInstance instance_s1 = population.getInstanceAt(i, "default");
//        cout<<instance_s1.getVariable<int>("x");
//    }
//    return ALIVE;
	printf("hello\n");
}

FLAME_GPU_AGENT_STATUS input_func(void)
{
}

FLAME_GPU_AGENT_STATUS move_func(void)
{
}
FLAME_GPU_AGENT_STATUS stay_func(void)
{
}

//TODO: Some example code of the handle class and an example function

class FLAMEGPU_agent_handle{

	template<type> getVariable(string name)
	{
		agent->getVariable<float>(agent.getName(), "x")
	}

private:
	CUDAAgent agent;
	CUDAAgentMemoryHashMap agent_memory_map;
};

FLAME_GPU_AGENT_STATUS temp_func(FLAMEGPU_agent_handle* h){

	float x = h->getVariable<float>("x")
}