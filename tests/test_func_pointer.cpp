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

//singleton class (only one of them)
class FLAMEGPU_API{

	//use the has list to get the right piece of memory
	template<type> getVariable(string name)
	{
		agent->getVariable<float>(agent.getName(), "x")
	}

private:
	//data structures for hash list
};

//before this function would be called the GPU simulation engine would configure the FLAMEGPU_API object.
//This would map any variables to the hash table that the function has access to (but only these variables)
FLAME_GPU_AGENT_STATUS temp_func(FLAMEGPU_API *api){

	int x = api->getVariable<int>("x");
	x++;
	api->setVariable<int>("x", x);
}