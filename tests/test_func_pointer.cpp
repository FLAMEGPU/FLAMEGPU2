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

