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
__device__ FLAMEGPU_AGENT_FUNCTION(output_func)
{


//    for (int i = 0; i< population.getStateMemory().getStateListSize(); i++)
//    {
//        // we should be able to print all the variable values from here
//        AgentInstance instance_s1 = population.getInstanceAt(i, "default");
//        cout<<instance_s1.getVariable<int>("x");
//    }

    printf("Hello from output_func\n");

//    int x = FLAMEGPU->getVariable<int>("x");
//    cout << "x = " << x << endl;
//    FLAMEGPU->setVariable<int>("x", x*10);

    return ALIVE;
}

__device__ FLAMEGPU_AGENT_FUNCTION(input_func)
{
    printf("Hello from input_func\n");
    return ALIVE;
}

__device__ FLAMEGPU_AGENT_FUNCTION(move_func)
{
    printf("Hello from move_func\n");
    return ALIVE;
}

__device__ FLAMEGPU_AGENT_FUNCTION(stay_func)
{
    printf("Hello from stay_func\n");
    return ALIVE;
}
