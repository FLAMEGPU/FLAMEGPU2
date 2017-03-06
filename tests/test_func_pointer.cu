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
FLAMEGPU_AGENT_FUNCTION(output_func)
{

    printf("Hello from output_func\n");

//    int x = FLAMEGPU->getVariable<int>("x");
//    cout << "x = " << x << endl;
//    FLAMEGPU->setVariable<int>("x", x*10);

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(input_func)
{
    printf("Hello from input_func\n");
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(move_func)
{
    printf("Hello from move_func\n");
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(stay_func)
{
    printf("Hello from stay_func\n");
    return ALIVE;
}
