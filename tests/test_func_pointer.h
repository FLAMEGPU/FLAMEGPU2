#ifndef TEST_FUNC_POINTER_H_
#define TEST_FUNC_POINTER_H_

//include all host API classes (one from each module)

//FLAMEGPU_AGENT_FUNCTION(input_func);
//FLAMEGPU_AGENT_FUNCTION(output_func);
//FLAMEGPU_AGENT_FUNCTION(move_func);
//FLAMEGPU_AGENT_FUNCTION(stay_func);

__device__ FLAME_GPU_AGENT_STATUS output_func_impl(FLAMEGPU_API* FLAMEGPU)
{
   printf("Hello from output_func\n");

//    int x = FLAMEGPU->getVariable<int>("x");
//    cout << "x = " << x << endl;
//    FLAMEGPU->setVariable<int>("x", x*10);

    return ALIVE;
}
__device__ FLAME_GPU_AGENT_STATUS input_func_impl(FLAMEGPU_API* FLAMEGPU)
{
    printf("Hello from input_func\n");
    return ALIVE;
}
__device__ FLAME_GPU_AGENT_STATUS move_func_impl(FLAMEGPU_API* FLAMEGPU)
{
    printf("Hello from move_func\n");
    return ALIVE;
}
__device__ FLAME_GPU_AGENT_STATUS stay_func_impl(FLAMEGPU_API* FLAMEGPU)
{
    printf("Hello from stay_func\n");
    return ALIVE;
}

// Declaring function pointers as symbols on device
__device__ FLAMEGPU_AGENT_FUNCTION_POINTER output_func = output_func_impl;
__device__ FLAMEGPU_AGENT_FUNCTION_POINTER input_func = input_func_impl;
__device__ FLAMEGPU_AGENT_FUNCTION_POINTER move_func = move_func_impl;
__device__ FLAMEGPU_AGENT_FUNCTION_POINTER stay_func = stay_func_impl;


#endif /* TEST_FUNC_POINTER_H_ */

