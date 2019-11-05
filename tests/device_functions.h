#ifndef TESTS_DEVICE_FUNCTIONS_H_
#define TESTS_DEVICE_FUNCTIONS_H_

#include <flamegpu/flame_api.h>

/*
 Boost unit testing does not allow test_all.cpp to be built with NVCC on windows (Linux is fine). As a result it i snecessary to built all unit test with a pure C/C++ compiler as such all device specific code muct be moved to sperate CUDA compiled modules. The only part of the FLAME GPU API which is CUDA aware is the specification of FLAME GPU functions so all functions are declared in a separate CUDA module file. An attach function is decared in the CUDA module to allow the FLAME GPU functions to be attached to model description at runtime (permittable through relocatable device code)
*/
void attach_add_func(AgentFunctionDescription& func);
void attach_subtract_func(AgentFunctionDescription& func);
void attach_input_func(AgentFunctionDescription& func);
void attach_move_func(AgentFunctionDescription& func);
void attach_stay_func(AgentFunctionDescription& func);
void attach_output_func(AgentFunctionDescription& func);

/**
 * test_actor_random.h
 */
void attach_random1_func(AgentFunctionDescription *const func);
void attach_random2_func(AgentFunctionDescription *const func);

#endif  //TESTS_DEVICE_FUNCTIONS_H_
