#ifndef TESTS_HELPERS_DEVICE_TEST_FUNCTIONS_H_
#define TESTS_HELPERS_DEVICE_TEST_FUNCTIONS_H_

#include "flamegpu/flame_api.h"

/*
 Boost unit testing does not allow test_all.cpp to be built with NVCC on windows (Linux is fine). As a result it i snecessary to built all unit test with a pure C/C++ compiler as such all device specific code muct be moved to sperate CUDA compiled modules. The only part of the FLAME GPU API which is CUDA aware is the specification of FLAME GPU functions so all functions are declared in a separate CUDA module file. An attach function is decared in the CUDA module to allow the FLAME GPU functions to be attached to model description at runtime (permittable through relocatable device code)
*/
AgentFunctionDescription& attach_add_func(AgentDescription& agent);
AgentFunctionDescription& attach_subtract_func(AgentDescription& agent);
AgentFunctionDescription& attach_input_func(AgentDescription& agent);
AgentFunctionDescription& attach_move_func(AgentDescription& agent);
AgentFunctionDescription& attach_stay_func(AgentDescription& agent);
AgentFunctionDescription& attach_output_func(AgentDescription& agent);

/**
 * test_actor_random.h
 */
AgentFunctionDescription& attach_random1_func(AgentDescription& agent);
AgentFunctionDescription& attach_random2_func(AgentDescription& agent);

#endif  // TESTS_HELPERS_DEVICE_TEST_FUNCTIONS_H_
