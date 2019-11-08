/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_func_pointer.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @brief      Example of device agent functions
 *
 * These are example device agent functions to be used for testing. These are not tests themselves but are included by most of the unit test.
 * Each function returns a ALIVE or DEAD value indicating where the agent is dead and should be removed or not.
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#ifndef TESTS_TEST_CASES_RUNTIME_TEST_FUNC_POINTER_H_
#define TESTS_TEST_CASES_RUNTIME_TEST_FUNC_POINTER_H_


#include "gtest/gtest.h"

#include "helpers/common.h"

// include all host API classes (one from each module)

// FLAMEGPU_AGENT_FUNCTION(input_func);
// FLAMEGPU_AGENT_FUNCTION(output_func);
// FLAMEGPU_AGENT_FUNCTION(move_func);
// FLAMEGPU_AGENT_FUNCTION(stay_func);

/**
 * @brief      Example device function
 *
 * @param[in]  FLAMEGPU  Pointer to FLAMEGPU_API
 *
 * @retval     FLAME_GPU_AGENT_STATUS The agent is alive
 */
__device__ FLAME_GPU_AGENT_STATUS output_func_impl(FLAMEGPU_API* FLAMEGPU) {
    printf("Hello from output_func\n");

    // should've returned error if the type was not correct. Needs type check
    float x = FLAMEGPU->getVariable<float>("x");
    printf("x = %f\n", x);
    FLAMEGPU->setVariable<float>("x", x+2);
    x = FLAMEGPU->getVariable<float>("x");
    printf("x after set = %f\n", x);

    return ALIVE;
}
/**
 * @brief      Example device function
 *
 * @param      [in] FLAMEGPU Pointer to FLAMEGPU_API
 *
 * @retval     FLAME_GPU_AGENT_STATUS The agent is alive
 */
__device__ FLAME_GPU_AGENT_STATUS input_func_impl(FLAMEGPU_API* FLAMEGPU) {
    printf("Hello from input_func\n");
    return ALIVE;
}
/**
 * @brief      Example device function
 *
 * @param      [in] FLAMEGPU Pointer to FLAMEGPU_API
 *
 * @retval     FLAME_GPU_AGENT_STATUS The agent is alive
 */
__device__ FLAME_GPU_AGENT_STATUS move_func_impl(FLAMEGPU_API* FLAMEGPU) {
    printf("Hello from move_func\n");
    return ALIVE;
}
/**
 * @brief      Example device function
 *
 * @param      [in] FLAMEGPU Pointer to FLAMEGPU_API
 *
 * @retval     FLAME_GPU_AGENT_STATUS The agent is alive
 */
__device__ FLAME_GPU_AGENT_STATUS stay_func_impl(FLAMEGPU_API* FLAMEGPU) {
    printf("Hello from stay_func\n");
    return ALIVE;
}

// // Declaring function pointers as symbols on device
// __device__ FLAMEGPU_AGENT_FUNCTION_POINTER output_func = output_func_impl;
// __device__ FLAMEGPU_AGENT_FUNCTION_POINTER input_func = input_func_impl;
// __device__ FLAMEGPU_AGENT_FUNCTION_POINTER move_func = move_func_impl;
// __device__ FLAMEGPU_AGENT_FUNCTION_POINTER stay_func = stay_func_impl;


#endif  // TESTS_TEST_CASES_RUNTIME_TEST_FUNC_POINTER_H_

