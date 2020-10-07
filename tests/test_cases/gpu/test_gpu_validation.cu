#ifndef TESTS_TEST_CASES_GPU_TEST_GPU_VALIDATION_H_
#define TESTS_TEST_CASES_GPU_TEST_GPU_VALIDATION_H_
/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_gpu_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @brief      Test suite for validating methods in GPU folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include "helpers/common.h"
#include "flamegpu/flame_api.h"

#include "gtest/gtest.h"

namespace test_gpu_validation {
FLAMEGPU_AGENT_FUNCTION(add_func, MsgNone, MsgNone) {
    // should've returned error if the type was not correct. Needs type check
    double x = FLAMEGPU->getVariable<double>("x");

    FLAMEGPU->setVariable<double>("x", x + 2);

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(subtract_func, MsgNone, MsgNone) {
    double x = FLAMEGPU->getVariable<double>("x");
    double y = FLAMEGPU->getVariable<double>("y");

    FLAMEGPU->setVariable<double>("y", x - y);

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(output_func, MsgNone, MsgNone) {
    float x = FLAMEGPU->getVariable<float>("x");
    FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");

    return ALIVE;
}

// FLAMEGPU_AGENT_FUNCTION(move_func, MsgNone, MsgNone) {
//     float x = FLAMEGPU->getVariable<float>("x");
//    FLAMEGPU->setVariable<float>("x", x + 2);
//    x = FLAMEGPU->getVariable<float>("x");
//
//    // ??
//
//
//
//    return ALIVE;
// }

FLAMEGPU_AGENT_FUNCTION(input_func, MsgNone, MsgNone) {
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(move_func, MsgNone, MsgNone) {
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(stay_func, MsgNone, MsgNone) {
    return ALIVE;
}
/**
 * @brief      To verify the correctness of set and get variable function without
 * simulating any function
 *
 * To ensure initial values for agent population is transferred correctly onto
 * the GPU, this test compares the values copied back from the device with the initial
 * population data.
 * To test the case separately, run: make run_BOOST_TEST TSuite=GPUTest/GPUMemoryTest
*/
TEST(GPUTest, GPUMemoryTest) {
    ModelDescription flame_model("circles_model");
    AgentDescription &circle_agent = flame_model.newAgent("circle");


    circle_agent.newVariable<int>("id");

    AgentPopulation population(circle_agent, 100);
    for (int i = 0; i< 100; i++) {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<int>("id", i);
    }

    CUDASimulation cuda_model(flame_model);
    cuda_model.setPopulationData(population);

    cuda_model.getPopulationData(population);

    GTEST_COUT << "Testing values copied back from device without simulating any functions .." << std::endl;

    // check values are the same
    for (int i = 0; i < 10; i++) {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        // use AgentInstance equality operator
        EXPECT_EQ(i1.getVariable<int>("id"), i);
    }
}

/**
* @brief      To verify the correctness of ::AgentInstance::setVariable  and
*  ::AgentInstance::getVariable variable function and hashing after simulating a function
*
* To ensure initial values for agent population is transferred correctly onto
* the GPU, this test checks the correctness of the values copied back from the device
* after being updated/changed during the simulation of an agent function. The 'add_function' agent function simply increases the agent variable x by a value of 2.
* This test will re-use the original population to read the results of the simulation step so it also acts to test that the original values are correctly overwritten.
*
* To test the case separately, run: make run_BOOST_TEST TSuite=GPUTest/GPUSimulationTest
*/
TEST(GPUTest, GPUSimulationTest) {
    // create  single FLAME GPU model and agent
    ModelDescription flame_model("circles_model");
    AgentDescription &circle_agent = flame_model.newAgent("circle");

    // test requires only a  single agent variable
    circle_agent.newVariable<double>("x");


    AgentFunctionDescription &add_data = circle_agent.newFunction("add", add_func);

    AgentPopulation population(circle_agent, 10);
    for (int i = 0; i< 10; i++) {
        AgentInstance instance = population.getNextInstance("default");
        // The value of x will be the index of the agent
        instance.setVariable<double>("x", i);
    }

    GTEST_COUT << "Testing initial values .." << std::endl;

    LayerDescription &add_layer = flame_model.newLayer("add_layer");
    add_layer.addAgentFunction(add_data);


    CUDASimulation cuda_model(flame_model);
    const int STEPS = 5;
    cuda_model.SimulationConfig().steps = STEPS;

    cuda_model.setPopulationData(population);

    cuda_model.simulate();

    GTEST_COUT << "Testing values copied back from device after simulating functions .." << std::endl;

    // Re-use the same population to read back the simulation step results
    cuda_model.getPopulationData(population);

    // check values are the same
    for (int i = 0; i < 10; i++) {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        // use AgentInstance equality operator
        EXPECT_EQ(i1.getVariable<double>("x"), i + (2 * STEPS));
    }
}

/**
 * @brief      To verify the correctness of running multiple functions simultaneously
 *
 * To test CUDA streams for overlapping host and device operations. This test is a test for concurrency.
 * It is expected that add and subtract functions should execute simultaneously as they are functions belonging to different agents with the functions on the same simulation layer.
 * Note: To observe that the functions are actually executing concurrently requires that you profile the test and observe the kernels in the NVProf profiler.
 * To test the case separately, run: make run_BOOST_TEST TSuite=GPUTest/GPUSimulationTestMultiple
 */
TEST(GPUTest, GPUSimulationTestMultiple) {
    /* Multi agent model */
    ModelDescription flame_model("circles_model");

    AgentDescription &circle1_agent = flame_model.newAgent("circle1");
    circle1_agent.newVariable<double>("x");

    AgentDescription &circle2_agent = flame_model.newAgent("circle2");
    circle2_agent.newVariable<double>("x");
    circle2_agent.newVariable<double>("y");

    AgentFunctionDescription &add_data = circle1_agent.newFunction("add", add_func);
    AgentFunctionDescription &subtract_data = circle2_agent.newFunction("subtract", subtract_func);


    #define SIZE 10
    AgentPopulation population1(circle1_agent, SIZE);
    for (int i = 0; i < SIZE; i++) {
        AgentInstance instance = population1.getNextInstance("default");
        instance.setVariable<double>("x", i);
    }

    AgentPopulation population2(circle2_agent, SIZE);
    for (int i=0; i< SIZE; i++) {
        AgentInstance instance = population2.getNextInstance("default");
        instance.setVariable<double>("x", i);
        instance.setVariable<double>("y", i);
    }

    // multiple functions per simulation layer (from different agents)
    LayerDescription &concurrent_layer = flame_model.newLayer("concurrent_layer");
    concurrent_layer.addAgentFunction(add_data);
    concurrent_layer.addAgentFunction(subtract_data);


    /* Run the model */
    CUDASimulation cuda_model(flame_model);
    cuda_model.SimulationConfig().steps = 1;

    cuda_model.setPopulationData(population1);
    cuda_model.setPopulationData(population2);

    cuda_model.simulate();

    cuda_model.getPopulationData(population1);
    cuda_model.getPopulationData(population2);


    // check values are the same
    for (int i = 0; i < SIZE; i++) {
        AgentInstance i1 = population1.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");

        // use AgentInstance equality operator
        EXPECT_EQ(i1.getVariable<double>("x"), i + 2);
        EXPECT_EQ(i2.getVariable<double>("y"), 0);
    }
}
    }  // namespace test_gpu_validation

#endif  // TESTS_TEST_CASES_GPU_TEST_GPU_VALIDATION_H_
