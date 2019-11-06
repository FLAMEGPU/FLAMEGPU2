#ifndef TESTS_TEST_GPU_VALIDATION_H_
#define TESTS_TEST_GPU_VALIDATION_H_
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

#include "device_functions.h"

#include "flamegpu/flame_api.h"

BOOST_AUTO_TEST_SUITE(GPUTest)  // name of the test suite is GPUTest

/**
 * @brief      To verify the correctness of set and get variable function without
 * simulating any function
 *
 * To ensure initial values for agent population is transferred correctly onto
 * the GPU, this test compares the values copied back from the device with the initial
 * population data.
 * To test the case separately, run: make run_BOOST_TEST TSuite=GPUTest/GPUMemoryTest
*/
BOOST_AUTO_TEST_CASE(GPUMemoryTest) {
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");


    circle_agent.addAgentVariable<int>("id");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);
    for (int i = 0; i< 100; i++) {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<int>("id", i);
    }

    CUDAAgentModel cuda_model(flame_model);
    cuda_model.setInitialPopulationData(population);

    cuda_model.getPopulationData(population);

    BOOST_TEST_MESSAGE("\nTesting values copied back from device without simulating any functions ..");

    // check values are the same
    for (int i = 0; i < 10; i++) {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        // use AgentInstance equality operator
        BOOST_CHECK(i1.getVariable<int>("id") == i);
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
BOOST_AUTO_TEST_CASE(GPUSimulationTest) {
    // create  single FLAME GPU model and agent
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    // test requires only a  single agent variable
    circle_agent.addAgentVariable<double>("x");


    AgentFunctionDescription add_data("add_data");
    attach_add_func(add_data);
    circle_agent.addAgentFunction(add_data);

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 10);
    for (int i = 0; i< 10; i++) {
        AgentInstance instance = population.getNextInstance("default");
        // The value of x will be the index of the agent
        instance.setVariable<double>("x", i);
    }

    BOOST_TEST_MESSAGE("\nTesting initial values ..");

    Simulation simulation(flame_model);

    SimulationLayer add_layer(simulation, "add_layer");
    add_layer.addAgentFunction("add_data");

    simulation.addSimulationLayer(add_layer);

    // simulation.setSimulationSteps(10);

    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population);

    cuda_model.simulate(simulation);

    BOOST_TEST_MESSAGE("\nTesting values copied back from device after simulating functions ..");

    // Re-use the same population to read back the simulation step results
    cuda_model.getPopulationData(population);

    // check values are the same
    for (int i = 0; i < 10; i++) {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        // use AgentInstance equality operator
        BOOST_CHECK(i1.getVariable<double>("x") == i + 2);
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
BOOST_AUTO_TEST_CASE(GPUSimulationTestMultiple) {
    /* Multi agent model */
    ModelDescription flame_model("circles_model");

    AgentDescription circle1_agent("circle1");
    circle1_agent.addAgentVariable<double>("x");

    AgentDescription circle2_agent("circle2");
    circle2_agent.addAgentVariable<double>("x");
    circle2_agent.addAgentVariable<double>("y");

    AgentFunctionDescription add_data("add_data");

    attach_add_func(add_data);
    circle1_agent.addAgentFunction(add_data);

    AgentFunctionDescription subtract_data("subtract_data");

    attach_subtract_func(subtract_data);
    circle2_agent.addAgentFunction(subtract_data);


    flame_model.addAgent(circle1_agent);
    flame_model.addAgent(circle2_agent);

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

    Simulation simulation(flame_model);

    // multiple functions per simulation layer (from different agents)
    SimulationLayer concurrent_layer(simulation, "concurrent_layer");
    concurrent_layer.addAgentFunction("add_data");
    concurrent_layer.addAgentFunction("subtract_data");
    simulation.addSimulationLayer(concurrent_layer);

    simulation.setSimulationSteps(1);

    /* Run the model */
    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population1);
    cuda_model.setInitialPopulationData(population2);

    cuda_model.simulate(simulation);

    cuda_model.getPopulationData(population1);
    cuda_model.getPopulationData(population2);


    // check values are the same
    for (int i = 0; i < SIZE; i++) {
        AgentInstance i1 = population1.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");

        // use AgentInstance equality operator
        BOOST_CHECK(i1.getVariable<double>("x") == i + 2);
        BOOST_CHECK(i2.getVariable<double>("y") == 0);
    }
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TESTS_TEST_GPU_VALIDATION_H_
