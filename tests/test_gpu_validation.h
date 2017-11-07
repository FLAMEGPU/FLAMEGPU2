/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_gpu_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @date       16 Oct 2017
 * @brief      Test suite for validating methods in GPU folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */


#include "../flame_api.h"

FLAMEGPU_AGENT_FUNCTION(add_func)
{
    // should've returned error if the type was not correct. Needs type check
    double x = FLAMEGPU->getVariable<double>("m");

    FLAMEGPU->setVariable<double>("m", x + 2);
    x = FLAMEGPU->getVariable<double>("m");


    return ALIVE;
}

using namespace std;

BOOST_AUTO_TEST_SUITE(GPUTest) //name of the test suite is modelTest


/**
 * @brief      To verify the correctness of set and get variable function without
 * simulating any function
 *
 * To ensure initial values for agent population is transferred correctly onto
 * the GPU, this test compares the values copied back from the device with the initial
 * population data. This test should pass.
*/
BOOST_AUTO_TEST_CASE(GPUMemoryTest)
{
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");


    circle_agent.addAgentVariable<int>("id");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);
    for (int i = 0; i< 100; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<int>("id", i);
    }

    CUDAAgentModel cuda_model(flame_model);
    cuda_model.setInitialPopulationData(population);

    AgentPopulation population2(circle_agent, 100);
    cuda_model.getPopulationData(population2);

    BOOST_TEST_MESSAGE( "\nTesting values copied back from device without simulating any functions .." );

    //check values are the same
    for (int i = 0; i < 100; i++)
    {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        BOOST_CHECK(i1.getVariable<int>("id") == i2.getVariable<int>("id"));
    }

}

/**
 * @brief      To verify the correctness of ::AgentInstance::setVariable  and
 *  ::AgentInstance::getVariable variable function after simulating a function
 *
 * To ensure initial values for agent population is transferred correctly onto
 * the GPU, this test checks the correctness of the values copied back from the device
 * after being updated/changed during the simulation of an agent function.
 * This test should pass.
*/
//and hashing however every 4th variable in the array is updated
BOOST_AUTO_TEST_CASE(GPUSimulationTest)
{

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");


    circle_agent.addAgentVariable<double>("m");

    AgentFunctionDescription add_data("add_data");
    AgentFunctionOutput add_location("location");
    add_data.addOutput(add_location);
    add_data.setFunction(&add_func);
    circle_agent.addAgentFunction(add_data);

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent,10);
    for (int i = 0; i< 10; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<double>("m", i);
    }

    BOOST_TEST_MESSAGE( "\nTesting initial values .." );

//    for (int i = 0; i< 10; i++)
//    {
//        AgentInstance instance = population.getInstanceAt(i,"default");
//        BOOST_TEST_MESSAGE( i << "th value is : "<< instance.getVariable<double>("m")<< "!");
//    }

    Simulation simulation(flame_model);

    SimulationLayer add_layer(simulation, "add_layer");
    add_layer.addAgentFunction("add_data");

    simulation.addSimulationLayer(add_layer);

    //simulation.setSimulationSteps(10);

    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population);

    cuda_model.addSimulation(simulation);

    cuda_model.step(simulation);

    BOOST_TEST_MESSAGE( "\nTesting values copied back from device after simulating functions .." );


    AgentPopulation population2(circle_agent, 10);
   cuda_model.getPopulationData(population2);
//
//    for (int i = 0; i < 10; i++)
//    {
//        AgentInstance i1 = population2.getInstanceAt(i, "default");
//        BOOST_TEST_MESSAGE( i << "th value is : "<< i1.getVariable<double>("m")<< "!");
//
//    }

    //check values are the same
    for (int i = 0; i < 10; i++)
    {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        BOOST_CHECK(i1.getVariable<double>("m") + 2 == i2.getVariable<double>("m"));
    }
}

/**
 * @brief      To verify the correctness of running multiple functions simultaneously
 *
 * To test CUDA streams for overlapping host and device operations
 * This test should pass.
*/
BOOST_AUTO_TEST_CASE(GPUSimulationTest)
{

 /* Multi agent model */
    ModelDescription flame_model("circles_model");

    AgentDescription circle1_agent("circle1");
    circle1_agent.addAgentVariable<float>("x");
    circle1_agent.addAgentVariable<float>("y");

    AgentDescription circle2_agent("circle2");
    circle2_agent.addAgentVariable<float>("x");
    circle2_agent.addAgentVariable<float>("y");

    AgentFunctionDescription add_data("add_data");
    //add_data.addInput(input_location);
    add_data.setFunction(&add_func);
    circle1_agent.addAgentFunction(add_data);

    AgentFunctionDescription subtract_data("subtract_data");
    //subtract_data.addInput(input_location);
    subtract_data.setFunction(&subtract_func);
    circle2_agent.addAgentFunction(subtract_data);


    flame_model.addAgent(circle1_agent);
    flame_model.addAgent(circle2_agent);

    #define SIZE 10
    AgentPopulation population1(circle1_agent, SIZE);
    for (int i=0; i< SIZE; i++)
    {
        AgentInstance instance = population1.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
        instance.setVariable<float>("y", i*0.1f);
    }

    AgentPopulation population2(circle2_agent, SIZE);
    for (int i=0; i< SIZE; i++)
    {
        AgentInstance instance = population2.getNextInstance("default");
        instance.setVariable<float>("x", i*0.2f);
        instance.setVariable<float>("y", i*0.2f);
    }

    Simulation simulation(flame_model);

    //multiple functions per simulation layer (from different agents)
    SimulationLayer concurrent_layer(simulation, "concurrent_layer");
    concurrent_layer.addAgentFunction("add_data");
    concurrent_layer.addAgentFunction("subtract_data");
    simulation.addSimulationLayer(concurrent_layer);

    simulation.setSimulationSteps(1);

    /* Run the model */
    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population1);
    cuda_model.setInitialPopulationData(population2);

    cuda_model.addSimulation(simulation);

    cuda_model.step(simulation);

    cuda_model.getPopulationData(population1);
    cuda_model.getPopulationData(population2);

    //check values are the same
    for (int i = 0; i < SIZE; i++)
    {
        AgentInstance i1 = population1.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        BOOST_CHECK(i1.getVariable<float>("y") == i1.getVariable<float>("x")+1);
        BOOST_CHECK(i2.getVariable<float>("y") == 0);
    }
    }

BOOST_AUTO_TEST_SUITE_END()

