/**
* @file test_gpu_validation.h
* @author
* @date    Feb 2017
* @brief Testing Using the Boost Unit Test Framework
*/

#include "../flame_api.h"
#include "test_func_pointer.h"



using namespace std;

BOOST_AUTO_TEST_SUITE(GPUTest) //name of the test suite is modelTest

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

    //check values are the same
    for (int i = 0; i < 100; i++)
    {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        BOOST_CHECK(i1.getVariable<int>("id") == i2.getVariable<int>("id"));
    }

}

// change file type
//BOOST_AUTO_TEST_CASE(GPUSimulationTest)
//{
//
//    ModelDescription flame_model("circles_model");
//    AgentDescription circle_agent("circle");
//
//
//    circle_agent.addAgentVariable<float>("x");
//
//
//
//    AgentFunctionDescription output_data("output_data");
//    AgentFunctionOutput output_location("location");
//    output_data.addOutput(output_location);
//    //output_data.setInitialState("state1");
//    output_data.setFunction(&output_func);
//    circle_agent.addAgentFunction(output_data);
//
//    flame_model.addAgent(circle_agent);
//
//    AgentPopulation population(circle_agent, 10);
//    for (int i = 0; i< 10; i++)
//    {
//        AgentInstance instance = population.getNextInstance("default");
//        instance.setVariable<float>("x", i);
//    }
//
//    Simulation simulation(flame_model);
//
//    SimulationLayer output_layer(simulation, "output_layer");
//    output_layer.addAgentFunction("output_data");
//
//    simulation.addSimulationLayer(output_layer);
//
//    simulation.setSimulationSteps(10);
//
//    CUDAAgentModel cuda_model(flame_model);
//
//    cuda_model.setInitialPopulationData(population);
//
//    cuda_model.addSimulation(simulation);
//
//    cuda_model.step(simulation);
//
//    cuda_model.getPopulationData(population);
//    for (int i = 0; i < 100; i++)
//    {
//        AgentInstance i1 = population.getInstanceAt(i, "default");
//        BOOST_TEST_MESSAGE( "i value is : "<< i1.getVariable<float>("x")<< "!\n");
//
//    }
//}
    BOOST_AUTO_TEST_SUITE_END()

