/**
 * @file test_gpu_validation.h
 * @brief Testing Using the Boost Unit Test Framework
 */
#include "../flame_api.h"


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
	for (int i = 0; i < 100; i++){
		AgentInstance i1 = population.getInstanceAt(i, "default");
		AgentInstance i2 = population2.getInstanceAt(i, "default");
		//use AgentInstance equality operator
		BOOST_CHECK(i1.getVariable<int>("id") == i2.getVariable<int>("id"));
	}

}
//
//BOOST_AUTO_TEST_CASE(GPUSimulationTest)
//{
//
//    ModelDescription flame_model("circles_model");
//    AgentDescription circle_agent("circle");
//
//
//	circle_agent.addAgentVariable<int>("id");
//
//    flame_model.addAgent(circle_agent);
//
//	AgentPopulation population(circle_agent, 100);
//	for (int i = 0; i< 100; i++)
//	{
//		AgentInstance instance = population.getNextInstance("default");
//		instance.setVariable<int>("id", i);
//	}
//
//
//    CUDAAgentModel cuda_model(flame_model);
//    cuda_model.setInitialPopulationData(population);
//
//
//    cuda_model.step(..);
//
//
//	AgentPopulation population2(circle_agent, 100);
//	cuda_model.getPopulationData(population2);
//
//	//check values are the same
//	for (int i = 0; i < 100; i++){
//		AgentInstance i1 = population.getInstanceAt(i, "default");
//		AgentInstance i2 = population2.getInstanceAt(i, "default");
//		//use AgentInstance equality operator
//		BOOST_CHECK(i1.getVariable<int>("id") == i2.getVariable<int>("id"));
//	}
//
//}

BOOST_AUTO_TEST_SUITE_END()

