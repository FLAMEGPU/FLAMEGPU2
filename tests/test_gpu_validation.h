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


	circle_agent.addAgentVariable<float>("x");
	circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

	AgentPopulation population(circle_agent);
	for (int i = 0; i< 100; i++)
	{
		AgentInstance instance = population.getNextInstance("default");
		instance.setVariable<float>("x", i*0.1f);
		instance.setVariable<float>("y", i*0.1f);
	}
	

    CUDAAgentModel cuda_model(flame_model);
    cuda_model.setInitialPopulationData(population);

	AgentPopulation population2(circle_agent);

	/*
    BOOST_TEST_MESSAGE( "\nTesting CUDA Agent model name" );
    BOOST_CHECK_MESSAGE(cuda_model.);
	*/

    //cuda_model.simulate(simulation);

    //BOOST_CHECK_THROW(..,InvalidCudaAgent); // expecting an error
    //BOOST_CHECK_THROW(..,InvalidCudaAgentDesc); // expecting an error
    //BOOST_CHECK_THROW(..,InvalidCudaAgentMapSize); // expecting an error
    //BOOST_CHECK_THROW(..,InvalidHashList); // expecting an error

}

BOOST_AUTO_TEST_SUITE_END()

