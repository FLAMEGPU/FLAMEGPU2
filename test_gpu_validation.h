/**
 * @file test_gpu_validation.h
 * @brief Testing Using the Boost Unit Test Framework
 */
#include "pop/AgentPopulation.h"
#include "sim/Simulation.h"
#include "gpu/CUDAAgentModel.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(GPUTest) //name of the test suite is modelTest

BOOST_AUTO_TEST_CASE(SimulationNameCheck)
{
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);

    circle_agent.addAgentFunction(output_data);
    flame_model.addAgent(circle_agent);

    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer");
    output_layer.addAgentFunction("output_data");
    simulation.addSimulationLayer(output_layer);

	AgentPopulation population(flame_model, "circle");
	for (int i = 0; i< 100; i++)
	{
		AgentInstance instance = population.addInstance("default");
		instance.setVariable<float>("x", i*0.1f);
		instance.setVariable<float>("y", i*0.1f);
		instance.setVariable<float>("dx", 0);
		instance.setVariable<float>("dy", 0);
	}

    CUDAAgentModel cuda_model(flame_model);
    cuda_model.setInitialPopulationData(population);



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

