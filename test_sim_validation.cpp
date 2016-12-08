// NOTE (mozhgan#1#07/12/16): We SHOULD have each BOOST_CHECK as a seperate Test case. The reason for this is if it fails one test, it never reach the next BOOST_CHECK that exist in the same TEST_CASE.

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Sim_TestSuites

//#include "boost/test/unit_test.hpp"
#include <boost/test/included/unit_test.hpp>
#include "model/ModelDescription.h"
#include "sim/Simulation.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(SimTest) //name of the test suite is modelTest

BOOST_AUTO_TEST_CASE(SimulationNameCheck)
{
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    AgentFunctionDescription output_data("output_data");
	AgentFunctionOutput output_location("location");
	output_data.addOutput(output_location);
	//output_data.setInitialState("state1");
	circle_agent.addAgentFunction(output_data);
	flame_model.addAgent(circle_agent);


    Simulation simulation(flame_model);

	SimulationLayer output_layer(simulation, "output_layer");
	output_layer.addAgentFunction("output_data");
	simulation.addSimulationLayer(output_layer);


    BOOST_TEST_MESSAGE( "\nTesting Simulation Name and Size .." );
    //BOOST_CHECK(simulation.getModelDescritpion().getName()=="circle");
}


BOOST_AUTO_TEST_SUITE_END()

