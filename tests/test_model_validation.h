/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_model_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @date       16 Oct 2017
 * @brief      Test suite for validating methods in model folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include "../flame_api.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(ModelDescTest) //name of the test suite is modelTest

BOOST_AUTO_TEST_CASE(AgentCheck)
{

    BOOST_TEST_MESSAGE( "\nTesting Agent Name and Size .." );

    AgentDescription circle_agent("circle");

    BOOST_CHECK(circle_agent.getName() == "circle");
    BOOST_CHECK(circle_agent.getMemorySize()== 0);
}

BOOST_AUTO_TEST_CASE(AgentVarCheck)
{
    BOOST_TEST_MESSAGE( "Testing Agent Variable Size, Type, and Number .." );
    AgentDescription circle_agent("circle");
    circle_agent.addAgentVariable<float>("x");

    BOOST_CHECK(circle_agent.getNumberAgentVariables() == 1);
    BOOST_CHECK(circle_agent.getAgentVariableSize("x") == 4);
    BOOST_CHECK(circle_agent.getVariableType("x") == typeid(float));
	BOOST_CHECK_THROW(circle_agent.getAgentVariableSize("y"), InvalidAgentVar); // expecting an error
}

/*
//default values removed from model as no longer using boost::any
BOOST_AUTO_TEST_CASE(DefaultValueCheck)
{

    BOOST_TEST_MESSAGE( "Testing Agent Variable Default Value" );
    AgentDescription circle_agent("circle");
    circle_agent.addAgentVariable<float>("f");
    circle_agent.addAgentVariable<int>("i");

	float default_f = float();

    BOOST_CHECK(boost::any_cast<float>(circle_agent.getDefaultValue("f")) == 0.0f);
    BOOST_CHECK(boost::any_cast<int>(circle_agent.getDefaultValue("i")) == 0);

	//TODO: Should be a population check which checks the default value of population values

}
*/


BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(MessageTest)

BOOST_AUTO_TEST_CASE(MessageNameCheck)
{

    BOOST_TEST_MESSAGE( "\nTesting Message Name .." );
    MessageDescription location_message("location");

    BOOST_CHECK(location_message.getName()== "location");
    location_message.addVariable<float>("x");
// TODO (mozhgan#1#06/12/16): Test the variable

}

BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(FunctionTest)

BOOST_AUTO_TEST_CASE(FunctionCheck)
{
    BOOST_TEST_MESSAGE( "\nTesting Function and Message Name .." );

    ModelDescription flame_model("circles_model");

    AgentDescription circle_agent("circle");

    MessageDescription location_message("location");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
    //output_data.setInitialState("state1");
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    circle_agent.addAgentFunction(move);


    //model
    flame_model.addMessage(location_message);
    flame_model.addAgent(circle_agent);


    BOOST_CHECK(output_data.getName()=="output_data");
    BOOST_CHECK(output_location.getMessageName()=="location");

    BOOST_CHECK(circle_agent.hasAgentFunction("output_data")==true);

    BOOST_CHECK(output_data.getIntialState()=="default");

    BOOST_CHECK(flame_model.getName()== "circles_model");
    BOOST_CHECK(flame_model.getAgentDescription("circle").hasAgentFunction("move") == true);


    BOOST_CHECK_THROW(flame_model.getAgentDescription("error"),InvalidAgentVar); // expecting an error

}
BOOST_AUTO_TEST_SUITE_END()
