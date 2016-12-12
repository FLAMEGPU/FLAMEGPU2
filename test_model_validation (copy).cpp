// NOTE (mozhgan#1#07/12/16): We SHOULD have each BOOST_CHECK as a seperate Test case. The reason for this is if it fails one test, it never reach the next BOOST_CHECK that exist in the same TEST_CASE.

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Model_TestSuites

//#include "boost/test/unit_test.hpp"
#include <boost/test/included/unit_test.hpp>
#include <boost/any.hpp>
#include "model/ModelDescription.h"
#include <typeinfo>

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
}

BOOST_AUTO_TEST_CASE(DefaultValueCheck)
{
    BOOST_TEST_MESSAGE( "Testing Agent Variable Default Value" );
    AgentDescription circle_agent("circle");
    circle_agent.addAgentVariable<float>("f");
    circle_agent.addAgentVariable<int>("i");


    BOOST_CHECK(boost::any_cast<float>(circle_agent.getDefaultValue("f")) == 0.0f);
    BOOST_CHECK(boost::any_cast<int>(circle_agent.getDefaultValue("i")) == 0);

}

// NOTE (mozhgan#1#06/12/16): What do we expect the 'getVarableType' to do?
// FIXME (mozhgan#1#06/12/16): This test FAILS as the type is not equal. Is it gonna be 'f' or 'float' ?


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


}
BOOST_AUTO_TEST_SUITE_END()

