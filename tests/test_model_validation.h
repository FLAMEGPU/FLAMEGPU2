/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_model_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @brief      Test suite for validating methods in model folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include <flamegpu/flame_api.h>

using namespace std;

BOOST_AUTO_TEST_SUITE(ModelDescTest) //name of the test suite is modelDescTest

/**
 * @brief      To verify the correctness of agent name and size.
 * To test the case separately, run: make run_BOOST_TEST TSuite=ModelDescTest/AgentCheck
 *
*/
BOOST_AUTO_TEST_CASE(AgentCheck)
{

    BOOST_TEST_MESSAGE( "\nTesting Agent Name and Size .." );

    AgentDescription circle_agent("circle");

    BOOST_CHECK(circle_agent.getName() == "circle");
    BOOST_CHECK(circle_agent.getMemorySize()== 0);
}

/**
 * @brief     To verify the correctness of agent variable size and type.
 * To test the case separately, run: make run_BOOST_TEST TSuite=ModelDescTest/AgentVarCheck
 *
*/
BOOST_AUTO_TEST_CASE(AgentVarCheck)
{
    BOOST_TEST_MESSAGE( "Testing Agent Variable Size, Type, and Number .." );
    AgentDescription circle_agent("circle");
    circle_agent.addAgentVariable<float>("x");

   /**
     * @brief      Checks the number of agent variables
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(circle_agent.getNumberAgentVariables() == 1);

   /**
     * @brief      Checks the agent variable size
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(circle_agent.getAgentVariableSize("x") == 4);

   /**
     * @brief      Checks the agent variable type
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(circle_agent.getVariableType("x") == typeid(float));

    /**
     * @brief      Checks if the agent variable exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
	BOOST_CHECK_THROW(circle_agent.getAgentVariableSize("y"), InvalidAgentVar); // expecting an error
}


/**
 * @brief      To verify the correctness of function name and existence.
 * To test the case separately, run: make run_BOOST_TEST TSuite=ModelDescTest/MessageFunctionCheck
 *
*/
BOOST_AUTO_TEST_CASE(MessageFunctionCheck)
{
    BOOST_TEST_MESSAGE( "\nTesting Function and Message Name .." );

    ModelDescription flame_model("circles_model");

    AgentDescription circle_agent("circle");

    AgentFunctionDescription output_data("output_data");
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    circle_agent.addAgentFunction(move);

    flame_model.addAgent(circle_agent);


   /**
     * @brief      Checks the name of agent function description
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(output_data.getName()=="output_data");

   /**
     * @brief      Checks whether the agent function exists or not
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(circle_agent.hasAgentFunction("output_data")==true);

   /**
     * @brief      Checks the name of the initial state
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(output_data.getInitialState()=="default");

   /**
     * @brief      Checks the name of model description
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(flame_model.getName()== "circles_model");

   /**
     * @brief      Checks whether the agent function exists or not
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(flame_model.getAgentDescription("circle").hasAgentFunction("move") == true);

    /**
     * @brief      Checks if the agent description name exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
    BOOST_CHECK_THROW(flame_model.getAgentDescription("error"),InvalidAgentVar); // expecting an error

}
BOOST_AUTO_TEST_SUITE_END()
