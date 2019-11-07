#ifndef TESTS_TEST_CASES_MODEL_TEST_MESSAGE_VALIDATION_H_
#define TESTS_TEST_CASES_MODEL_TEST_MESSAGE_VALIDATION_H_
/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_message_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @brief      Test suite for validating methods in for messages in model folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include <string>
#include <utility>

#include "flamegpu/flame_api.h"

BOOST_AUTO_TEST_SUITE(MessageTest)  // name of the test suite is MessageTest

/**
 * @brief      To verify the correctness of message name, size, and type.
 * To test the case separately, run: make run_BOOST_TEST TSuite=MessageTest/MessageCheck
 *
*/
BOOST_AUTO_TEST_CASE(MessageCheck) {
    BOOST_TEST_MESSAGE("\nTesting Message Name and Size, Type, and Number ..");

    MessageDescription location_message("location");

   /**
     * @brief      Checks the number of message name
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(location_message.getName() == "location");

   /**
     * @brief      Checks the number of message memory size
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(location_message.getMemorySize()== 0);


    location_message.addVariable<float>("x");
    location_message.addVariable<float>("y");

   /**
     * @brief      Checks the number of message variables
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(location_message.getNumberMessageVariables()== 2);

   /**
     * @brief      Checks the message variable size
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(location_message.getMessageVariableSize("x")== 4);

   /**
     * @brief      Checks the message variable type
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(location_message.getVariableType("x")== typeid(float));


    /**
    * @brief      Checks the mapped message variables
    * @todo change the boost test message style to boost_check
    */
    const VariableMap &mem = location_message.getVariableMap();
    for (const VariableMapPair& mm : mem) {
        // get the variable name
        std::string var_name = mm.first;
        BOOST_TEST_MESSAGE("variable names:" << var_name);
    }

   /**
     * @brief      Checks if the message variable exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
    BOOST_CHECK_THROW(location_message.getMessageVariableSize("z"), InvalidMessageVar);  // expecting an error
}


/**
 * @brief      To verify the correctness of message names.
 * To test the case separately, run: make run_BOOST_TEST TSuite=MessageTest/MessageFunctionCheck
 *
*/
BOOST_AUTO_TEST_CASE(MessageFunctionCheck) {
    BOOST_TEST_MESSAGE("\nTesting Function and Message Name ..");

    ModelDescription flame_model("circles_model");

    AgentDescription circle_agent("circle");

    MessageDescription location_message("location");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    circle_agent.addAgentFunction(move);

    flame_model.addMessage(location_message);


   /**
     * @brief      Checks the name of agent function description
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(output_data.getName() == "output_data");

   /**
     * @brief      Checks whether the agent function reads an input message
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(output_data.hasInputMessage() == false);
    BOOST_CHECK(move.hasInputMessage() == false);

   /**
     * @brief      Checks whether the agent function outputs a message
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(output_data.hasOutputMessage() == true);
    BOOST_CHECK(move.hasOutputMessage() == false);

   /**
     * @brief      Checks the message name
     * This is to validate the predicate value. The test should pass.
     */
    BOOST_CHECK(output_location.getMessageName() == "location");


    /**
     * @brief      Checks if the message description name exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
    BOOST_CHECK_THROW(flame_model.getMessageDescription("error"), InvalidMessageVar);  // expecting an error
}

// TODO: Check that we can output (single) messages during simulation without error

// TODO: Ensure that agents can not output more than a single message (correct error must be thrown)

// TODO: Check that we can output (optional) messages during simulation without error - must check that the sparse message list becomes dense

// TODO: Check that message iterators count over the correct number of agents

// TODO: More advanced input of messages to check the values are correct

BOOST_AUTO_TEST_SUITE_END()

#endif  // TESTS_TEST_CASES_MODEL_TEST_MESSAGE_VALIDATION_H_
