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

#include "gtest/gtest.h"

#include "helpers/common.h"

#include "flamegpu/flame_api.h"

/**
 * @brief      To verify the correctness of message name, size, and type.
 * To test the case separately, run: make run_BOOST_TEST TSuite=MessageTest/MessageCheck
 *
*/
TEST(MessageTest, MessageCheck) {
    GTEST_COUT << "Testing Message Name and Size, Type, and Number .." << std::endl;

    MessageDescription location_message("location");

   /**
     * @brief      Checks the number of message name
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(location_message.getName() == "location");

   /**
     * @brief      Checks the number of message memory size
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(location_message.getMemorySize()== 0);


    location_message.addVariable<float>("x");
    location_message.addVariable<float>("y");

   /**
     * @brief      Checks the number of message variables
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(location_message.getNumberMessageVariables()== 2);

   /**
     * @brief      Checks the message variable size
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(location_message.getMessageVariableSize("x")== 4);

   /**
     * @brief      Checks the message variable type
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(location_message.getVariableType("x")== typeid(float));


    /**
    * @brief      Checks the mapped message variables
    * @todo change the boost test message style to EXPECT_TRUE
    */
    const VariableMap &mem = location_message.getVariableMap();
    for (const VariableMapPair& mm : mem) {
        // get the variable name
        std::string var_name = mm.first;
        GTEST_COUT << "variable names:" << var_name << std::endl;
    }

   /**
     * @brief      Checks if the message variable exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
    EXPECT_THROW(location_message.getMessageVariableSize("z"), InvalidMessageVar);  // expecting an error
}


/**
 * @brief      To verify the correctness of message names.
 * To test the case separately, run: make run_BOOST_TEST TSuite=MessageTest/MessageFunctionCheck
 *
*/
TEST(MessageTest, MessageFunctionCheck) {
    GTEST_COUT << "Testing Function and Message Name .." << std::endl;

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
    EXPECT_TRUE(output_data.getName() == "output_data");

   /**
     * @brief      Checks whether the agent function reads an input message
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(output_data.hasInputMessage() == false);
    EXPECT_TRUE(move.hasInputMessage() == false);

   /**
     * @brief      Checks whether the agent function outputs a message
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(output_data.hasOutputMessage() == true);
    EXPECT_TRUE(move.hasOutputMessage() == false);

   /**
     * @brief      Checks the message name
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(output_location.getMessageName() == "location");


    /**
     * @brief      Checks if the message description name exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
    EXPECT_THROW(flame_model.getMessageDescription("error"), InvalidMessageVar);  // expecting an error
}

// TODO: Check that we can output (single) messages during simulation without error

// TODO: Ensure that agents can not output more than a single message (correct error must be thrown)

// TODO: Check that we can output (optional) messages during simulation without error - must check that the sparse message list becomes dense

// TODO: Check that message iterators count over the correct number of agents

// TODO: More advanced input of messages to check the values are correct


#endif  // TESTS_TEST_CASES_MODEL_TEST_MESSAGE_VALIDATION_H_
