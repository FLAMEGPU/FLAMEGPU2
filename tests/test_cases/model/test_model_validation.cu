#ifndef TESTS_TEST_CASES_MODEL_TEST_MODEL_VALIDATION_H_
#define TESTS_TEST_CASES_MODEL_TEST_MODEL_VALIDATION_H_
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

#include <utility>

#include "gtest/gtest.h"

#include "helpers/common.h"

#include "flamegpu/flame_api.h"

/**
 * @brief      To verify the correctness of agent name and size.
 * To test the case separately, run: make run_BOOST_TEST TSuite=ModelDescTest/AgentCheck
 *
*/
TEST(ModelDescTest, AgentCheck) {
    GTEST_COUT << "Testing Agent Name and Size .." << std::endl;

    AgentDescription circle_agent("circle");

    EXPECT_EQ(circle_agent.getName(), "circle");
    EXPECT_EQ(circle_agent.getMemorySize(), 0llu);
}

/**
 * @brief     To verify the correctness of agent variable size and type.
 * To test the case separately, run: make run_BOOST_TEST TSuite=ModelDescTest/AgentVarCheck
 *
*/
TEST(ModelDescTest, AgentVarCheck) {
    GTEST_COUT << "Testing Agent Variable Size, Type, and Number .." << std::endl;
    AgentDescription circle_agent("circle");
    circle_agent.addAgentVariable<float>("x");

   /**
     * @brief      Checks the number of agent variables
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_EQ(circle_agent.getNumberAgentVariables(), 1u);

   /**
     * @brief      Checks the agent variable size
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_EQ(circle_agent.getAgentVariableSize("x"), 4llu);

   /**
     * @brief      Checks the agent variable type
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_EQ(circle_agent.getVariableType("x"), typeid(float));

    /**
     * @brief      Checks if the agent variable exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
    EXPECT_THROW(circle_agent.getAgentVariableSize("y"), InvalidAgentVar);  // expecting an error
}


/**
 * @brief      To verify the correctness of function name and existence.
 * To test the case separately, run: make run_BOOST_TEST TSuite=ModelDescTest/MessageFunctionCheck
 *
*/
TEST(ModelDescTest, MessageFunctionCheck) {
    GTEST_COUT << "Testing Function and Message Name .." << std::endl;

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
    EXPECT_EQ(output_data.getName(), "output_data");

   /**
     * @brief      Checks whether the agent function exists or not
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(circle_agent.hasAgentFunction("output_data"));

   /**
     * @brief      Checks the name of the initial state
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_EQ(output_data.getInitialState(), "default");

   /**
     * @brief      Checks the name of model description
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_EQ(flame_model.getName(), "circles_model");

   /**
     * @brief      Checks whether the agent function exists or not
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_TRUE(flame_model.getAgentDescription("circle").hasAgentFunction("move"));

    /**
     * @brief      Checks if the agent description name exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     */
    EXPECT_THROW(flame_model.getAgentDescription("error"), InvalidAgentVar);  // expecting an error
}

#endif  // TESTS_TEST_CASES_MODEL_TEST_MODEL_VALIDATION_H_
