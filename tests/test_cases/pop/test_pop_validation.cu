#ifndef TESTS_TEST_CASES_POP_TEST_POP_VALIDATION_H_
#define TESTS_TEST_CASES_POP_TEST_POP_VALIDATION_H_
/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_pop_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @brief      Test suite for validating methods in population folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include "gtest/gtest.h"

#include "helpers/common.h"

#include "flamegpu/flame_api.h"

/**
 * @brief      To verify the correctness of agent population name and exception handler
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationNameCheck
 * This test should pass by throwing the correct exception.
*/
TEST(PopTest, PopulationNameCheck) {
    const int POPULATION_SIZE = 100;
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);
    AgentPopulation population(circle_agent, POPULATION_SIZE);
    for (int i=0; i< POPULATION_SIZE; i++) {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }

    GTEST_COUT << "Testing Agent population Name .." << std::endl;
    EXPECT_EQ(population.getAgentName(), "circle");
    // what do we expect here?default or circle
    EXPECT_THROW(population.getStateMemory("circe"), InvalidStateName);  // expecting an error
}


/**
 * @brief      To verify the correctness of exception handler when setting or
 * getting a variable of invalid type.
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationInstVarCheck1
 *
 * This test should pass by throwing the correct exception.
*/
TEST(PopTest, PopulationInstVarCheck1) {
    GTEST_COUT << "Testing Agent population Instance Variable .." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    AgentInstance instance = population.getNextInstance("default");

#if defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable : 4244)
#endif
    EXPECT_THROW(instance.setVariable<int>("x", 0.1f), InvalidVarType);
#if defined(_MSC_VER)
#pragma warning(pop)
#endif

    instance.setVariable<float>("x", 0.1f);

    EXPECT_TRUE(instance.getVariable<float>("x") == 0.1f) << "Variable is " << instance.getVariable<float>("x") << " and not 0.1f!!";

    EXPECT_THROW(instance.getVariable<int>("x"), InvalidVarType);  // expecting an error
}

/**
 * @brief      To verify the correctness of ::AgentInstance::getVariable and
 * ::AgentInstance::getVariable functions by checking the population data
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationInstVarCheck2
*/
TEST(PopTest, PopulationInstVarCheck2) {
    GTEST_COUT << "Testing Agent population Instance Variable .." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    AgentInstance instance = population.getNextInstance("default");
    instance.setVariable<float>("x", 0.1f);


    EXPECT_TRUE(instance.getVariable<float>("y") == 0.0f) << "Variable is " << instance.getVariable<float>("y") << " and not 0.0f by default!!";
}


/**
 * @brief      To verify the correctness of exception handler when trying to
 * access the invalid variable that does not exist.
 *
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationInstVarCheck3
 * This test should pass by throwing the correct exception.
*/
TEST(PopTest, PopulationInstVarCheck3) {
    GTEST_COUT << "Testing Agent population Instance Variable .." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    AgentInstance instance = population.getNextInstance("default");
    instance.setVariable<float>("x", 0.1f);

    EXPECT_THROW(instance.getVariable<float>("z"), InvalidAgentVar);
}

/**
 * @brief      To verify the correctness of ::AgentPopulation::getMaximumStateListCapacity
 *
 *  In cases where the agent population is not set, the maximum state list size
 *  would be set to 1024. The test checks the maximum state list size.
 *  To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationSizeCheck
 *
*/
TEST(PopTest, PopulationSizeCheck) {
    GTEST_COUT << "Testing Agent population size set by default .." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    EXPECT_TRUE(population.getMaximumStateListCapacity() == AgentPopulation::DEFAULT_POPULATION_SIZE);
}

/**
 * @brief      To verify the correctness of exception handler when trying to
 * reduce the maximum state list size.
 *
 * Once the state list size is set, it can only be increased. This is to catch
 * the exception when the population capacity is set below the previous size.
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationAddMoreCapacity
 *
 * This test should pass by throwing the correct exception.
*/
TEST(PopTest, PopulationAddMoreCapacity) {
    GTEST_COUT << "Testing changing the capacity.." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);
    EXPECT_EQ(population.getMaximumStateListCapacity(), 100u);

    population.setStateListCapacity(200);
    EXPECT_EQ(population.getMaximumStateListCapacity(), 200u);

    // Catch exception that fails on above call (can't reduce population capacity)
    EXPECT_THROW(population.setStateListCapacity(100), InvalidPopulationData);
}

/**
 * @brief      To verify the correctness of exception handler when trying to
 * access agents more than population size.
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationOverflowCapacity
 *
 * This test should pass by throwing the correct exception.
*/
TEST(PopTest, PopulationOverflowCapacity) {
    GTEST_COUT << "Testing overflowing the capacity of a state list.." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);
    EXPECT_EQ(population.getMaximumStateListCapacity(), 100u);

    // add 100 instances (no problem)
    for (int i = 0; i< 100; i++)     {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }
    // getNextInstance fails if capacity is too small when for loop creates 101 agents
    EXPECT_THROW(population.getNextInstance("default"), InvalidMemoryCapacity);
}

/**
 * @brief      To verify the correctness of exception handler when trying to
 * access agent population value that is not set.
 *
 * After setting initial values for any number of agent population that is less
 * than the maximum population capacity, only the variables been set can be accessed.
 * Accessing index less than actual population size should be handled through an exception.
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationCheckGetInstanceBeyondSize
 *
 * This test should pass by throwing the correct exception.
*/
TEST(PopTest, PopulationCheckGetInstanceBeyondSize) {
    GTEST_COUT << "Testing getting an instance beyond current size .." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    AgentStateDescription s1("default");
    circle_agent.addState(s1);

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);

    AgentInstance instance_s1 = population.getNextInstance("default");
    instance_s1.setVariable<float>("x", 0.1f);

    // check that getInstanceAt should fail if index is less than size
    EXPECT_THROW(population.getInstanceAt(1, "default"), InvalidMemoryCapacity);
}

/**
 * @brief      To verify the correctness of population data values when using multiple states.
 * To test the case separately, run: make run_BOOST_TEST TSuite=PopTest/PopulationDataValuesMultipleStates
 *
*/
TEST(PopTest, PopulationDataValuesMultipleStates) {
    GTEST_COUT << "Testing the population data with multiple states .." << std::endl;

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");
    AgentStateDescription s1("s1");
    AgentStateDescription s2("s2");
    circle_agent.addState(s1);
    circle_agent.addState(s2);

    circle_agent.addAgentVariable<int>("id");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);

    // add 100 instances (no problem)
    for (int i = 0; i< 100; i++) {
        AgentInstance instance_s1 = population.getNextInstance("s1");
        instance_s1.setVariable<int>("id", i);

        AgentInstance instance_s2 = population.getNextInstance("s2");
        instance_s2.setVariable<int>("id", i + 1000);
    }

    // check values are correct
    for (int i = 0; i< 100; i++) {
        AgentInstance instance_s1 = population.getInstanceAt(i, "s1");
        EXPECT_TRUE(instance_s1.getVariable<int>("id") == i);

        AgentInstance instance_s2 = population.getInstanceAt(i, "s2");
        EXPECT_TRUE(instance_s2.getVariable<int>("id") == i + 1000);
    }
}

#endif  // TESTS_TEST_CASES_POP_TEST_POP_VALIDATION_H_
