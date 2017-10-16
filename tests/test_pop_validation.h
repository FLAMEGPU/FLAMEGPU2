/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_pop_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @date       16 Oct 2017
 * @brief      Test suite for validating methods in population folder
 *
 * These are example device agent functions to be used for testing.
 * Each function returns a ALIVE or DEAD value indicating where the agent is dead and should be removed or not.
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include "../flame_api.h"


using namespace std;

BOOST_AUTO_TEST_SUITE(PopTest) //name of the test suite is modelTest

/**
 * @brief      To verify the correctness of agent population name and exception handler
 *
 * This test should pass by throwing the correct exception.
*/
BOOST_AUTO_TEST_CASE(PopulationNameCheck)
{
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);
    AgentPopulation population(circle_agent);
    for (int i=0; i< 100; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }

    BOOST_TEST_MESSAGE( "\nTesting Agent population Name .." );
    BOOST_CHECK(population.getAgentName()=="circle");
// what do we expect here?default or circle
    BOOST_CHECK_THROW(population.getStateMemory("circe"),InvalidStateName); // expecting an error
}


/**
 * @brief      To verify the correctness of exception handler when setting or
 * getting a variable of invalid type.
 *
 * This test should pass by throwing the correct exception.
*/
BOOST_AUTO_TEST_CASE(PopulationInstVarCheck1)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    AgentInstance instance = population.getNextInstance("default");

#pragma warning( push )
#pragma warning( disable : 4244)
    BOOST_CHECK_THROW(instance.setVariable<int>("x", 0.1f),InvalidVarType);
#pragma warning( pop )

    instance.setVariable<float>("x", 0.1f);

    BOOST_CHECK_MESSAGE(instance.getVariable<float>("x")==0.1f, "Variable is "<< instance.getVariable<float>("x") << " and not 0.1f!!");

    BOOST_CHECK_THROW(instance.getVariable<int>("x"), InvalidVarType); // expecting an error

}

/**
 * @brief      To verify the correctness of ::AgentInstance::getVariable and
 * ::AgentInstance::getVariable functions by checking the population data
 *
 * This test should pass.
*/
BOOST_AUTO_TEST_CASE(PopulationInstVarCheck2)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    AgentInstance instance = population.getNextInstance("default");
    instance.setVariable<float>("x", 0.1f);


    BOOST_CHECK_MESSAGE(instance.getVariable<float>("y")==0.0f, "Variable is "<< instance.getVariable<float>("y") << " and not 0.0f by default!!");

}


/**
 * @brief      To verify the correctness of exception handler when trying to
 * access the invalid variable that does not exist.
 *
 * This test should pass by throwing the correct exception.
*/
BOOST_AUTO_TEST_CASE(PopulationInstVarCheck3)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    AgentInstance instance = population.getNextInstance("default");
    instance.setVariable<float>("x", 0.1f);

    BOOST_CHECK_THROW(instance.getVariable<float>("z"), InvalidAgentVar);

}


BOOST_AUTO_TEST_CASE(PopulationSizeCheck)
{

    BOOST_TEST_MESSAGE( "\nTesting Agent population size set by default .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent);

    BOOST_CHECK(population.getMaximumStateListCapacity()==1024);

}

BOOST_AUTO_TEST_CASE(PopulationAddMoreCapacity)
{


    BOOST_TEST_MESSAGE("\nTesting changing the capacity..");

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);
    BOOST_CHECK(population.getMaximumStateListCapacity() == 100);

    population.setStateListCapacity(200);
    BOOST_CHECK(population.getMaximumStateListCapacity() == 200);

    //Catch exception that fails on above call (can't reduce population capacity)
    BOOST_CHECK_THROW(population.setStateListCapacity(100),InvalidPopulationData);
}

BOOST_AUTO_TEST_CASE(PopulationOverflowCapacity)
{

    BOOST_TEST_MESSAGE("\nTesting overflowing the capacity of a state list..");

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);
    BOOST_CHECK(population.getMaximumStateListCapacity() == 100);

    //add 100 instances (no problem)
    for (int i = 0; i< 100; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }
    // getNextInstance fails if capacity is too small when for loop creates 101 agents
    BOOST_CHECK_THROW(population.getNextInstance("default"),InvalidMemoryCapacity);
}

BOOST_AUTO_TEST_CASE(PopulationDataValuesMultipleStates)
{


    BOOST_TEST_MESSAGE("\nTesting adding changing the capacity size ..");

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");
    AgentStateDescription s1("s1");
    AgentStateDescription s2("s2");
    circle_agent.addState(s1);
    circle_agent.addState(s2);

    circle_agent.addAgentVariable<int>("id");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);

    //add 100 instances (no problem)
    for (int i = 0; i< 100; i++)
    {
        AgentInstance instance_s1 = population.getNextInstance("s1");
        instance_s1.setVariable<int>("id", i);

        AgentInstance instance_s2 = population.getNextInstance("s2");
        instance_s2.setVariable<int>("id", i + 1000);
    }

    //check values are correct
    for (int i = 0; i< 100; i++)
    {
        AgentInstance instance_s1 = population.getInstanceAt(i, "s1");
        BOOST_CHECK(instance_s1.getVariable<int>("id") == i);

        AgentInstance instance_s2 = population.getInstanceAt(i, "s2");
        BOOST_CHECK(instance_s2.getVariable<int>("id") == i + 1000);
    }
}


BOOST_AUTO_TEST_CASE(PopulationCheckGetInstanceBeyondSize)
{


    BOOST_TEST_MESSAGE( "\nTesting getting an instance beynd current size .." );

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

    //check that getInstanceAt should fail if index is less than size
    BOOST_CHECK_THROW(population.getInstanceAt(1,"default"),InvalidMemoryCapacity);

}

BOOST_AUTO_TEST_SUITE_END()

