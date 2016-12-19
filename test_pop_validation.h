/**
 * @file test_pop_validation.h
 * @brief Testing Using the Boost Unit Test Framework
 */
#include "model/ModelDescription.h"
#include "pop/AgentPopulation.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(PopTest) //name of the test suite is modelTest

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
        AgentInstance instance = population.pushBackInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }

    BOOST_TEST_MESSAGE( "\nTesting Agent population Name .." );
    BOOST_CHECK(population.getAgentName()=="circle");

    BOOST_CHECK_THROW(population.getStateMemory("circe"),InvalidStateName); // expecting an error
}



BOOST_AUTO_TEST_CASE(PopulationInstVarCheck1)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

	AgentPopulation population(circle_agent);

    AgentInstance instance = population.pushBackInstance("default");

#pragma warning( push )
#pragma warning( disable : 4244)
    BOOST_CHECK_THROW(instance.setVariable<int>("x", 0.1f),InvalidVarType);
#pragma warning( pop )

    instance.setVariable<float>("x", 0.1f);

    BOOST_CHECK_MESSAGE(instance.getVariable<float>("x")==0.1f, "Variable is "<< instance.getVariable<float>("x") << " and not 0.1f!!");

    BOOST_CHECK_THROW(instance.getVariable<int>("x"), InvalidVarType); // expecting an error

}

BOOST_AUTO_TEST_CASE(PopulationInstVarCheck2)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

	AgentPopulation population(circle_agent);

    AgentInstance instance = population.pushBackInstance("default");
    instance.setVariable<float>("x", 0.1f);


    BOOST_CHECK_MESSAGE(instance.getVariable<float>("y")==0.0f, "Variable is "<< instance.getVariable<float>("y") << " and not 0.0f by default!!");

}



BOOST_AUTO_TEST_CASE(PopulationInstVarCheck3)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

	AgentPopulation population(circle_agent);

    AgentInstance instance = population.pushBackInstance("default");
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

	AgentPopulation population(circle_agent, 10);

    BOOST_CHECK(population.getMaximumStateListSize()==10);

    for (int i=0; i< 100; i++)
    {
        AgentInstance instance = population.pushBackInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }


    BOOST_CHECK_MESSAGE(population.getMaximumStateListSize()==110, "population is " << population.getMaximumStateListSize() << " and not 110!!");

}


BOOST_AUTO_TEST_CASE(PopulationSizeExtraCheck)
{


    BOOST_TEST_MESSAGE( "\nTesting adding agents more than the max population .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

	AgentPopulation population(circle_agent); //default size is 1024

    for (int i=0; i< 1026; i++)
    {
        AgentInstance instance = population.pushBackInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }


    BOOST_CHECK_MESSAGE(population.getMaximumStateListSize()==2050, "population is " << population.getMaximumStateListSize() << " and not 2050!!");

}

BOOST_AUTO_TEST_SUITE_END()

