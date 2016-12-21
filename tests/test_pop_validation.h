/**
 * @file test_pop_validation.h
 * @brief Testing Using the Boost Unit Test Framework
 */
#include "../flame_api.h"


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
        AgentInstance instance = population.getNextInstance("default");
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

    AgentInstance instance = population.getNextInstance("default");

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

    AgentInstance instance = population.getNextInstance("default");
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

	//population.setStateListCapacity(100);
	//TODO: Catch exception that fails on above call (can't reduce population capacity)

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
	//add one more (should fail)
	//TODO: Test must check that getNextInstance fails if capacity is too small when for loop creates 101 agents
	//AgentInstance instance = population.getNextInstance("default");
	//instance.setVariable<float>("x", 101*0.1f);

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

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

	AgentPopulation population(circle_agent, 100);

	//TODO: Test must check that getInstanceAt should fail if index is less than size
	//AgentInstance instance = population.getInstanceAt(0, "default");

}

BOOST_AUTO_TEST_SUITE_END()

