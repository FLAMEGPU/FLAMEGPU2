// NOTE (mozhgan#1#07/12/16): We SHOULD have each BOOST_CHECK as a seperate Test case. The reason for this is if it fails one test, it never reach the next BOOST_CHECK that exist in the same TEST_CASE.

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Pop_TestSuites

//#include "boost/test/unit_test.hpp"
#include <boost/test/included/unit_test.hpp>
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
    AgentPopulation population(flame_model, "circle");
    for (int i=0; i< 100; i++)
    {
        AgentInstance instance = population.addInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }

    BOOST_TEST_MESSAGE( "\nTesting Agent population Name .." );
    BOOST_CHECK(population.getAgentName()=="circle");
}

BOOST_AUTO_TEST_CASE(PopulationInstVarCheck1)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(flame_model, "circle");

    AgentInstance instance = population.addInstance("default");
    instance.setVariable<float>("x", 0.1f);

    BOOST_CHECK_MESSAGE(instance.getVariable<float>("x")==0.1f, "Variable is "<< instance.getVariable<float>("x") << " and not 0.1f!!");
    BOOST_CHECK_MESSAGE(instance.getVariable<int>("x")==0.1f, "Variable is "<< instance.getVariable<int>("x") << " and not 0.1f or there is problem with the type!!");

}

BOOST_AUTO_TEST_CASE(PopulationInstVarCheck2)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(flame_model, "circle");

    AgentInstance instance = population.addInstance("default");
    instance.setVariable<float>("x", 0.1f);

    //cout<< instance.getVariable<float>("y") << endl;
    BOOST_CHECK_MESSAGE(instance.getVariable<float>("y")==0, "Variable is "<< instance.getVariable<float>("y") << " and not 0 by default!!");

}

BOOST_AUTO_TEST_CASE(PopulationInstVarCheck3)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population Instance Variable .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(flame_model, "circle");

    AgentInstance instance = population.addInstance("default");
    instance.setVariable<float>("x", 0.1f);

    BOOST_CHECK_MESSAGE(instance.getVariable<float>("z")==0, "Variable does not exist Error -->  "<< instance.getVariable<float>("z") << " !!");

}


BOOST_AUTO_TEST_CASE(PopulationSizeCheck)
{

    BOOST_TEST_MESSAGE( "\nTesting Agent population size set by default .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(flame_model, "circle", 10);

    BOOST_CHECK(population.getMaximumPopulationSize()==10);

    for (int i=0; i< 100; i++)
    {
        AgentInstance instance = population.addInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }


    BOOST_CHECK_MESSAGE(population.getMaximumPopulationSize()==1034, "population is " << population.getMaximumPopulationSize() << " and not 1034!!");

}


BOOST_AUTO_TEST_CASE(PopulationSizeExtraCheck)
{


    BOOST_TEST_MESSAGE( "\nTesting adding agents more than the max population .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(flame_model, "circle", 1);

    for (int i=0; i< 1026; i++)
    {
        AgentInstance instance = population.addInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }


    BOOST_CHECK_MESSAGE(population.getMaximumPopulationSize()==2049, "population is " << population.getMaximumPopulationSize() << " and not 2049!!");

}


BOOST_AUTO_TEST_SUITE_END()

/*
Build object files by compiling with g++

nvcc -c test_model_validation.cpp -o test.o -std=c++11 -I/usr/include/boost/test/included/

To Link:
/usr/local/cuda-8.0//bin/nvcc -ccbin g++   -m64    -Xlinker -L  -gencode arch=compute_37,code=sm_37 -gencode arch=compute_50,code=sm_50 -gencode arch=compute_52,code=sm_52 -gencode arch=compute_52,code=compute_52 -o suites model/AgentDescription.o model/MessageDescription.o model/AgentFunctionOutput.o model/AgentStateDescription.o model/ModelDescription.o model/AgentFunctionInput.o model/AgentFunctionDescription.o test.o

To run:
./suites --log_level=test_suite
*/
