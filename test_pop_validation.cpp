// NOTE (mozhgan#1#06/12/16): We can have each BOOST_CHECK as a seperate Test case, or we can seperate them with a message, or leave it as it is jsut now.

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
    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");

    AgentPopulation population(flame_model, "circle");
    for (int i=0; i< 100; i++)
    {
        AgentInstance instance = population.addInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }

    BOOST_TEST_MESSAGE( "\nTesting Agent population .." );
    BOOST_CHECK(population.getAgentName()=="circle");
    BOOST_CHECK(population.getMaximumPopulationSize()==100);


}


BOOST_AUTO_TEST_CASE(PopulationSizeCheck)
{


    BOOST_TEST_MESSAGE( "\nTesting Agent population size .." );

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");


    AgentPopulation population(flame_model, "circle");

    AgentInstance instance = population.addInstance("default");
    instance.setVariable<float>("x", i*0.1f);

    BOOST_CHECK(population.getAgentVariable<float>("y")==0);

    BOOST_CHECK(population.getMaximumPopulationSize()==1);



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
