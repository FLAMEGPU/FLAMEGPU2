#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Suites

//#include "boost/test/unit_test.hpp"
#include <boost/test/included/unit_test.hpp>
#include "model/ModelDescription.h"
//#include "model/AgentDescription.h"
using namespace std;

BOOST_AUTO_TEST_SUITE(modelTest) //name of the test suite is modelTest

BOOST_AUTO_TEST_CASE(AgentNameCheck)
{

    ModelDescription flame_model("circles_model");
	AgentDescription circle_agent("circle");

    BOOST_CHECK(circle_agent.getName() == "circle");
}

BOOST_AUTO_TEST_CASE(memSizeCheck)
{

    ModelDescription flame_model("circles_model");
	AgentDescription circle_agent("circle");

    BOOST_CHECK(circle_agent.getMemorySize()== 0);
}

BOOST_AUTO_TEST_CASE(AgentVarNumCheck)
{

    ModelDescription flame_model("circles_model");
	AgentDescription circle_agent("circle");
	circle_agent.addAgentVariable<float>("x");


    BOOST_CHECK(circle_agent.getNumberAgentVariables() == 1);

}

BOOST_AUTO_TEST_CASE(AgentVarSizeCheck)
{

    ModelDescription flame_model("circles_model");
	AgentDescription circle_agent("circle");
	circle_agent.addAgentVariable<float>("x");

    BOOST_CHECK(circle_agent.getAgentVariableSize("x") == 4);
}


BOOST_AUTO_TEST_CASE(AgentVarTypeCheck)
{

    std::type_info *temp ;
    ModelDescription flame_model("circles_model");
	AgentDescription circle_agent("circle");
	circle_agent.addAgentVariable<float>("x");

    BOOST_CHECK(circle_agent.getVariableType("x").name() == "f");
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
