/**
 * @file test_sim_validation.h
 * @brief Testing Using the Boost Unit Test Framework
 */

#include "../flame_api.h"
#include "test_func_pointer.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(SimTest)

BOOST_AUTO_TEST_CASE(SimulationFunctionCheck)
{

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

//    circle_agent.addAgentVariable<float>("x");
//    circle_agent.addAgentVariable<float>("y");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
	output_data.setFunction(output_func);
    //output_data.setInitialState("state1");
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    circle_agent.addAgentFunction(move);

    flame_model.addAgent(circle_agent);


//    	AgentPopulation population(circle_agent);
//
//    for (int i=0; i< 10; i++)
//    {
//        AgentInstance instance = population.getNextInstance("default");
//        instance.setVariable<float>("x", i*0.1f);
//        instance.setVariable<float>("y", i*0.1f);
//    }


    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer");
    output_layer.addAgentFunction("output_data");
    simulation.addSimulationLayer(output_layer);


    SimulationLayer input_layer(simulation, "input_layer");
    //check if the function name exists
    BOOST_CHECK_THROW(input_layer.addAgentFunction("output_",output_func), InvalidAgentFunc); // expecting an error

    SimulationLayer move_layer(simulation, "move_layer");
    move_layer.addAgentFunction("move",move_func);
    simulation.addSimulationLayer(move_layer);

    BOOST_TEST_MESSAGE( "\nTesting Simulation function layers .." );


    // check the name of the agent function
    for (auto i: simulation.getFunctionAtLayer(0)){
        BOOST_CHECK(i.first==output_func);
        }

    // check the name of the agent function
    for (auto i: simulation.getFunctionAtLayer(1)){
        BOOST_CHECK(i.first==move_func);
        }

    //check that getFunctionAtLayer should fail if layer does not exist
    BOOST_CHECK_THROW(simulation.getFunctionAtLayer(2), InvalidMemoryCapacity); // expecting an error

    size_t one = 1;
     const AgentFunctionMap& func = simulation.getFunctionAtLayer(0);
    BOOST_CHECK(func.size()==one);  //check the layer 0 size

    //BOOST_CHECK(simulation.getModelDescritpion().getName()=="circle");
}

BOOST_AUTO_TEST_CASE(SimulationFunctionLayerCheck)
{

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    AgentFunctionDescription move("move");
    circle_agent.addAgentFunction(move);

    AgentFunctionDescription stay("stay");
    circle_agent.addAgentFunction(stay);

    flame_model.addAgent(circle_agent);


    Simulation simulation(flame_model);

    SimulationLayer move_layer(simulation, "move_layer");
    move_layer.addAgentFunction("move",move_func);
    move_layer.addAgentFunction("stay",stay_func);
    simulation.addSimulationLayer(move_layer);


    BOOST_TEST_MESSAGE( "\nTesting each simulation function layer .." );

    //check more than one function per layer
    size_t two = 2;
    const AgentFunctionMap& func = simulation.getFunctionAtLayer(0);
    BOOST_CHECK(func.size()==two);
   // BOOST_CHECK(func.at(0)->first==move_func);
   // BOOST_CHECK(func.at(1)->first==stay_func);


    //BOOST_CHECK(simulation.getModelDescritpion().getName()=="circle");
}



BOOST_AUTO_TEST_SUITE_END()

