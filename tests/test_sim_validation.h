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
    move.setFunction(move_func);
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
    BOOST_CHECK_THROW(input_layer.addAgentFunction("output_"), InvalidAgentFunc); // expecting an error

    SimulationLayer move_layer(simulation, "move_layer");
    move_layer.addAgentFunction("move");
    simulation.addSimulationLayer(move_layer);

    BOOST_TEST_MESSAGE( "\nTesting Simulation function layers (names and function pointers - one func per layer) .." );

    // NOTE : this test only works if there is one func per layer.
	//TODO: What is this testing? Layers have multiple functions so the getFunctionAtLayer semantics is wrong. Perhaps getFunctionsAtLayer() then check all functions if the function is in the layer. Or getLayerAt(int).getFunctionAt(int)
	/*
    // check the name of the agent function
    for (auto i: simulation.getFunctionAtLayer(0))
    {
        BOOST_CHECK(i.first=="output_data");
    }
    // check the name of the function pointer
    for (auto i: simulation.getFunctionAtLayer(0))
    {
        //BOOST_CHECK(i.second.getFunction()==output_func);
    }

    // check the name of the agent function
    for (auto i: simulation.getFunctionAtLayer(1))
    {
        BOOST_CHECK(i.first=="move");
    }
    // check the name of the function pointer
    for (auto i: simulation.getFunctionAtLayer(1))
    {
        //BOOST_CHECK(i.second.getFunction()==move_func);
    }

    //check that getFunctionAtLayer should fail if layer does not exist
    BOOST_CHECK_THROW(simulation.getFunctionAtLayer(2), InvalidMemoryCapacity); // expecting an error

    size_t one = 1;
    const FunctionDesMap& func = simulation.getFunctionAtLayer(0);
    BOOST_CHECK(func.size()==one);  //check the layer 0 size
	*/

    BOOST_CHECK(simulation.getModelDescritpion().getName()=="circles_model");
}

BOOST_AUTO_TEST_CASE(SimulationFunctionLayerCheck)
{

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    AgentFunctionDescription move("move");
    move.setFunction(move_func);
    circle_agent.addAgentFunction(move);


    AgentFunctionDescription stay("stay");
    stay.setFunction(stay_func);
    circle_agent.addAgentFunction(stay);

    flame_model.addAgent(circle_agent);


    Simulation simulation(flame_model);

    SimulationLayer moveStay_layer(simulation, "move_layer");
    moveStay_layer.addAgentFunction("move");
    moveStay_layer.addAgentFunction("stay");
    simulation.addSimulationLayer(moveStay_layer);


    BOOST_TEST_MESSAGE( "\nTesting each simulation function layer  (names and function pointers - multi func per layer) .." );

    //check more than one function per layer
  //  size_t two = 2;
//    const FunctionDesMap& func = simulation.getFunctionAtLayer(0);
 //   BOOST_CHECK(func.size()==two);
    //BOOST_CHECK(func.at(0)->second.getFunction()==move_func);
    // BOOST_CHECK(func.at(1)->second.getFunction()==stay_func);
}

// TODO : test funcs executing (printing hello only)
// TODO : test funcs executing (printing values only)
// TODO : test funcs executing (on device)

BOOST_AUTO_TEST_SUITE_END()

