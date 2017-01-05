/**
 * @file test_sim_validation.h
 * @brief Testing Using the Boost Unit Test Framework
 */

#include "../flame_api.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(SimTest) //name of the test suite is modelTest

BOOST_AUTO_TEST_CASE(SimulationFunctionCheck)
{
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
    //output_data.setInitialState("state1");
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    circle_agent.addAgentFunction(move);

    flame_model.addAgent(circle_agent);


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

    BOOST_TEST_MESSAGE( "\nTesting Simulation function layers .." );


    // check the name of the agent function
    for (auto i: simulation.getFunctionAtLayer(0))
        BOOST_CHECK(i=="output_data");

    // check the name of the agent function
    for (auto i: simulation.getFunctionAtLayer(1))
        BOOST_CHECK(i=="move");

    //check that getFunctionAtLayer should fail if layer does not exist
    BOOST_CHECK_THROW(simulation.getFunctionAtLayer(2), InvalidMemoryCapacity); // expecting an error

    size_t one = 1;
    std::vector<std::string> func = simulation.getFunctionAtLayer(0);
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
    move_layer.addAgentFunction("move");
    move_layer.addAgentFunction("stay");
    simulation.addSimulationLayer(move_layer);


    BOOST_TEST_MESSAGE( "\nTesting each simulation function layer .." );

    //check more than one function per layer
    size_t two = 2;
    std::vector<std::string> func = simulation.getFunctionAtLayer(0);
    BOOST_CHECK(func.size()==two);
    BOOST_CHECK(func.at(0)=="move");
    BOOST_CHECK(func.at(1)=="stay");

    //BOOST_CHECK(simulation.getModelDescritpion().getName()=="circle");
}


BOOST_AUTO_TEST_SUITE_END()

