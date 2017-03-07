 /**
 * @file test_sim_validation.h
 * @author
 * @date    Feb 2017
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

    circle_agent.addAgentVariable<int>("x");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
    output_data.setFunction(&output_func);
    //output_data.setInitialState("state1");
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    move.setFunction(&move_func);
    circle_agent.addAgentFunction(move);

    AgentFunctionDescription stay("stay");
    stay.setFunction(&stay_func);
    circle_agent.addAgentFunction(stay);

    flame_model.addAgent(circle_agent);


    AgentPopulation population(circle_agent);

    for (int i=0; i< 10; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<int>("x", i*0.1f);
    }


    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer");
    output_layer.addAgentFunction("output_data");
    simulation.addSimulationLayer(output_layer);


    SimulationLayer input_layer(simulation, "input_layer");

    //check if the function name exists
    BOOST_CHECK_THROW(input_layer.addAgentFunction("output_"), InvalidAgentFunc); // expecting an error

    SimulationLayer moveStay_layer(simulation, "move_layer");
    moveStay_layer.addAgentFunction("move");
    moveStay_layer.addAgentFunction("stay");
    simulation.addSimulationLayer(moveStay_layer);



    BOOST_TEST_MESSAGE( "\nTesting simulation of functions per layers .." );

    // check number of function layers
    BOOST_CHECK(simulation.getLayerCount()==2);

    //for each each sim layer
    for (unsigned int i = 0; i < simulation.getLayerCount(); i++)
    {
        const FunctionDescriptionVector& functions = simulation.getFunctionsAtLayer(i);

        //for each func function
        for (AgentFunctionDescription func_des : functions)
        {
            // check functions - printing function name only
            BOOST_TEST_MESSAGE( "Calling agent function "<< func_des.getName() << " at layer " << i << "!\n");
        }
    }

    BOOST_CHECK(simulation.getModelDescritpion().getName()=="circles_model");
}
//
//BOOST_AUTO_TEST_CASE(SimulationFunctionLayerCheck)
//{
//
//    ModelDescription flame_model("circles_model");
//    AgentDescription circle_agent("circle");
//
//    AgentFunctionDescription move("move");
//    move.setFunction(move_func);
//    circle_agent.addAgentFunction(move);
//
//
//    AgentFunctionDescription stay("stay");
//    stay.setFunction(stay_func);
//    circle_agent.addAgentFunction(stay);
//
//    flame_model.addAgent(circle_agent);
//
//
//    Simulation simulation(flame_model);
//
//    SimulationLayer moveStay_layer(simulation, "move_layer");
//    moveStay_layer.addAgentFunction("move");
//    moveStay_layer.addAgentFunction("stay");
//    simulation.addSimulationLayer(moveStay_layer);
//
//}

// DONE : test funcs executing (printing hello only)
// TODO : test funcs executing (printing values only)
// TODO : test funcs executing (on device)

BOOST_AUTO_TEST_SUITE_END()

