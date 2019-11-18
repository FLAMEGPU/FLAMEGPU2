#ifndef TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
#define TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_sim_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @brief      Test suite for validating methods in simulation folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include <utility>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

#include "helpers/device_test_functions.h"
#include "helpers/common.h"

/**
 * @brief      To verify the correctness of simulation functions
 *
 * This test checks whether functions are executing on device by printing 'Hello'.
 * To test the case separately, run: make run_BOOST_TEST TSuite=SimTest/SimulationFunctionCheck
*/
TEST(SimTest, SimulationFunctionCheck) {
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    // output_data.setOutput(output_location);
    // output_data.setFunction(&output_func);
    attach_output_func(output_data);
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    // move.setFunction(&move_func);
    attach_move_func(move);
    circle_agent.addAgentFunction(move);

    AgentFunctionDescription stay("stay");
    // stay.setFunction(&stay_func);
    attach_stay_func(stay);
    circle_agent.addAgentFunction(stay);

    flame_model.addAgent(circle_agent);


    AgentPopulation population(circle_agent, 5);

    for (int i=0; i< 5; i++) {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }

    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer");
    output_layer.addAgentFunction("output_data");
    simulation.addSimulationLayer(output_layer);

    /**
     * @brief      Checks if the function name exists
     * This is to perform an exception detection check. It executes the supplied
     * statement and checks if it throws the exception or not. The second argument
     * is the expected exception.
     * To test the case separately, run: make run_BOOST_TEST TSuite=SimTest/SimulationFunctionCheck
     */
    EXPECT_THROW(output_layer.addAgentFunction("output_"), InvalidAgentFunc);  // expecting an error

    SimulationLayer moveStay_layer(simulation, "move_layer");
    moveStay_layer.addAgentFunction("move");
    moveStay_layer.addAgentFunction("stay");
    simulation.addSimulationLayer(moveStay_layer);


    GTEST_COUT << "Testing simulation of functions per layers .." << std::endl;

    /**
     * @brief      Checks the number of function layers
     * This is to validate the predicate value. The test should pass.
     */
    EXPECT_EQ(simulation.getLayerCount(), 2u);

    // for each each simulation layer
    for (unsigned int i = 0; i < simulation.getLayerCount(); i++) {
        const SimulationLayer::FunctionDescriptionVector& functions = simulation.getFunctionsAtLayer(i);

        // for each function per simulation layer
        for (AgentFunctionDescription func_des : functions) {
            // check functions - printing function name only
            GTEST_COUT << "Calling agent function "<< func_des.getName() << " at layer " << i << "!" << std::endl;
        }
    }


    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population);

    cuda_model.simulate(simulation);

    /**
     * @todo : may not need this below test
     */
    EXPECT_EQ(simulation.getModelDescritpion().getName(), "circles_model");
}

#endif  // TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
