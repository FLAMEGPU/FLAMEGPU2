#include <algorithm>
#ifndef TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
#define TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_

#include <utility>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"


namespace {
enum FunctionType { Init, Step, Exit, HostLayer, ExitCondition, Init2, Step2, Exit2, HostLayer2, ExitCondition2 };
std::vector<FunctionType> function_order;
#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
FLAMEGPU_INIT_FUNCTION(init_function) {
    function_order.push_back(Init);
}
FLAMEGPU_STEP_FUNCTION(step_function) {
    function_order.push_back(Step);
}
FLAMEGPU_EXIT_FUNCTION(exit_function) {
    function_order.push_back(Exit);
}
FLAMEGPU_HOST_FUNCTION(host_function) {
    function_order.push_back(HostLayer);
}
FLAMEGPU_EXIT_CONDITION(exit_condition) {
    function_order.push_back(ExitCondition);
    int i = 0;
    for (auto a : function_order)
        if (a == ExitCondition)
            i++;
    if (i > 1)
        return EXIT;
    return CONTINUE;
}
FLAMEGPU_INIT_FUNCTION(init_function2) {
    function_order.push_back(Init2);
}
FLAMEGPU_STEP_FUNCTION(step_function2) {
    function_order.push_back(Step2);
}
FLAMEGPU_EXIT_FUNCTION(exit_function2) {
    function_order.push_back(Exit2);
}
FLAMEGPU_HOST_FUNCTION(host_function2) {
    function_order.push_back(HostLayer2);
}
FLAMEGPU_EXIT_CONDITION(exit_condition2) {
    function_order.push_back(ExitCondition2);
    return CONTINUE;
}
#ifdef _MSC_VER
#pragma warning(pop)
#endif

class MiniSim {
 public:
    MiniSim() :
      model("model"),
      agent("agent"),
      population(agent, AGENT_COUNT),
      simulation(model),
      hostfn_layer(simulation, "hostfn_layer") {
        simulation.addInitFunction(&init_function);
        simulation.addStepFunction(&step_function);
        simulation.addExitFunction(&exit_function);
        simulation.addExitCondition(&exit_condition);

        hostfn_layer.addHostFunction(&host_function);
        simulation.addSimulationLayer(hostfn_layer);

        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentInstance instance = population.getNextInstance();
        }
        model.addAgent(agent);
        // Run until exit condition triggers
        simulation.setSimulationSteps(0);
    }
    void run() {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.addAgent(agent) first
        CUDAAgentModel cuda_model(model);
        // This fails as agentMap is empty
        cuda_model.setInitialPopulationData(population);
        ASSERT_NO_THROW(cuda_model.simulate(simulation));
        // The negative of this, is that cuda_model is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cuda_model.getPopulationData(population));
    }

    const unsigned int AGENT_COUNT = 5;
    ModelDescription model;
    AgentDescription agent;
    AgentPopulation population;
    Simulation simulation;
    SimulationLayer hostfn_layer;
};
/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class HostFunctionTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
        function_order.clear();
    }

    void TearDown() override {
        delete ms;
        function_order.clear();
    }

    MiniSim *ms = nullptr;
};
}  // namespace

/**
 * Test order
 */
TEST_F(HostFunctionTest, ExitConditionWorks) {
    ms->run();
    ASSERT_EQ(function_order.size(), 8llu) << "Exit condition triggered exit correctly";
}
TEST_F(HostFunctionTest, InitFuncCorrectOrder) {
    ms->run();
    EXPECT_EQ(function_order[0], Init) << "Init function in correct place";
}
TEST_F(HostFunctionTest, HostLayerFuncCorrectOrder) {
    ms->run();
    EXPECT_EQ(function_order[1], HostLayer) << "HostLayer function#1 in correct place";
    EXPECT_EQ(function_order[4], HostLayer) << "HostLayer function#2 in correct place";
}
TEST_F(HostFunctionTest, StepFuncCorrectOrder) {
    ms->run();
    EXPECT_EQ(function_order[2], Step) << "Step function#1 in correct place";
    EXPECT_EQ(function_order[5], Step) << "Step function#2 in correct place";
}
TEST_F(HostFunctionTest, ExitConditionCorrectOrder) {
    ms->run();
    EXPECT_EQ(function_order[3], ExitCondition) << "ExitCondition#1 in correct place";
    EXPECT_EQ(function_order[6], ExitCondition) << "ExitCondition#2 in correct place";
}
TEST_F(HostFunctionTest, ExitFuncCorrectOrder) {
    ms->run();
    EXPECT_EQ(function_order[7], Exit) << "Exit function in correct place";
}

/**
 * Test Dual Host Function support
 */
TEST_F(HostFunctionTest, InitFuncMultiple) {
    ms->simulation.setSimulationSteps(1);
    ms->simulation.addInitFunction(&init_function2);
    ms->run();
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Init), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Init2), function_order.end());
}
TEST_F(HostFunctionTest, HostLayerFuncMultiple) {
    ms->simulation.setSimulationSteps(1);
    ms->hostfn_layer.addHostFunction(&host_function2);
    ms->run();
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), HostLayer), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), HostLayer2), function_order.end());
}
TEST_F(HostFunctionTest, StepFuncMultiple) {
    ms->simulation.setSimulationSteps(1);
    ms->simulation.addStepFunction(&step_function2);
    ms->run();
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Step), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Step2), function_order.end());
}
TEST_F(HostFunctionTest, ExitConditionMultiple) {
    ms->simulation.setSimulationSteps(1);
    ms->simulation.addExitCondition(&exit_condition2);
    ms->run();
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), ExitCondition), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), ExitCondition2), function_order.end());
}
TEST_F(HostFunctionTest, ExitFuncMultiple) {
    ms->simulation.setSimulationSteps(1);
    ms->simulation.addExitFunction(&exit_function2);
    ms->run();
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Exit), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Exit2), function_order.end());
}
#endif  // TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
