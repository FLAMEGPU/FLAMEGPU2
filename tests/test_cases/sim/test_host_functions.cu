#ifndef TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
#define TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_

#include <utility>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"


namespace {
enum FunctionType { Init, Step, Exit, HostLayer, ExitCondition };
std::vector<FunctionType> function_order;
#pragma warning(push)
#pragma warning(disable : 4100)
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
#pragma warning(pop)

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
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.addAgent(agent) first
        CUDAAgentModel cuda_model(ms->model);
        // This fails as agentMap is empty
        cuda_model.setInitialPopulationData(ms->population);
        cuda_model.simulate(ms->simulation);
        // The negative of this, is that cuda_model is inaccessible within the test!
        // So copy across population data here
        cuda_model.getPopulationData(ms->population);
    }

    void TearDown() override {
        delete ms;
        function_order.clear();
    }

    MiniSim *ms = nullptr;
};
}  // namespace


TEST_F(HostFunctionTest, ExitConditionWorks) {
    ASSERT_EQ(function_order.size(), 8llu) << "Exit condition triggered exit correctly";
}
TEST_F(HostFunctionTest, InitFuncCorrectOrder) {
    ASSERT_EQ(function_order[0], Init) << "Init function in correct place";
}
TEST_F(HostFunctionTest, HostLayerFuncCorrectOrder) {
    ASSERT_EQ(function_order[1], HostLayer) << "HostLayer function#1 in correct place";
    ASSERT_EQ(function_order[4], HostLayer) << "HostLayer function#2 in correct place";
}
TEST_F(HostFunctionTest, StepFuncCorrectOrder) {
    ASSERT_EQ(function_order[2], Step) << "Step function#1 in correct place";
    ASSERT_EQ(function_order[5], Step) << "Step function#2 in correct place";
}
TEST_F(HostFunctionTest, ExitConditionCorrectOrder) {
    ASSERT_EQ(function_order[3], ExitCondition) << "ExitCondition#1 in correct place";
    ASSERT_EQ(function_order[6], ExitCondition) << "ExitCondition#2 in correct place";
}
TEST_F(HostFunctionTest, ExitFuncCorrectOrder) {
    ASSERT_EQ(function_order[7], Exit) << "Exit function in correct place";
}
#endif  // TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
