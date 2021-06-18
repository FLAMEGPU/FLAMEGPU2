#ifndef TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
#define TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_

#include <algorithm>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"
namespace flamegpu {


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
      agent(model.newAgent("agent")),
      population(agent, AGENT_COUNT),
      hostfn_layer(model.newLayer("hostfn_layer")) {
        model.addInitFunction(init_function);
        model.addStepFunction(step_function);
        model.addExitFunction(exit_function);
        model.addExitCondition(exit_condition);

        hostfn_layer.addHostFunction(host_function);
    }
    void run(unsigned int steps = 0) {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        CUDASimulation cudaSimulation(model);
        cudaSimulation.SimulationConfig().steps = steps;
        // This fails as agentMap is empty
        cudaSimulation.setPopulationData(population);
        ASSERT_NO_THROW(cudaSimulation.simulate());
        // The negative of this, is that cudaSimulation is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    }

    const unsigned int AGENT_COUNT = 5;
    ModelDescription model;
    AgentDescription &agent;
    AgentVector population;
    LayerDescription &hostfn_layer;
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
    ASSERT_EQ(function_order.size(), 0u);
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
    ms->model.addInitFunction(init_function2);
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), Init), function_order.end());
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), Init2), function_order.end());
    ms->run(1);
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Init), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Init2), function_order.end());
}
TEST_F(HostFunctionTest, HostLayerFuncMultiple) {
    EXPECT_THROW(ms->hostfn_layer.addHostFunction(host_function2), InvalidLayerMember);
}
TEST_F(HostFunctionTest, StepFuncMultiple) {
    ms->model.addStepFunction(step_function2);
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), Step), function_order.end());
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), Step2), function_order.end());
    ms->run(1);
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Step), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Step2), function_order.end());
}
TEST_F(HostFunctionTest, ExitConditionMultiple) {
    ms->model.addExitCondition(exit_condition2);
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), ExitCondition), function_order.end());
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), ExitCondition2), function_order.end());
    ms->run(1);
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), ExitCondition), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), ExitCondition2), function_order.end());
}
TEST_F(HostFunctionTest, ExitFuncMultiple) {
    ms->model.addExitFunction(exit_function2);
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), Exit), function_order.end());
    EXPECT_EQ(std::find(function_order.begin(), function_order.end(), Exit2), function_order.end());
    ms->run(1);
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Exit), function_order.end());
    EXPECT_NE(std::find(function_order.begin(), function_order.end(), Exit2), function_order.end());
}

/**
 * Test Duplication Host Function exception thrown
 */
TEST_F(HostFunctionTest, InitFuncDuplicateException) {
    EXPECT_THROW(ms->model.addInitFunction(init_function), InvalidHostFunc);
}
TEST_F(HostFunctionTest, HostLayerFuncDuplicateException) {
    EXPECT_THROW(ms->hostfn_layer.addHostFunction(host_function), InvalidLayerMember);
}
TEST_F(HostFunctionTest, StepFuncDuplicateException) {
    EXPECT_THROW(ms->model.addStepFunction(step_function), InvalidHostFunc);
}
TEST_F(HostFunctionTest, ExitConditionDuplicateException) {
    EXPECT_THROW(ms->model.addExitCondition(exit_condition), InvalidHostFunc);
}
TEST_F(HostFunctionTest, ExitFuncDuplicateException) {
    EXPECT_THROW(ms->model.addExitFunction(exit_function), InvalidHostFunc);
}

/**
 * Special case, host function can be added to multiple different layers
 */
TEST_F(HostFunctionTest, HostLayerFuncDuplicateLayerNoException) {
    EXPECT_EQ(std::count(function_order.begin(), function_order.end(), HostLayer), 0u);
    ASSERT_NO_THROW(ms->run(1));
    EXPECT_EQ(std::count(function_order.begin(), function_order.end(), HostLayer), 1u);
    function_order.clear();
    EXPECT_EQ(std::count(function_order.begin(), function_order.end(), HostLayer), 0u);
    LayerDescription &hostfn_layer2 = ms->model.newLayer("hostfn_layer2");
    ASSERT_NO_THROW(hostfn_layer2.addHostFunction(host_function));
    ASSERT_NO_THROW(ms->run(1));
    EXPECT_EQ(std::count(function_order.begin(), function_order.end(), HostLayer), 2u);
}
}  // namespace flamegpu
#endif  // TESTS_TEST_CASES_SIM_TEST_SIM_VALIDATION_H_
