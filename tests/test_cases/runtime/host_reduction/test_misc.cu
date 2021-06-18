#include "helpers/host_reductions_common.h"

namespace flamegpu {

namespace test_host_reductions {
FLAMEGPU_CUSTOM_REDUCTION(customMax2, a, b) {
    return a > b ? a : b;
}
FLAMEGPU_CUSTOM_REDUCTION(customSum2, a, b) {
    return a + b;
}
FLAMEGPU_CUSTOM_TRANSFORM(customTransform2, a) {
    return a <= 0 ? 1 : 0;
}

FLAMEGPU_STEP_FUNCTION(step_sumException) {
    EXPECT_THROW(FLAMEGPU->agent("agedddnt"), InvalidAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<unsigned char>("float"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<int64_t>("uint64_t"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<double>("intsssssssss16_t"), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<uint64_t>("isssssssssssnt"), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_minException) {
    EXPECT_THROW(FLAMEGPU->agent("agsssedddnt"), InvalidAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<uint64_t>("char"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<int64_t>("uint64_t"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<double>("intssssssssssssss16_t"), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<uint64_t>("issssssssssssnt"), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_maxException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<double>("float"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<float>("uint64_t"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<double>("intsssssssssss16_t"), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<uint64_t>("ssssssssssssssint"), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_customReductionException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<double>("float", customMax2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<float>("uint64_t", customMax2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<double>("intsssssssssss16_t", customMax2, 0), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<uint64_t>("ssssssssssssssint", customMax2, 0), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<double>("float", 10, 0, 10), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<float>("uint64_t", 10, 0, 10), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<double>("intsssssssssss16_t", 10, 0, 10), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<uint64_t>("ssssssssssssssint", 10, 0, 10), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<int>("int", 10, 0, 0), InvalidArgument);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<double>("double", 10, 11, 10), InvalidArgument);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<int32_t>("uint16_t", customTransform2, customSum2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<float>("uint64_t", customTransform2, customSum2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<double>("intsssssssssss16_t", customTransform2, customSum2, 0), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<uint64_t>("ssssssssssssssint", customTransform2, customSum2, 0), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_countException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").count<int32_t>("double", 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").count<float>("uint64_t", 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").count<double>("intsssssssssss16_t", 0), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").count<uint64_t>("ssssssssssssssint", 0), InvalidAgentVar);
}

/**
 * Bad Types
 */
TEST_F(HostReductionTest, SumException) {
    ms->model.addStepFunction(step_sumException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
    }
    ms->run();
}
TEST_F(HostReductionTest, MinException) {
    ms->model.addStepFunction(step_minException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
    }
    ms->run();
}
TEST_F(HostReductionTest, MaxException) {
    ms->model.addStepFunction(step_maxException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
    }
    ms->run();
}
TEST_F(HostReductionTest, CustomReductionException) {
    ms->model.addStepFunction(step_customReductionException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
    }
    ms->run();
}
TEST_F(HostReductionTest, HistogramEvenException) {
    ms->model.addStepFunction(step_histogramEvenException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
    }
    ms->run();
}
TEST_F(HostReductionTest, CustomTransformException) {
    ms->model.addStepFunction(step_transformReduceException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
    }
    ms->run();
}
TEST_F(HostReductionTest, CountException) {
    ms->model.addStepFunction(step_countException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
    }
    ms->run();
}

    FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_sum1) {
    FLAMEGPU->agent("agent_name").sum<int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_sum2) {
    FLAMEGPU->agent("agent_name").sum<int, int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_min) {
    FLAMEGPU->agent("agent_name").min<int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_max) {
    FLAMEGPU->agent("agent_name").max<int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_count) {
    FLAMEGPU->agent("agent_name").count<int>("array_var", 0);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_hist1) {
    FLAMEGPU->agent("agent_name").histogramEven<int>("array_var", 10, 0, 9);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_hist2) {
    FLAMEGPU->agent("agent_name").histogramEven<int, int>("array_var", 10, 0, 9);
}
FLAMEGPU_CUSTOM_REDUCTION(SampleReduction, a, b) {
        return a + b;
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_reduce) {
    FLAMEGPU->agent("agent_name").reduce<int>("array_var", SampleReduction, 0);
}
FLAMEGPU_CUSTOM_TRANSFORM(SampleTransform, a) {
    return a + 1;
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_transformReduce) {
    FLAMEGPU->agent("agent_name").transformReduce<int>("array_var", SampleTransform, SampleReduction, 0);
}
// Array variables
const unsigned int AGENT_COUNT = 1024;
TEST(HostMiscTest, ArrayVarNotSupported_sum1) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_sum1);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_sum2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_sum2);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_min) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_min);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_max) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_max);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_count) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_count);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_hist1) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_hist1);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_hist2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_hist2);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_reduce) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_reduce);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
TEST(HostMiscTest, ArrayVarNotSupported_transformReduce) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentVector init_population(agent, AGENT_COUNT);
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_transformReduce);
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), UnsupportedVarType);
}
}  // namespace test_host_reductions
}  // namespace flamegpu
