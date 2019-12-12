#include "helpers/host_reductions_common.h"

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
    EXPECT_THROW(FLAMEGPU->agent("agedddnt"), InvalidCudaAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<unsigned char>("float"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<int64_t>("uint64_t"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<double>("intsssssssss16_t"), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").sum<uint64_t>("isssssssssssnt"), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_minException) {
    EXPECT_THROW(FLAMEGPU->agent("agsssedddnt"), InvalidCudaAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<uint64_t>("char"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<int64_t>("uint64_t"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<double>("intssssssssssssss16_t"), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").min<uint64_t>("issssssssssssnt"), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_maxException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidCudaAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<double>("float"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<float>("uint64_t"), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<double>("intsssssssssss16_t"), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").max<uint64_t>("ssssssssssssssint"), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_customReductionException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidCudaAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<double>("float", customMax2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<float>("uint64_t", customMax2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<double>("intsssssssssss16_t", customMax2, 0), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").reduce<uint64_t>("ssssssssssssssint", customMax2, 0), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidCudaAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<double>("float", 10, 0, 10), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<float>("uint64_t", 10, 0, 10), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<double>("intsssssssssss16_t", 10, 0, 10), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<uint64_t>("ssssssssssssssint", 10, 0, 10), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<int>("int", 10, 0, 0), InvalidArgument);
    EXPECT_THROW(FLAMEGPU->agent("agent").histogramEven<double>("double", 10, 11, 10), InvalidArgument);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidCudaAgent);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<int32_t>("uint16_t", customTransform2, customSum2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<float>("uint64_t", customTransform2, customSum2, 0), InvalidVarType);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<double>("intsssssssssss16_t", customTransform2, customSum2, 0), InvalidAgentVar);
    EXPECT_THROW(FLAMEGPU->agent("agent").transformReduce<uint64_t>("ssssssssssssssint", customTransform2, customSum2, 0), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(step_countException) {
    EXPECT_THROW(FLAMEGPU->agent("ageaadddnt"), InvalidCudaAgent);
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
        AgentInstance instance = ms->population->getNextInstance();
    }
    ms->run();
}
TEST_F(HostReductionTest, MinException) {
    ms->model.addStepFunction(step_minException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
    }
    ms->run();
}
TEST_F(HostReductionTest, MaxException) {
    ms->model.addStepFunction(step_maxException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
    }
    ms->run();
}
TEST_F(HostReductionTest, CustomReductionException) {
    ms->model.addStepFunction(step_customReductionException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
    }
    ms->run();
}
TEST_F(HostReductionTest, HistogramEvenException) {
    ms->model.addStepFunction(step_histogramEvenException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
    }
    ms->run();
}
TEST_F(HostReductionTest, CustomTransformException) {
    ms->model.addStepFunction(step_transformReduceException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
    }
    ms->run();
}
TEST_F(HostReductionTest, CountException) {
    ms->model.addStepFunction(step_countException);
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
    }
    ms->run();
}
}  // namespace test_host_reductions
