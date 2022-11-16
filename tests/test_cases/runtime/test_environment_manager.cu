/**
 * Tests of class: EnvironmentManager
 * Most of this class is tested indirectly through the linked classes AgentEnvironment, HostEnvironment
 * 
 * Tests cover:
 * > init() [does it work, can we host multiple models]
 * > free() [does it work, can we re-host a model]
 * > Out of space exception
 * Implied tests: (Covered as a result of other tests)
 * > defrag() [init uses this]
 */

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace {
const int TEST_LEN = 256;  // Agent count
const float MS1_VAL = 12.0f;
const float MS1_VAL2 = 36.0f;
const double MS2_VAL = 13.0;
FLAMEGPU_STEP_FUNCTION(DEFAULT_STEP) {
    // Do nothing
}
FLAMEGPU_STEP_FUNCTION(AlignTest) {
    ASSERT_EQ(FLAMEGPU->environment.getProperty<bool>("a"), false);
    ASSERT_EQ(FLAMEGPU->environment.getProperty<uint64_t>("b"), static_cast<uint64_t>(UINT64_MAX / 2));
    ASSERT_EQ(FLAMEGPU->environment.getProperty<int8_t>("c"), 12);
    ASSERT_EQ(FLAMEGPU->environment.getProperty<int64_t>("d"), static_cast<int64_t>(INT64_MAX / 2));
    ASSERT_EQ(FLAMEGPU->environment.getProperty<int8_t>("e"), 21);
    ASSERT_EQ(FLAMEGPU->environment.getProperty<float>("f"), 13.0f);
}
FLAMEGPU_STEP_FUNCTION(Multi_ms1) {
    ASSERT_EQ(FLAMEGPU->environment.getProperty<float>("ms1_float"), MS1_VAL);
    ASSERT_EQ(FLAMEGPU->environment.getProperty<float>("ms1_float2"), MS1_VAL2);
    ASSERT_THROW(FLAMEGPU->environment.getProperty<double>("ms2_double"), exception::InvalidEnvProperty);
}
FLAMEGPU_STEP_FUNCTION(Multi_ms2) {
    ASSERT_EQ(FLAMEGPU->environment.getProperty<double>("ms2_double"), MS2_VAL);
    ASSERT_EQ(FLAMEGPU->environment.getProperty<float>("ms1_float"), static_cast<float>(MS2_VAL));
    ASSERT_EQ(FLAMEGPU->environment.getProperty<double>("ms1_float2"), MS2_VAL);
}

class MiniSim {
 public:
    explicit MiniSim(const char *model_name = "model")
        : model(model_name)
        , agent(model.newAgent("agent"))
        , population(nullptr)
        , env(model.Environment()) {
        population = new AgentVector(agent, TEST_LEN);
        model.addStepFunction(DEFAULT_STEP);
    }
    ~MiniSim() {
        delete population;
    }
    void run() {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        CUDASimulation cudaSimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        // This fails as agentMap is empty
        cudaSimulation.setPopulationData(*population);
        ASSERT_NO_THROW(cudaSimulation.simulate());
        // The negative of this, is that cudaSimulation is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cudaSimulation.getPopulationData(*population));
    }
    ModelDescription model;
    AgentDescription agent;
    AgentVector *population;
    EnvironmentDescription env;
};
/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class EnvironmentManagerTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
    }

    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};
}  // namespace


// Test alignment
TEST_F(EnvironmentManagerTest, Alignment) {
    ms->env.newProperty<bool>("a", static_cast<bool>(false));
    ms->env.newProperty<uint64_t>("b", static_cast<uint64_t>(UINT64_MAX/2));
    ms->env.newProperty<int8_t>("c", 12);
    ms->env.newProperty<int64_t>("d", static_cast<int64_t>(INT64_MAX / 2));
    ms->env.newProperty<int8_t>("e", 21);
    ms->env.newProperty<float>("f", 13.0f);
    ms->model.addStepFunction(AlignTest);
    ms->run();
}

// Multiple models
TEST(EnvironmentManagerTest2, MultipleModels) {
    MiniSim *ms1 = new MiniSim("ms1");
    MiniSim *ms2 = new MiniSim("ms2");
    ms1->env.newProperty<float>("ms1_float", MS1_VAL);
    ms1->env.newProperty<float>("ms1_float2", MS1_VAL2);
    ms2->env.newProperty<double>("ms2_double", MS2_VAL);
    ms2->env.newProperty<float>("ms1_float", static_cast<float>(MS2_VAL));
    ms2->env.newProperty<double>("ms1_float2", MS2_VAL);
    ms1->model.addStepFunction(Multi_ms1);
    ms2->model.addStepFunction(Multi_ms2);
    EXPECT_NO_THROW(ms1->run());
    EXPECT_NO_THROW(ms2->run());
    delete ms1;
    delete ms2;
}
}  // namespace flamegpu
