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

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace {
const int TEST_LEN = 256;  // Agent count
const float MS1_VAL = 12.0f;
const float MS1_VAL2 = 36.0f;
const double MS2_VAL = 13.0;
FLAMEGPU_STEP_FUNCTION(DEFAULT_STEP) {
    EXPECT_ANY_THROW(FLAMEGPU->environment.remove<float>("asdfsdfsdf"));
}
FLAMEGPU_STEP_FUNCTION(AlignTest) {
    ASSERT_EQ(FLAMEGPU->environment.get<bool>("a"), false);
    ASSERT_EQ(FLAMEGPU->environment.get<uint64_t>("b"), static_cast<uint64_t>(UINT64_MAX / 2));
    ASSERT_EQ(FLAMEGPU->environment.get<int8_t>("c"), 12);
    ASSERT_EQ(FLAMEGPU->environment.get<int64_t>("d"), static_cast<int64_t>(INT64_MAX / 2));
    ASSERT_EQ(FLAMEGPU->environment.get<int8_t>("e"), 21);
    ASSERT_EQ(FLAMEGPU->environment.get<float>("f"), 13.0f);
}
FLAMEGPU_STEP_FUNCTION(Multi_ms1) {
    ASSERT_EQ(FLAMEGPU->environment.get<float>("ms1_float"), MS1_VAL);
    ASSERT_EQ(FLAMEGPU->environment.get<float>("ms1_float2"), MS1_VAL2);
    ASSERT_THROW(FLAMEGPU->environment.get<double>("ms2_double"), InvalidEnvProperty);
}
FLAMEGPU_STEP_FUNCTION(Multi_ms2) {
    ASSERT_EQ(FLAMEGPU->environment.get<double>("ms2_double"), MS2_VAL);
    ASSERT_EQ(FLAMEGPU->environment.get<float>("ms1_float"), static_cast<float>(MS2_VAL));
    ASSERT_EQ(FLAMEGPU->environment.get<double>("ms1_float2"), MS2_VAL);
}

class MiniSim {
 public:
    explicit MiniSim(const char *model_name = "model")
        : model(model_name)
        , agent(model.newAgent("agent"))
        , population(nullptr)
        , env(model.Environment()) {
        population = new AgentPopulation(agent, TEST_LEN);
        model.addStepFunction(DEFAULT_STEP);
    }
    ~MiniSim() {
        delete population;
    }
    void run() {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        CUDAAgentModel cuda_model(model);
        cuda_model.SimulationConfig().steps = 1;
        // This fails as agentMap is empty
        cuda_model.setPopulationData(*population);
        ASSERT_NO_THROW(cuda_model.simulate());
        // The negative of this, is that cuda_model is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cuda_model.getPopulationData(*population));
    }
    ModelDescription model;
    AgentDescription &agent;
    AgentPopulation *population;
    EnvironmentDescription &env;
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
    ms->env.add<bool>("a", static_cast<bool>(false));
    ms->env.add<uint64_t>("b", static_cast<uint64_t>(UINT64_MAX/2));
    ms->env.add<int8_t>("c", 12);
    ms->env.add<int64_t>("d", static_cast<int64_t>(INT64_MAX / 2));
    ms->env.add<int8_t>("e", 21);
    ms->env.add<float>("f", 13.0f);
    ms->model.addStepFunction(AlignTest);
    ms->run();
}

// Test bounds limit
TEST_F(EnvironmentManagerTest, OutOfMemory1) {
    std::array<char, EnvironmentManager::MAX_BUFFER_SIZE / 2> char_5kb_a;
    std::array<char, EnvironmentManager::MAX_BUFFER_SIZE / 2> char_5kb_b;
    std::array<char, EnvironmentManager::MAX_BUFFER_SIZE / 2> char_5kb_c;
    ms->env.add<char, EnvironmentManager::MAX_BUFFER_SIZE / 2>("char_5kb_a", char_5kb_a);
    ms->env.add<char, EnvironmentManager::MAX_BUFFER_SIZE / 2>("char_5kb_b", char_5kb_b);
    ms->env.add<char, EnvironmentManager::MAX_BUFFER_SIZE / 2>("char_5kb_c", char_5kb_c);
    EXPECT_THROW(ms->run(), OutOfMemory);
}
TEST_F(EnvironmentManagerTest, OutOfMemory2) {
    std::array<char, EnvironmentManager::MAX_BUFFER_SIZE / 2> char_5kb_a;
    std::array<char, EnvironmentManager::MAX_BUFFER_SIZE / 2> char_5kb_b;
    ms->env.add<char, EnvironmentManager::MAX_BUFFER_SIZE / 2>("char_5kb_a", char_5kb_a);
    ms->env.add<char, EnvironmentManager::MAX_BUFFER_SIZE / 2>("char_5kb_b", char_5kb_b);
    EXPECT_NO_THROW(ms->run());
}

// Multiple models
TEST(EnvironmentManagerTest2, MultipleModels) {
    MiniSim *ms1 = new MiniSim("ms1");
    MiniSim *ms2 = new MiniSim("ms2");
    ms1->env.add<float>("ms1_float", MS1_VAL);
    ms1->env.add<float>("ms1_float2", MS1_VAL2);
    ms2->env.add<double>("ms2_double", MS2_VAL);
    ms2->env.add<float>("ms1_float", static_cast<float>(MS2_VAL));
    ms2->env.add<double>("ms1_float2", MS2_VAL);
    ms1->model.addStepFunction(Multi_ms1);
    ms2->model.addStepFunction(Multi_ms2);
    EXPECT_NO_THROW(ms1->run());
    EXPECT_NO_THROW(ms2->run());
    delete ms1;
    delete ms2;
}

// Free/Rehost model
// This test will need expanding once Environment manager is separated into host and device management so the EnvDescriptionAlreadyLoaded can be thrown at construction rather than first use.
TEST(EnvironmentManagerTest2, RehostModel) {
    MiniSim *ms1 = new MiniSim();
    ms1->env.add<float>("ms1_float", MS1_VAL);
    CUDAAgentModel *cuda_model1 = new CUDAAgentModel(ms1->model);
    cuda_model1->SimulationConfig().steps = 1;
    cuda_model1->setPopulationData(*ms1->population);
    EXPECT_NO_THROW(cuda_model1->simulate());
    MiniSim *ms2 = new MiniSim();
    ms2->env.add<float>("ms1_float", MS1_VAL);
    {
        CUDAAgentModel *cuda_model2 = nullptr;
        cuda_model2 = new CUDAAgentModel(ms2->model);
        // Errors because ms1 CUDAAgentModel still alive, but a method must be called on it to initialise the device (rather than error at construction)
        EXPECT_THROW(cuda_model2->step(), EnvDescriptionAlreadyLoaded);
        delete cuda_model2;
    }
    delete cuda_model1;
    // Try again now ms1 has been deleted
    CUDAAgentModel *cuda_model2 = nullptr;
    cuda_model2 = new CUDAAgentModel(ms2->model);
    EXPECT_NO_THROW(cuda_model2->step());
    cuda_model2->setPopulationData(*ms2->population);
    cuda_model2->SimulationConfig().steps = 1;
    EXPECT_NO_THROW(cuda_model2->simulate());
    delete cuda_model2;
    delete ms1;
    delete ms2;
}
