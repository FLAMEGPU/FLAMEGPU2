#ifndef TESTS_TEST_CASES_RUNTIME_TEST_HOST_RANDOM_H_
#define TESTS_TEST_CASES_RUNTIME_TEST_HOST_RANDOM_H_

#include <array>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

namespace {
const unsigned int TEST_LEN = 256;

std::array<float, TEST_LEN> float_out;
std::array<double, TEST_LEN> double_out;
std::array<unsigned char, TEST_LEN> unsigned_char_out;
std::array<char, TEST_LEN> char_out;
std::array<uint16_t, TEST_LEN> unsigned_short_out;
std::array<int16_t, TEST_LEN> short_out;
std::array<uint32_t, TEST_LEN> unsigned_int_out;
std::array<int32_t, TEST_LEN> int_out;
std::array<uint64_t, TEST_LEN> unsigned_longlong_out;
std::array<int64_t, TEST_LEN> longlong_out;
FLAMEGPU_STEP_FUNCTION(step_uniform_float) {
    for (float &i : float_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<float>());
}
FLAMEGPU_STEP_FUNCTION(step_uniform_double) {
    for (double &i : double_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<double>());
}
FLAMEGPU_STEP_FUNCTION(step_normal_float) {
    for (float &i : float_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.normal<float>());
}
FLAMEGPU_STEP_FUNCTION(step_normal_double) {
    for (double &i : double_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.normal<double>());
}
FLAMEGPU_STEP_FUNCTION(step_logNormal_float) {
    for (float &i : float_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.logNormal<float>(0, 1));
}
FLAMEGPU_STEP_FUNCTION(step_logNormal_double) {
    for (double &i : double_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.logNormal<double>(0, 1));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_uchar) {
    for (unsigned char &i : unsigned_char_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<unsigned char>(0, UCHAR_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_char) {
    for (char &i : char_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<char>(CHAR_MIN, CHAR_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_ushort) {
    for (uint16_t &i : unsigned_short_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<uint16_t>(0, UINT16_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_short) {
    for (int16_t &i : short_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<int16_t>(INT16_MIN, INT16_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_uint) {
    for (uint32_t &i : unsigned_int_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<uint32_t>(0, UINT32_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_int) {
    for (int32_t &i : int_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<int32_t>(INT32_MIN, INT32_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_ulonglong) {
    for (uint64_t &i : unsigned_longlong_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<uint64_t>(0, UINT64_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_longlong) {
    for (int64_t &i : longlong_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<int64_t>(INT64_MIN, INT64_MAX));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_uchar_range) {
    for (auto &i : unsigned_char_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<unsigned char>(
            static_cast<unsigned char>(UCHAR_MAX * 0.25),
            static_cast<unsigned char>(UCHAR_MAX * 0.75)));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_char_range) {
    for (char &i : char_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<char>(
            static_cast<char>(CHAR_MIN * 0.5),
            static_cast<char>(CHAR_MAX * 0.5)));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_ushort_range) {
    for (auto &i : unsigned_short_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<uint16_t>(
            static_cast<uint16_t>(UINT16_MAX * 0.25),
            static_cast<uint16_t>(UINT16_MAX * 0.75)));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_short_range) {
    for (auto &i : short_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<int16_t>(
            static_cast<int16_t>(INT16_MIN * 0.5),
            static_cast<int16_t>(INT16_MAX * 0.5)));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_uint_range) {
    for (auto &i : unsigned_int_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<unsigned int>(
            static_cast<unsigned int>(UINT_MAX * 0.25),
            static_cast<unsigned int>(UINT_MAX * 0.75)));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_int_range) {
    for (auto &i : int_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<int>(
            static_cast<int>(INT_MIN * 0.5),
            static_cast<int>(INT_MAX * 0.5)));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_ulonglong_range) {
    for (auto &i : unsigned_longlong_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<uint64_t>(
            static_cast<uint64_t>(UINT64_MAX * 0.25),
            static_cast<uint64_t>(UINT64_MAX * 0.75)));
}
FLAMEGPU_STEP_FUNCTION(step_uniform_longlong_range) {
    for (auto &i : longlong_out)
        ASSERT_NO_THROW(i = FLAMEGPU->random.uniform<int64_t>(
            static_cast<int64_t>(INT64_MIN >> 1),
            static_cast<int64_t>(INT64_MAX >> 1)));
}
class MiniSim {
 public:
    MiniSim() :
      model("model"),
      agent("agent"),
      population(agent, AGENT_COUNT),
      simulation(model) {
        // agent.addAgentVariable<float>("float");
        // agent.addAgentVariable<double>("double");
        // agent.addAgentVariable<unsigned char>("unsigned_char");
        // agent.addAgentVariable<char>("char");
        // agent.addAgentVariable<uint16_t>("uint16_t");
        // agent.addAgentVariable<int16_t>("int16_t");
        // agent.addAgentVariable<unsigned int>("unsigned_int");
        // agent.addAgentVariable<int>("int");
        // agent.addAgentVariable<uint64_t>("uint64_t");
        // agent.addAgentVariable<int64_t>("int64_t");

        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentInstance instance = population.getNextInstance();
            // instance.setVariable<float>("float", 0);
            // instance.setVariable<double>("double", 0);
            // instance.setVariable<unsigned char>("unsigned_char", 0);
            // instance.setVariable<char>("char", 0);
            // instance.setVariable<uint16_t>("uint16_t", 0);
            // instance.setVariable<int16_t>("int16_t", 0);
            // instance.setVariable<unsigned int>("unsigned_int", 0);
            // instance.setVariable<int>("int", 0);
            // instance.setVariable<uint64_t>("uint64_t", 0);
            // instance.setVariable<int64_t>("int64_t", 0);
        }
        model.addAgent(agent);
        simulation.setSimulationSteps(1);
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
};
/**
 * This defines a common fixture used as a base for all test cases in the file
 * @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
 */
class HostRandomTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
        for (float&i : float_out)
            i = 0.0f;
        for (double&i : double_out)
            i = 0.0;
        for (unsigned char&i : unsigned_char_out)
            i = 0;
        for (char&i : char_out)
            i = 0;
        for (uint16_t&i : unsigned_short_out)
            i = 0;
        for (int16_t&i : short_out)
            i = 0;
        for (uint32_t&i : unsigned_int_out)
            i = 0u;
        for (int32_t&i : int_out)
            i = 0;
        for (uint64_t&i : unsigned_longlong_out)
            i = 0llu;
        for (int64_t&i : longlong_out)
            i = 0ll;
    }

    void TearDown() override {
        delete ms;
    }

    std::string _t_unused = std::string();
    MiniSim *ms = nullptr;
};

// @note seeds 0 and 1 conflict with std::linear_congruential_engine, the default on GCC so using mt19937 to avoid this.
const char *args_1[4] = { "process.exe", "input.xml", "-r", "0" };
const char *args_2[4] = { "process.exe", "input.xml", "-r", "1" };

}  // namespace


TEST_F(HostRandomTest, UniformFloat) {
    ms->simulation.addStepFunction(&step_uniform_float);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (float&i : float_out)
        EXPECT_EQ(i, 0.0f);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < float_out.size(); ++i)
        for (unsigned int j = 0; j < float_out.size(); ++j)
            if (i != j)
                if (float_out[i] != float_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<float, TEST_LEN> _float_out = float_out;
    for (float&i : float_out)
        i = 0.0f;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < float_out.size(); ++i)
        if (float_out[i] != _float_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (float&i : float_out)
        i = 0.0f;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < float_out.size(); ++i)
        EXPECT_EQ(float_out[i], _float_out[i]);
}
TEST_F(HostRandomTest, UniformDouble) {
    ms->simulation.addStepFunction(&step_uniform_double);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (double&i : double_out)
        EXPECT_EQ(i, 0.0);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < double_out.size(); ++i)
        for (unsigned int j = 0; j < double_out.size(); ++j)
            if (i != j)
                if (double_out[i] != double_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<double, TEST_LEN> _double_out = double_out;
    for (double&i : double_out)
        i = 0.0;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < double_out.size(); ++i)
        if (double_out[i] != _double_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (double&i : double_out)
        i = 0.0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < double_out.size(); ++i)
        EXPECT_EQ(double_out[i], _double_out[i]);
}

TEST_F(HostRandomTest, NormalFloat) {
    ms->simulation.addStepFunction(&step_normal_float);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (float&i : float_out)
        EXPECT_EQ(i, 0.0f);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < float_out.size(); ++i)
        for (unsigned int j = 0; j < float_out.size(); ++j)
            if (i != j)
                if (float_out[i] != float_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<float, TEST_LEN> _float_out = float_out;
    for (float&i : float_out)
        i = 0.0f;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < float_out.size(); ++i)
        if (float_out[i] != _float_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (float&i : float_out)
        i = 0.0f;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < float_out.size(); ++i)
        EXPECT_EQ(float_out[i], _float_out[i]);
}
TEST_F(HostRandomTest, NormalDouble) {
    ms->simulation.addStepFunction(&step_normal_double);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (double&i : double_out)
        EXPECT_EQ(i, 0.0);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < double_out.size(); ++i)
        for (unsigned int j = 0; j < double_out.size(); ++j)
            if (i != j)
                if (double_out[i] != double_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<double, TEST_LEN> _double_out = double_out;
    for (double&i : double_out)
        i = 0.0;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < double_out.size(); ++i)
        if (double_out[i] != _double_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (double&i : double_out)
        i = 0.0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < double_out.size(); ++i)
        EXPECT_EQ(double_out[i], _double_out[i]);
}

TEST_F(HostRandomTest, LogNormalFloat) {
    ms->simulation.addStepFunction(&step_logNormal_float);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (float &i : float_out)
        EXPECT_EQ(i, 0.0f);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (float&i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < float_out.size(); ++i)
        for (unsigned int j = 0; j < float_out.size(); ++j)
            if (i != j)
                if (float_out[i] != float_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<float, TEST_LEN> _float_out = float_out;
    for (float&i : float_out)
        i = 0.0f;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < float_out.size(); ++i)
        if (float_out[i] != _float_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (float&i : float_out)
        i = 0.0f;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (float &i : float_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < float_out.size(); ++i)
        EXPECT_EQ(float_out[i], _float_out[i]);
}
TEST_F(HostRandomTest, LogNormalDouble) {
    ms->simulation.addStepFunction(&step_logNormal_double);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (double&i : double_out)
        EXPECT_EQ(i, 0.0);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < double_out.size(); ++i)
        for (unsigned int j = 0; j < double_out.size(); ++j)
            if (i != j)
                if (double_out[i] != double_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<double, TEST_LEN> _double_out = double_out;
    for (double&i : double_out)
        i = 0.0;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < double_out.size(); ++i)
        if (double_out[i] != _double_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (double&i : double_out)
        i = 0.0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (double &i : double_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < double_out.size(); ++i)
        EXPECT_EQ(double_out[i], _double_out[i]);
}

TEST_F(HostRandomTest, UniformUChar) {
    ms->simulation.addStepFunction(&step_uniform_uchar);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (unsigned char&i : unsigned_char_out)
        EXPECT_EQ(i, 0);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (unsigned char &i : unsigned_char_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < unsigned_char_out.size(); ++i)
        for (unsigned int j = 0; j < unsigned_char_out.size(); ++j)
            if (i != j)
                if (unsigned_char_out[i] != unsigned_char_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<unsigned char, TEST_LEN> _unsigned_char_out = unsigned_char_out;
    for (unsigned char&i : unsigned_char_out)
        i = 0;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (unsigned char &i : unsigned_char_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < unsigned_char_out.size(); ++i)
        if (unsigned_char_out[i] != _unsigned_char_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (unsigned char&i : unsigned_char_out)
        i = 0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (unsigned char &i : unsigned_char_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < unsigned_char_out.size(); ++i)
        EXPECT_EQ(unsigned_char_out[i], _unsigned_char_out[i]);
}
TEST_F(HostRandomTest, UniformChar) {
    ms->simulation.addStepFunction(&step_uniform_char);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (char&i : char_out)
        EXPECT_EQ(i, 0);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (char&i : char_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < char_out.size(); ++i)
        for (unsigned int j = 0; j < char_out.size(); ++j)
            if (i != j)
                if (char_out[i] != char_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<char, TEST_LEN> _char_out = char_out;
    for (char&i : char_out)
        i = 0;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (char&i : char_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < char_out.size(); ++i)
        if (char_out[i] != _char_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (char&i : char_out)
        i = 0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (char&i : char_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < char_out.size(); ++i)
        EXPECT_EQ(char_out[i], _char_out[i]);
}

TEST_F(HostRandomTest, UniformUShort) {
    ms->simulation.addStepFunction(&step_uniform_ushort);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (uint16_t &i : unsigned_short_out)
        EXPECT_EQ(i, 0u);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (uint16_t &i : unsigned_short_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < unsigned_short_out.size(); ++i)
        for (unsigned int j = 0; j < unsigned_short_out.size(); ++j)
            if (i != j)
                if (unsigned_short_out[i] != unsigned_short_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<uint16_t, TEST_LEN> _unsigned_short_out = unsigned_short_out;
    for (uint16_t &i : unsigned_short_out)
        i = 0u;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (uint16_t &i : unsigned_short_out)
        if (i != 0u)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < unsigned_short_out.size(); ++i)
        if (unsigned_short_out[i] != _unsigned_short_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (uint16_t &i : unsigned_short_out)
        i = 0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (uint16_t &i : unsigned_short_out)
        if (i != 0ll)
            diff++;
    // Old Seed == old values
    for (unsigned int i = 0; i < unsigned_short_out.size(); ++i)
        EXPECT_EQ(unsigned_short_out[i], _unsigned_short_out[i]);
}
TEST_F(HostRandomTest, UniformShort) {
    ms->simulation.addStepFunction(&step_uniform_short);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (int16_t &i : short_out)
        EXPECT_EQ(i, 0);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (int16_t &i : short_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < short_out.size(); ++i)
        for (unsigned int j = 0; j < short_out.size(); ++j)
            if (i != j)
                if (short_out[i] != short_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<int16_t, TEST_LEN> _short_out = short_out;
    for (int16_t &i : short_out)
        i = 0;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (int16_t &i : short_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < short_out.size(); ++i)
        if (short_out[i] != _short_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (int16_t &i : short_out)
        i = 0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (int16_t &i : short_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < short_out.size(); ++i)
        EXPECT_EQ(short_out[i], _short_out[i]);
}

TEST_F(HostRandomTest, UniformUInt) {
    ms->simulation.addStepFunction(&step_uniform_uint);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (unsigned int&i : unsigned_int_out)
        EXPECT_EQ(i, 0u);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (unsigned int &i : unsigned_int_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < unsigned_int_out.size(); ++i)
        for (unsigned int j = 0; j < unsigned_int_out.size(); ++j)
            if (i != j)
                if (unsigned_int_out[i] != unsigned_int_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<unsigned int, TEST_LEN> _unsigned_int_out = unsigned_int_out;
    for (unsigned int&i : unsigned_int_out)
        i = 0u;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (unsigned int &i : unsigned_int_out)
        if (i != 0u)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < unsigned_int_out.size(); ++i)
        if (unsigned_int_out[i] != _unsigned_int_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (unsigned int&i : unsigned_int_out)
        i = 0u;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (unsigned int &i : unsigned_int_out)
        if (i != 0u)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < unsigned_int_out.size(); ++i)
        EXPECT_EQ(unsigned_int_out[i], _unsigned_int_out[i]);
}
TEST_F(HostRandomTest, UniformInt) {
    ms->simulation.addStepFunction(&step_uniform_int);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (int&i : int_out)
        EXPECT_EQ(i, 0);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (int32_t &i : int_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < int_out.size(); ++i)
        for (unsigned int j = 0; j < int_out.size(); ++j)
            if (i != j)
                if (int_out[i] != int_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<int, TEST_LEN> _int_out = int_out;
    for (int&i : int_out)
        i = 0;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (int32_t &i : int_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < int_out.size(); ++i)
        if (int_out[i] != _int_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (int&i : int_out)
        i = 0;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (int32_t &i : int_out)
        if (i != 0)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < int_out.size(); ++i)
        EXPECT_EQ(int_out[i], _int_out[i]);
}

TEST_F(HostRandomTest, UniformULongLong) {
    ms->simulation.addStepFunction(&step_uniform_ulonglong);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (uint64_t &i : unsigned_longlong_out)
        EXPECT_EQ(i, 0llu);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (uint64_t &i : unsigned_longlong_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < unsigned_longlong_out.size(); ++i)
        for (unsigned int j = 0; j < unsigned_longlong_out.size(); ++j)
            if (i != j)
                if (unsigned_longlong_out[i] != unsigned_longlong_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<uint64_t, TEST_LEN> _unsigned_longlong_out = unsigned_longlong_out;
    for (uint64_t &i : unsigned_longlong_out)
        i = 0llu;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (uint64_t &i : unsigned_longlong_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < unsigned_longlong_out.size(); ++i)
        if (unsigned_longlong_out[i] != _unsigned_longlong_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (uint64_t &i : unsigned_longlong_out)
        i = 0llu;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (uint64_t &i : unsigned_longlong_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < unsigned_longlong_out.size(); ++i)
        EXPECT_EQ(unsigned_longlong_out[i], _unsigned_longlong_out[i]);
}
TEST_F(HostRandomTest, UniformLongLong) {
    ms->simulation.addStepFunction(&step_uniform_longlong);
    // Seed RNG
    ms->simulation.checkArgs(4, args_1, _t_unused);
    // Initially 0
    for (int64_t &i : longlong_out)
        EXPECT_EQ(i, 0ll);
    ms->run();
    // Value has changed
    unsigned int diff = 0;
    for (int64_t &i : longlong_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Multiple calls == different values
    diff = 0;
    for (unsigned int i = 0; i < longlong_out.size(); ++i)
        for (unsigned int j = 0; j < longlong_out.size(); ++j)
            if (i != j)
                if (longlong_out[i] != longlong_out[j])
                    diff++;
    EXPECT_GT(diff, 0u);
    std::array<int64_t, TEST_LEN> _longlong_out = longlong_out;
    for (int64_t &i : longlong_out)
        i = 0ll;
    // Different Seed
    ms->simulation.checkArgs(4, args_2, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (int64_t &i : longlong_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // New Seed == new sequence
    diff = 0;
    for (unsigned int i = 0; i < longlong_out.size(); ++i)
        if (longlong_out[i]!= _longlong_out[i])
            diff++;
    EXPECT_GT(diff, 0u);
    for (int64_t &i : longlong_out)
        i = 0ll;
    // First Seed
    ms->simulation.checkArgs(4, args_1, _t_unused);
    ms->run();
    // Value has changed
    diff = 0;
    for (int64_t &i : longlong_out)
        if (i != 0ll)
            diff++;
    EXPECT_GT(diff, 0u);
    // Old Seed == old values
    for (unsigned int i = 0; i < longlong_out.size(); ++i)
        EXPECT_EQ(longlong_out[i], _longlong_out[i]);
}

/**
 * Range tests
 */
TEST_F(HostRandomTest, UniformFloatRange) {
    ms->simulation.addStepFunction(&step_uniform_float);
    ms->run();
    for (auto &i : float_out) {
        EXPECT_GE(i, 0.0f);
        EXPECT_LT(i, 1.0f);
    }
}
TEST_F(HostRandomTest, UniformDoubleRange) {
    ms->simulation.addStepFunction(&step_uniform_double);
    ms->run();
    for (auto &i : double_out) {
        EXPECT_GE(i, 0.0f);
        EXPECT_LT(i, 1.0f);
    }
}

TEST_F(HostRandomTest, UniformUCharRange) {
    ms->simulation.addStepFunction(&step_uniform_uchar_range);
    ms->run();
    for (auto &i : unsigned_char_out) {
        EXPECT_GE(i, static_cast<unsigned char>(UCHAR_MAX*0.25));
        EXPECT_LE(i, static_cast<unsigned char>(UCHAR_MAX*0.75));
    }
}
TEST_F(HostRandomTest, UniformCharRange) {
    ms->simulation.addStepFunction(&step_uniform_char_range);
    ms->run();
    for (auto &i : unsigned_char_out) {
        EXPECT_GE(i, static_cast<char>(CHAR_MIN*0.5));
        EXPECT_LE(i, static_cast<char>(CHAR_MAX*0.5));
    }
}

TEST_F(HostRandomTest, UniformUShortRange) {
    ms->simulation.addStepFunction(&step_uniform_ushort_range);
    ms->run();
    for (auto &i : unsigned_short_out) {
        EXPECT_GE(i, static_cast<uint16_t>(UINT16_MAX*0.25));
        EXPECT_LE(i, static_cast<uint16_t>(UINT16_MAX*0.75));
    }
}
TEST_F(HostRandomTest, UniformShortRange) {
    ms->simulation.addStepFunction(&step_uniform_short_range);
    ms->run();
    for (auto &i : short_out) {
        EXPECT_GE(i, static_cast<int16_t>(INT16_MIN*0.5));
        EXPECT_LE(i, static_cast<int16_t>(INT16_MAX*0.5));
    }
}

TEST_F(HostRandomTest, UniformUIntRange) {
    ms->simulation.addStepFunction(&step_uniform_uint_range);
    ms->run();
    for (auto &i : unsigned_int_out) {
        EXPECT_GE(i, static_cast<unsigned int>(UINT_MAX*0.25));
        EXPECT_LE(i, static_cast<unsigned int>(UINT_MAX*0.75));
    }
}
TEST_F(HostRandomTest, UniformIntRange) {
    ms->simulation.addStepFunction(&step_uniform_int_range);
    ms->run();
    for (auto &i : int_out) {
        EXPECT_GE(i, static_cast<int>(INT_MIN*0.5));
        EXPECT_LE(i, static_cast<int>(INT_MAX*0.5));
    }
}

TEST_F(HostRandomTest, UniformULongLongRange) {
    ms->simulation.addStepFunction(&step_uniform_ulonglong_range);
    ms->run();
    for (auto &i : unsigned_longlong_out) {
        EXPECT_GE(i, static_cast<uint64_t>(UINT64_MAX*0.25));
        EXPECT_LE(i, static_cast<uint64_t>(UINT64_MAX*0.75));
    }
}
TEST_F(HostRandomTest, UniformLongLongRange) {
    ms->simulation.addStepFunction(&step_uniform_longlong_range);
    ms->run();
    for (auto &i : longlong_out) {
        EXPECT_GE(i, static_cast<int64_t>(INT64_MIN >> 1));
        EXPECT_LE(i, static_cast<int64_t>(INT64_MAX >> 1));
    }
}

#endif  // TESTS_TEST_CASES_RUNTIME_TEST_HOST_RANDOM_H_
