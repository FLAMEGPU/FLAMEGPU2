#include <numeric>
#ifndef TESTS_TEST_CASES_RUNTIME_TEST_HOST_REDUCTIONS_H_
#define TESTS_TEST_CASES_RUNTIME_TEST_HOST_REDUCTIONS_H_

#include <array>
#include <random>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace {
const unsigned int TEST_LEN = 256;
float float_out = 0;
double double_out = 0;
char char_out = 0;
unsigned char uchar_out = 0;
uint16_t uint16_t_out = 0;
int16_t int16_t_out = 0;
uint32_t uint32_t_out = 0;
int32_t int32_t_out = 0;
uint64_t uint64_t_out = 0;
int64_t int64_t_out = 0;
FLAMEGPU_STEP_FUNCTION(step_minfloat) {
    float_out = FLAMEGPU->agent("agent").min<float>("float");
}
FLAMEGPU_STEP_FUNCTION(step_maxfloat) {
    float_out = FLAMEGPU->agent("agent").max<float>("float");
}
FLAMEGPU_STEP_FUNCTION(step_sumfloat) {
    float_out = FLAMEGPU->agent("agent").sum<float>("float");
}
FLAMEGPU_STEP_FUNCTION(step_mindouble) {
    double_out = FLAMEGPU->agent("agent").min<double>("double");
}
FLAMEGPU_STEP_FUNCTION(step_maxdouble) {
    double_out = FLAMEGPU->agent("agent").max<double>("double");
}
FLAMEGPU_STEP_FUNCTION(step_sumdouble) {
    double_out = FLAMEGPU->agent("agent").sum<double>("double");
}
FLAMEGPU_STEP_FUNCTION(step_minuchar) {
    uchar_out = FLAMEGPU->agent("agent").min<unsigned char>("uchar");
}
FLAMEGPU_STEP_FUNCTION(step_maxuchar) {
    uchar_out = FLAMEGPU->agent("agent").max<unsigned char>("uchar");
}
FLAMEGPU_STEP_FUNCTION(step_sumuchar) {
    uchar_out = FLAMEGPU->agent("agent").sum<unsigned char>("uchar");
}
FLAMEGPU_STEP_FUNCTION(step_minchar) {
    char_out = FLAMEGPU->agent("agent").min<char>("char");
}
FLAMEGPU_STEP_FUNCTION(step_maxchar) {
    char_out = FLAMEGPU->agent("agent").max<char>("char");
}
FLAMEGPU_STEP_FUNCTION(step_sumchar) {
    char_out = FLAMEGPU->agent("agent").sum<char>("char");
}
FLAMEGPU_STEP_FUNCTION(step_minuint16_t) {
    uint16_t_out = FLAMEGPU->agent("agent").min<uint16_t>("uint16_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxuint16_t) {
    uint16_t_out = FLAMEGPU->agent("agent").max<uint16_t>("uint16_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumuint16_t) {
    uint16_t_out = FLAMEGPU->agent("agent").sum<uint16_t>("uint16_t");
}
FLAMEGPU_STEP_FUNCTION(step_minint16_t) {
    int16_t_out = FLAMEGPU->agent("agent").min<int16_t>("int16_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxint16_t) {
    int16_t_out = FLAMEGPU->agent("agent").max<int16_t>("int16_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumint16_t) {
    int16_t_out = FLAMEGPU->agent("agent").sum<int16_t>("int16_t");
}
FLAMEGPU_STEP_FUNCTION(step_minuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").min<uint32_t>("uint32_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").max<uint32_t>("uint32_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").sum<uint32_t>("uint32_t");
}
FLAMEGPU_STEP_FUNCTION(step_minint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").min<int32_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").max<int32_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").sum<int32_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(step_minuint64_t) {
    uint64_t_out = FLAMEGPU->agent("agent").min<uint64_t>("uint64_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxuint64_t) {
    uint64_t_out = FLAMEGPU->agent("agent").max<uint64_t>("uint64_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumuint64_t) {
    uint64_t_out = FLAMEGPU->agent("agent").sum<uint64_t>("uint64_t");
}
FLAMEGPU_STEP_FUNCTION(step_minint64_t) {
    int64_t_out = FLAMEGPU->agent("agent").min<int64_t>("int64_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxint64_t) {
    int64_t_out = FLAMEGPU->agent("agent").max<int64_t>("int64_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumint64_t) {
    int64_t_out = FLAMEGPU->agent("agent").sum<int64_t>("int64_t");
}

class MiniSim {
 public:
    MiniSim() :
      model("model"),
      agent("agent"),
      simulation(model),
      population(nullptr) {
        agent.addAgentVariable<float>("float");
        agent.addAgentVariable<double>("double");
        agent.addAgentVariable<char>("char");
        agent.addAgentVariable<unsigned char>("uchar");
        agent.addAgentVariable<uint16_t>("uint16_t");
        agent.addAgentVariable<int16_t>("int16_t");
        agent.addAgentVariable<uint32_t>("uint32_t");
        agent.addAgentVariable<int32_t>("int32_t");
        agent.addAgentVariable<uint64_t>("uint64_t");
        agent.addAgentVariable<int64_t>("int64_t");
        population = new AgentPopulation(agent, TEST_LEN);
        simulation.setSimulationSteps(1);
    }
    void run() {
        model.addAgent(agent);
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.addAgent(agent) first
        CUDAAgentModel cuda_model(model);
        // This fails as agentMap is empty
        cuda_model.setInitialPopulationData(*population);
        ASSERT_NO_THROW(cuda_model.simulate(simulation));
        // The negative of this, is that cuda_model is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cuda_model.getPopulationData(*population));
    }
    ModelDescription model;
    AgentDescription agent;
    Simulation simulation;
    AgentPopulation *population;
};
/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class HostReductionTest : public testing::Test {
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

/**
 * Float
 */
TEST_F(HostReductionTest, MinFloat)
{
    ms->simulation.addStepFunction(&step_minfloat);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <float> dist(FLT_MIN, FLT_MAX);
    std::array<float, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<float>("float", in[i]);
    }
    ms->run();
    EXPECT_EQ(float_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxFloat)
{
    ms->simulation.addStepFunction(&step_maxfloat);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <float> dist(FLT_MIN, FLT_MAX);
    std::array<float, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<float>("float", in[i]);
    }
    ms->run();
    EXPECT_EQ(float_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumFloat)
{
    ms->simulation.addStepFunction(&step_sumfloat);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <float> dist(FLT_MIN, FLT_MAX);
    std::array<float, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<float>("float", in[i]);
    }
    ms->run();
    EXPECT_EQ(float_out, std::accumulate(in.begin(), in.end(), 0.0f));
}

/**
 * Double
 */
TEST_F(HostReductionTest, MinDouble)
{
    ms->simulation.addStepFunction(&step_mindouble);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <double> dist(DBL_MIN, DBL_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<double>("double", in[i]);
    }
    ms->run();
    EXPECT_EQ(double_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxDouble)
{
    ms->simulation.addStepFunction(&step_maxdouble);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <double> dist(DBL_MIN, DBL_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<double>("double", in[i]);
    }
    ms->run();
    EXPECT_EQ(double_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumDouble)
{
    ms->simulation.addStepFunction(&step_sumdouble);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <double> dist(DBL_MIN, DBL_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<double>("double", in[i]);
    }
    ms->run();
    EXPECT_EQ(double_out, std::accumulate(in.begin(), in.end(), 0.0));
}


/**
 * Char
 */
TEST_F(HostReductionTest, MinChar)
{
    ms->simulation.addStepFunction(&step_minchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(CHAR_MIN, CHAR_MAX);
    std::array<char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<char>(dist(rd));
        instance.setVariable<char>("char", in[i]);
    }
    ms->run();
    EXPECT_EQ(char_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxChar)
{
    ms->simulation.addStepFunction(&step_maxchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(CHAR_MIN, CHAR_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<char>(dist(rd));
        instance.setVariable<char>("char", in[i]);
    }
    ms->run();
    EXPECT_EQ(char_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumChar)
{
    ms->simulation.addStepFunction(&step_sumchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(CHAR_MIN, CHAR_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<char>(dist(rd));
        instance.setVariable<char>("char", in[i]);
    }
    ms->run();
    EXPECT_EQ(char_out, std::accumulate(in.begin(), in.end(), 0.0));
}

/**
 * Unsigned Char
 */
TEST_F(HostReductionTest, MinUnsignedChar)
{
    ms->simulation.addStepFunction(&step_minuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uchar_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxUnsignedChar)
{
    ms->simulation.addStepFunction(&step_maxuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uchar_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumUnsignedChar)
{
    ms->simulation.addStepFunction(&step_sumuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uchar_out, std::accumulate(in.begin(), in.end(), 0.0));
}

/**
 * int16_t
 */
TEST_F(HostReductionTest, MinInt16)
{
    ms->simulation.addStepFunction(&step_minint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(INT16_MIN, INT16_MAX);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int16_t_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxInt16)
{
    ms->simulation.addStepFunction(&step_maxint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(INT16_MIN, INT16_MAX);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int16_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumInt16)
{
    ms->simulation.addStepFunction(&step_sumint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(INT16_MIN, INT16_MAX);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int16_t_out, std::accumulate(in.begin(), in.end(), 0.0));
}

/**
 * uint16_t
 */
TEST_F(HostReductionTest, MinUnsignedInt16)
{
    ms->simulation.addStepFunction(&step_minuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint16_t_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxUnsignedInt16)
{
    ms->simulation.addStepFunction(&step_maxuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint16_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumUnsignedInt16)
{
    ms->simulation.addStepFunction(&step_sumuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint16_t_out, std::accumulate(in.begin(), in.end(), 0.0));
}

/**
 * int32_t
 */
TEST_F(HostReductionTest, MinInt32)
{
    ms->simulation.addStepFunction(&step_minint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(INT32_MIN, INT32_MAX);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int32_t_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxInt32)
{
    ms->simulation.addStepFunction(&step_maxint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(INT32_MIN, INT32_MAX);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int32_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumInt32)
{
    ms->simulation.addStepFunction(&step_sumint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(INT32_MIN, INT32_MAX);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int32_t_out, std::accumulate(in.begin(), in.end(), 0.0));
}

/**
 * uint32_t
 */
TEST_F(HostReductionTest, MinUnsignedInt32)
{
    ms->simulation.addStepFunction(&step_minuint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, UINT32_MAX);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxUnsignedInt32)
{
    ms->simulation.addStepFunction(&step_maxuint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, UINT32_MAX);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumUnsignedInt32)
{
    ms->simulation.addStepFunction(&step_sumuint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, UINT32_MAX);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::accumulate(in.begin(), in.end(), 0.0));
}

/**
 * int64_t
 */
TEST_F(HostReductionTest, MinInt64)
{
    ms->simulation.addStepFunction(&step_minint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int64_t_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxInt64)
{
    ms->simulation.addStepFunction(&step_maxint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int64_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumInt64)
{
    ms->simulation.addStepFunction(&step_sumint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int64_t_out, std::accumulate(in.begin(), in.end(), 0.0));
}

/**
 * uint64_t
 */
TEST_F(HostReductionTest, MinUnsignedInt64)
{
    ms->simulation.addStepFunction(&step_minuint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT64_MAX);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint64_t_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxUnsignedInt64)
{
    ms->simulation.addStepFunction(&step_maxuint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT64_MAX);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint64_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, SumUnsignedInt64)
{
    ms->simulation.addStepFunction(&step_sumuint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT64_MAX);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint64_t_out, std::accumulate(in.begin(), in.end(), 0.0));
}

#endif  // TESTS_TEST_CASES_RUNTIME_TEST_HOST_REDUCTIONS_H_
