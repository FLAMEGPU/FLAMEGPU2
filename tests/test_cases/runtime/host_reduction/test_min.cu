#include "helpers/host_reductions_common.h"

namespace test_host_reductions {
FLAMEGPU_STEP_FUNCTION(step_minfloat) {
    float_out = FLAMEGPU->agent("agent").min<float>("float");
}
FLAMEGPU_STEP_FUNCTION(step_mindouble) {
    double_out = FLAMEGPU->agent("agent").min<double>("double");
}
FLAMEGPU_STEP_FUNCTION(step_minuchar) {
    uchar_out = FLAMEGPU->agent("agent").min<unsigned char>("uchar");
}
FLAMEGPU_STEP_FUNCTION(step_minchar) {
    char_out = FLAMEGPU->agent("agent").min<char>("char");
}
FLAMEGPU_STEP_FUNCTION(step_minuint16_t) {
    uint16_t_out = FLAMEGPU->agent("agent").min<uint16_t>("uint16_t");
}
FLAMEGPU_STEP_FUNCTION(step_minint16_t) {
    int16_t_out = FLAMEGPU->agent("agent").min<int16_t>("int16_t");
}
FLAMEGPU_STEP_FUNCTION(step_minuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").min<uint32_t>("uint32_t");
}
FLAMEGPU_STEP_FUNCTION(step_minint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").min<int32_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(step_minuint64_t) {
    uint64_t_out = FLAMEGPU->agent("agent").min<uint64_t>("uint64_t");
}
FLAMEGPU_STEP_FUNCTION(step_minint64_t) {
    int64_t_out = FLAMEGPU->agent("agent").min<int64_t>("int64_t");
}

TEST_F(HostReductionTest, MinFloat) {
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
TEST_F(HostReductionTest, MinDouble) {
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
TEST_F(HostReductionTest, MinChar) {
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
TEST_F(HostReductionTest, MinUnsignedChar) {
    ms->simulation.addStepFunction(&step_minuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<unsigned char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uchar_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MinInt16) {
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
TEST_F(HostReductionTest, MinUnsignedInt16) {
    ms->simulation.addStepFunction(&step_minuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint16_t_out, *std::min_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MinInt32) {
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
TEST_F(HostReductionTest, MinUnsignedInt32) {
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
TEST_F(HostReductionTest, MinInt64) {
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
TEST_F(HostReductionTest, MinUnsignedInt64) {
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
}  // namespace test_host_reductions
