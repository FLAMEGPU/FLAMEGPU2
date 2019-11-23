#include "helpers/host_reductions_common.h"

namespace test_host_reductions {
FLAMEGPU_CUSTOM_REDUCTION(customMax, a, b) {
    return a > b ? a : b;
}

FLAMEGPU_STEP_FUNCTION(step_reducefloat) {
    float_out = FLAMEGPU->agent("agent").reduce<float>("float", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reducedouble) {
    double_out = FLAMEGPU->agent("agent").reduce<double>("double", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reduceuchar) {
    uchar_out = FLAMEGPU->agent("agent").reduce<unsigned char>("uchar", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reducechar) {
    char_out = FLAMEGPU->agent("agent").reduce<char>("char", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reduceuint16_t) {
    uint16_t_out = FLAMEGPU->agent("agent").reduce<uint16_t>("uint16_t", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reduceint16_t) {
    int16_t_out = FLAMEGPU->agent("agent").reduce<int16_t>("int16_t", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reduceuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").reduce<uint32_t>("uint32_t", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reduceint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").reduce<int32_t>("int32_t", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reduceuint64_t) {
    uint64_t_out = FLAMEGPU->agent("agent").reduce<uint64_t>("uint64_t", customMax, 0);
}
FLAMEGPU_STEP_FUNCTION(step_reduceint64_t) {
    int64_t_out = FLAMEGPU->agent("agent").reduce<int64_t>("int64_t", customMax, 0);
}

TEST_F(HostReductionTest, CustomReduceFloat) {
    ms->simulation.addStepFunction(&step_reducefloat);
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
TEST_F(HostReductionTest, CustomReduceDouble) {
    ms->simulation.addStepFunction(&step_reducedouble);
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
TEST_F(HostReductionTest, CustomReduceChar) {
    ms->simulation.addStepFunction(&step_reducechar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(CHAR_MIN, CHAR_MAX);
    std::array<char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        if (i < 256) {
            in[i] = static_cast<char>(dist(rd));
        } else {
            in[i] = 0;
        }
        instance.setVariable<char>("char", in[i]);
    }
    ms->run();
    EXPECT_EQ(char_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, CustomReduceUnsignedChar) {
    ms->simulation.addStepFunction(&step_reduceuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<unsigned char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uchar_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, CustomReduceInt16) {
    ms->simulation.addStepFunction(&step_reduceint16_t);
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
TEST_F(HostReductionTest, CustomReduceUnsignedInt16) {
    ms->simulation.addStepFunction(&step_reduceuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint16_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, CustomReduceInt32) {
    ms->simulation.addStepFunction(&step_reduceint32_t);
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
TEST_F(HostReductionTest, CustomReduceUnsignedInt32) {
    ms->simulation.addStepFunction(&step_reduceuint32_t);
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
TEST_F(HostReductionTest, CustomReduceInt64) {
    ms->simulation.addStepFunction(&step_reduceint64_t);
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
TEST_F(HostReductionTest, CustomReduceUnsignedInt64) {
    ms->simulation.addStepFunction(&step_reduceuint64_t);
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
}  // namespace test_host_reductions
