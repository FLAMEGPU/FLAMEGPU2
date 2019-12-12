#include "helpers/host_reductions_common.h"

namespace test_host_reductions {
FLAMEGPU_STEP_FUNCTION(step_sumfloat) {
    float_out = FLAMEGPU->agent("agent").sum<float>("float");
}
FLAMEGPU_STEP_FUNCTION(step_sumdouble) {
    double_out = FLAMEGPU->agent("agent").sum<double>("double");
}
FLAMEGPU_STEP_FUNCTION(step_sumuchar) {
    uchar_out = FLAMEGPU->agent("agent").sum<unsigned char>("uchar");
    uint64_t_out = FLAMEGPU->agent("agent").sum<unsigned char, int64_t>("uchar");
}
FLAMEGPU_STEP_FUNCTION(step_sumchar) {
    char_out = FLAMEGPU->agent("agent").sum<char>("char");
    int64_t_out = FLAMEGPU->agent("agent").sum<char, int64_t>("char");
}
FLAMEGPU_STEP_FUNCTION(step_sumuint16_t) {
    uint16_t_out = FLAMEGPU->agent("agent").sum<uint16_t>("uint16_t");
    uint64_t_out = FLAMEGPU->agent("agent").sum<uint16_t, int64_t>("uint16_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumint16_t) {
    int16_t_out = FLAMEGPU->agent("agent").sum<int16_t>("int16_t");
    int64_t_out = FLAMEGPU->agent("agent").sum<int16_t, int64_t>("int16_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").sum<uint32_t>("uint32_t");
    uint64_t_out = FLAMEGPU->agent("agent").sum<uint32_t, int64_t>("uint32_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").sum<int32_t>("int32_t");
    int64_t_out = FLAMEGPU->agent("agent").sum<int32_t, int64_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumuint64_t) {
    uint64_t_out = FLAMEGPU->agent("agent").sum<uint64_t>("uint64_t");
}
FLAMEGPU_STEP_FUNCTION(step_sumint64_t) {
    int64_t_out = FLAMEGPU->agent("agent").sum<int64_t>("int64_t");
}

TEST_F(HostReductionTest, SumFloat) {
    ms->model.addStepFunction(step_sumfloat);
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
TEST_F(HostReductionTest, SumDouble) {
    ms->model.addStepFunction(step_sumdouble);
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
TEST_F(HostReductionTest, SumChar) {
    ms->model.addStepFunction(step_sumchar);
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
    EXPECT_EQ(char_out, std::accumulate(in.begin(), in.end(), static_cast<char>(0)));
    EXPECT_EQ(int64_t_out, std::accumulate(in.begin(), in.end(), static_cast<int64_t>(0)));
}
TEST_F(HostReductionTest, SumUnsignedChar) {
    ms->model.addStepFunction(step_sumuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<unsigned char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uchar_out, std::accumulate(in.begin(), in.end(), static_cast<unsigned char>(0)));
    EXPECT_EQ(uint64_t_out, std::accumulate(in.begin(), in.end(), static_cast<uint64_t>(0)));
}
TEST_F(HostReductionTest, SumInt16) {
    ms->model.addStepFunction(step_sumint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(INT16_MIN, INT16_MAX);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int16_t_out, std::accumulate(in.begin(), in.end(), static_cast<int16_t>(0)));
    EXPECT_EQ(int64_t_out, std::accumulate(in.begin(), in.end(), static_cast<int64_t>(0)));
}
TEST_F(HostReductionTest, SumUnsignedInt16) {
    ms->model.addStepFunction(step_sumuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint16_t_out, std::accumulate(in.begin(), in.end(), static_cast<uint16_t>(0)));
    EXPECT_EQ(uint64_t_out, std::accumulate(in.begin(), in.end(), static_cast<uint64_t>(0)));
}
TEST_F(HostReductionTest, SumInt32) {
    ms->model.addStepFunction(step_sumint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(INT32_MIN, INT32_MAX);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int32_t_out, std::accumulate(in.begin(), in.end(), static_cast<int32_t>(0)));
    EXPECT_EQ(int64_t_out, std::accumulate(in.begin(), in.end(), static_cast<int64_t>(0)));
}
TEST_F(HostReductionTest, SumUnsignedInt32) {
    ms->model.addStepFunction(step_sumuint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, UINT32_MAX);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::accumulate(in.begin(), in.end(), static_cast<uint32_t>(0)));
    EXPECT_EQ(uint64_t_out, std::accumulate(in.begin(), in.end(), static_cast<uint64_t>(0)));
}
TEST_F(HostReductionTest, SumInt64) {
    ms->model.addStepFunction(step_sumint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int64_t_out, std::accumulate(in.begin(), in.end(), 0ll));
}
TEST_F(HostReductionTest, SumUnsignedInt64) {
    ms->model.addStepFunction(step_sumuint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT64_MAX);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentInstance instance = ms->population->getNextInstance();
        in[i] = dist(rd);
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint64_t_out, std::accumulate(in.begin(), in.end(), 0llu));
}
}  // namespace test_host_reductions
