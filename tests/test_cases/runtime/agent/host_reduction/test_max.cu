#include "helpers/host_reductions_common.h"
namespace flamegpu {


namespace test_host_reductions {
FLAMEGPU_STEP_FUNCTION(step_maxfloat) {
    float_out = FLAMEGPU->agent("agent").max<float>("float");
}
FLAMEGPU_STEP_FUNCTION(step_maxdouble) {
    double_out = FLAMEGPU->agent("agent").max<double>("double");
}
FLAMEGPU_STEP_FUNCTION(step_maxuchar) {
    uchar_out = FLAMEGPU->agent("agent").max<unsigned char>("uchar");
}
FLAMEGPU_STEP_FUNCTION(step_maxchar) {
    char_out = FLAMEGPU->agent("agent").max<char>("char");
}
FLAMEGPU_STEP_FUNCTION(step_maxuint16_t) {
    uint16_t_out = FLAMEGPU->agent("agent").max<uint16_t>("uint16_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxint16_t) {
    int16_t_out = FLAMEGPU->agent("agent").max<int16_t>("int16_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").max<uint32_t>("uint32_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").max<int32_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxuint64_t) {
    uint64_t_out = FLAMEGPU->agent("agent").max<uint64_t>("uint64_t");
}
FLAMEGPU_STEP_FUNCTION(step_maxint64_t) {
    int64_t_out = FLAMEGPU->agent("agent").max<int64_t>("int64_t");
}

TEST_F(HostReductionTest, MaxFloat) {
    ms->model.addStepFunction(step_maxfloat);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_real_distribution <float> dist(FLT_MIN, FLT_MAX);
    std::array<float, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<float>("float", in[i]);
    }
    ms->run();
    EXPECT_EQ(float_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxDouble) {
    ms->model.addStepFunction(step_maxdouble);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_real_distribution <double> dist(DBL_MIN, DBL_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<double>("double", in[i]);
    }
    ms->run();
    EXPECT_EQ(double_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxChar) {
    ms->model.addStepFunction(step_maxchar);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(CHAR_MIN, CHAR_MAX);
    std::array<char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
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
TEST_F(HostReductionTest, MaxUnsignedChar) {
    ms->model.addStepFunction(step_maxuchar);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<unsigned char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uchar_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxInt16) {
    ms->model.addStepFunction(step_maxint16_t);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(INT16_MIN, INT16_MAX);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int16_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxUnsignedInt16) {
    ms->model.addStepFunction(step_maxuint16_t);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint16_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxInt32) {
    ms->model.addStepFunction(step_maxint32_t);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(INT32_MIN, INT32_MAX);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int32_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxUnsignedInt32) {
    ms->model.addStepFunction(step_maxuint32_t);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, UINT32_MAX);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxInt64) {
    ms->model.addStepFunction(step_maxint64_t);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(int64_t_out, *std::max_element(in.begin(), in.end()));
}
TEST_F(HostReductionTest, MaxUnsignedInt64) {
    ms->model.addStepFunction(step_maxuint64_t);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT64_MAX);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint64_t_out, *std::max_element(in.begin(), in.end()));
}
}  // namespace test_host_reductions
}  // namespace flamegpu
