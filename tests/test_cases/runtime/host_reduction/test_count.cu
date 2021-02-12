#include "helpers/host_reductions_common.h"

namespace test_host_reductions {
FLAMEGPU_STEP_FUNCTION(step_countfloat) {
    uint32_t_out = FLAMEGPU->agent("agent").count<float>("float", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countdouble) {
    uint32_t_out = FLAMEGPU->agent("agent").count<double>("double", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countchar) {
    uint32_t_out = FLAMEGPU->agent("agent").count<char>("char", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countuchar) {
    uint32_t_out = FLAMEGPU->agent("agent").count<unsigned char>("uchar", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countint16_t) {
    uint32_t_out = FLAMEGPU->agent("agent").count<int16_t>("int16_t", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countuint16_t) {
    uint32_t_out = FLAMEGPU->agent("agent").count<uint16_t>("uint16_t", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").count<int32_t>("int32_t", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countuint32_t) {
    uint32_t_out = FLAMEGPU->agent("agent").count<uint32_t>("uint32_t", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countint64_t) {
    uint32_t_out = FLAMEGPU->agent("agent").count<int64_t>("int64_t", 0);
}
FLAMEGPU_STEP_FUNCTION(step_countuint64_t) {
    uint32_t_out = FLAMEGPU->agent("agent").count<uint64_t>("uint64_t", 0);
}

TEST_F(HostReductionTest, CountFloat) {
    ms->model.addStepFunction(step_countfloat);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <float> dist(FLT_MIN, FLT_MAX);
    std::array<float, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<float>("float", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<float>(0)));
}
TEST_F(HostReductionTest, CountDouble) {
    ms->model.addStepFunction(step_countdouble);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <double> dist(DBL_MIN, DBL_MAX);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<double>("double", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<double>(0)));
}
TEST_F(HostReductionTest, CountChar) {
    ms->model.addStepFunction(step_countchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(CHAR_MIN, CHAR_MAX);
    std::array<char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = static_cast<char>(dist(rd));
        } else {
            in[i] = 0;
        }
        instance.setVariable<char>("char", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<char>(0)));
}
TEST_F(HostReductionTest, CountUnsignedChar) {
    ms->model.addStepFunction(step_countuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<unsigned char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = static_cast<unsigned char>(dist(rd));
        } else {
            in[i] = 0;
        }
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<unsigned char>(0)));
}
TEST_F(HostReductionTest, CountInt16) {
    ms->model.addStepFunction(step_countint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(INT16_MIN, INT16_MAX);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<int16_t>(0)));
}
TEST_F(HostReductionTest, CountUnsignedInt16) {
    ms->model.addStepFunction(step_countuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<uint16_t>(0)));
}
TEST_F(HostReductionTest, CountInt32) {
    ms->model.addStepFunction(step_countint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(INT32_MIN, INT32_MAX);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<int32_t>(0)));
}
TEST_F(HostReductionTest, CountUnsignedInt32) {
    ms->model.addStepFunction(step_countuint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, UINT32_MAX);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<uint32_t>(0)));
}
TEST_F(HostReductionTest, CountInt64) {
    ms->model.addStepFunction(step_countint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<int64_t>(0)));
}
TEST_F(HostReductionTest, CountUnsignedInt64) {
    ms->model.addStepFunction(step_countuint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT64_MAX);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        if (i < TEST_LEN / 2) {
            in[i] = dist(rd);
        } else {
            in[i] = 0;
        }
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    EXPECT_EQ(uint32_t_out, std::count(in.begin(), in.end(), static_cast<uint64_t>(0)));
}
}  // namespace test_host_reductions
