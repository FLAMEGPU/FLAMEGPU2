#include "helpers/host_reductions_common.h"

namespace test_host_reductions {
FLAMEGPU_STEP_FUNCTION(step_histogramEvenfloat) {
    uint_vec = FLAMEGPU->agent("agent").histogramEven<float, unsigned int>("float", 10, 0.0f, 20.0f);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvendouble) {
    int_vec = FLAMEGPU->agent("agent").histogramEven<double, int>("double", 10, 0.0, 20.0);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenchar) {
    uint_vec = FLAMEGPU->agent("agent").histogramEven<char, unsigned int>("char", 10, 0, 20);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenuchar) {
    int_vec = FLAMEGPU->agent("agent").histogramEven<unsigned char, int>("uchar", 10, 0, 20);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenint16_t) {
    uint_vec = FLAMEGPU->agent("agent").histogramEven<int16_t, unsigned int>("int16_t", 10, 0, 20);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenuint16_t) {
    int_vec = FLAMEGPU->agent("agent").histogramEven<uint16_t, int>("uint16_t", 10, 0, 20);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenint32_t) {
    uint_vec = FLAMEGPU->agent("agent").histogramEven<int32_t, unsigned int>("int32_t", 10, 0, 20);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenuint32_t) {
    int_vec = FLAMEGPU->agent("agent").histogramEven<uint32_t, int>("uint32_t", 10, 0, 20);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenint64_t) {
    uint_vec = FLAMEGPU->agent("agent").histogramEven<int64_t, unsigned int>("int64_t", 10, 0, 20);
}
FLAMEGPU_STEP_FUNCTION(step_histogramEvenuint64_t) {
    int_vec = FLAMEGPU->agent("agent").histogramEven<uint64_t, int>("uint64_t", 10, 0, 20);
}

/**
 * Poor attempt to mimic cub::histogram::histogramEven()
 * Doesn't work great with odd boundaries and ranges near integer limits
 */
template<typename InT, typename OutT>
std::vector<OutT> histogramEven(const std::array<InT, TEST_LEN> &variables, const unsigned int &histogramBins, const InT &lowerBound, const InT &upperBound) {
    assert(upperBound > lowerBound);
    std::vector<OutT> rtn(histogramBins);
    for (auto &i : rtn)
        i = static_cast<OutT>(0);
    const InT diff = upperBound - lowerBound;
    const double diffP = diff / static_cast<double>(histogramBins);
    for (auto &i : variables) {
        if (i >= lowerBound && i < upperBound) {
            ++rtn[static_cast<int>(i/ diffP)];
        }
    }
    return rtn;
}
TEST_F(HostReductionTest, HistogramEvenFloat) {
    ms->model.addStepFunction(step_histogramEvenfloat);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <float> dist(0, 20);
    std::array<float, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<float>("float", in[i]);
    }
    ms->run();
    auto check = histogramEven<float, unsigned int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < uint_vec.size(); ++i) {
        EXPECT_EQ(uint_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenDouble) {
    ms->model.addStepFunction(step_histogramEvendouble);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_real_distribution <double> dist(0, 20);
    std::array<double, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<double>("double", in[i]);
    }
    ms->run();
    auto check = histogramEven<double, int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < int_vec.size(); ++i) {
        EXPECT_EQ(int_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenChar) {
    ms->model.addStepFunction(step_histogramEvenchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(0, 19);
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
    auto check = histogramEven<char, unsigned int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < uint_vec.size(); ++i) {
        EXPECT_EQ(uint_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenUnsignedChar) {
    ms->model.addStepFunction(step_histogramEvenuchar);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, 19);
    std::array<unsigned char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    auto check = histogramEven<unsigned char, int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < int_vec.size(); ++i) {
        EXPECT_EQ(int_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenInt16) {
    ms->model.addStepFunction(step_histogramEvenint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(0, 19);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    auto check = histogramEven<int16_t, unsigned int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < uint_vec.size(); ++i) {
        EXPECT_EQ(uint_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenUnsignedInt16) {
    ms->model.addStepFunction(step_histogramEvenuint16_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, 19);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    auto check = histogramEven<uint16_t, int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < int_vec.size(); ++i) {
        EXPECT_EQ(int_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenInt32) {
    ms->model.addStepFunction(step_histogramEvenint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(0, 19);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    auto check = histogramEven<int32_t, unsigned int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < uint_vec.size(); ++i) {
        EXPECT_EQ(uint_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenUnsignedInt32) {
    ms->model.addStepFunction(step_histogramEvenuint32_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, 19);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    auto check = histogramEven<uint32_t, int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < int_vec.size(); ++i) {
        EXPECT_EQ(int_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenInt64) {
    ms->model.addStepFunction(step_histogramEvenint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(0, 19);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    auto check = histogramEven<int64_t, unsigned int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < uint_vec.size(); ++i) {
        EXPECT_EQ(uint_vec[i], check[i]);
    }
}
TEST_F(HostReductionTest, HistogramEvenUnsignedInt64) {
    ms->model.addStepFunction(step_histogramEvenuint64_t);
    std::mt19937 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, 19);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    auto check = histogramEven<uint64_t, int>(in, 10, 0, 20);
    for (unsigned int i = 0; i < int_vec.size(); ++i) {
        EXPECT_EQ(int_vec[i], check[i]);
    }
}
}  // namespace test_host_reductions
