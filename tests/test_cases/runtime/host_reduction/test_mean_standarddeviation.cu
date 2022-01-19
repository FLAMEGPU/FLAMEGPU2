#include "helpers/host_reductions_common.h"
namespace flamegpu {

namespace test_host_reductions {
FLAMEGPU_STEP_FUNCTION(step_sum_float) {
    mean_sd_out = FLAMEGPU->agent("agent").meanStandardDeviation<float>("float");
}
FLAMEGPU_STEP_FUNCTION(step_sum_int32_t) {
    mean_sd_out = FLAMEGPU->agent("agent").meanStandardDeviation<int32_t>("int32_t");
}
FLAMEGPU_STEP_FUNCTION(step_sum_uint32_t) {
    mean_sd_out = FLAMEGPU->agent("agent").meanStandardDeviation<uint32_t>("uint32_t");
}

TEST_F(HostReductionTest, MeanStandardDeviation_float) {
    ms->model.addStepFunction(step_sum_float);
    ms->population->resize(101);
    uint64_t sum = 0;
    for (unsigned int i = 0; i < 101; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        instance.setVariable<float>("float", static_cast<float>(i));
        sum += i;
    }
    ms->run();
    EXPECT_DOUBLE_EQ(sum / 101.0, mean_sd_out.first);
    EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(mean_sd_out.second));  // Test value calculated with excel
}
TEST_F(HostReductionTest, MeanStandardDeviation_int32_t) {
    ms->model.addStepFunction(step_sum_int32_t);
    ms->population->resize(101);
    uint64_t sum = 0;
    for (unsigned int i = 0; i < 101; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        instance.setVariable<int>("int32_t", static_cast<int>(i + 1));
        sum += (i + 1);
    }
    ms->run();
    EXPECT_DOUBLE_EQ(sum / 101.0, mean_sd_out.first);
    EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(mean_sd_out.second));  // Test value calculated with excel
}
TEST_F(HostReductionTest, MeanStandardDeviation_uint32_t) {
    ms->model.addStepFunction(step_sum_uint32_t);
    ms->population->resize(101);
    uint64_t sum = 0;
    for (unsigned int i = 0; i < 101; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        instance.setVariable<unsigned int>("uint32_t", static_cast<uint32_t>(i + 2));
        sum += (i+2);
    }
    ms->run();
    EXPECT_DOUBLE_EQ(sum / 101.0, mean_sd_out.first);
    EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(mean_sd_out.second));  // Test value calculated with excel
}

}  // namespace test_host_reductions
}  // namespace flamegpu
