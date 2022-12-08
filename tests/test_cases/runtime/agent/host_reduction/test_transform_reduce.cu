#include "helpers/host_reductions_common.h"
namespace flamegpu {


namespace test_host_reductions {
FLAMEGPU_CUSTOM_REDUCTION(customSum, a, b) {
    return a + b;
}
FLAMEGPU_CUSTOM_TRANSFORM(customTransform, a) {
    return a <= 0 ? 1 : 0;
}

FLAMEGPU_STEP_FUNCTION(step_transformReduceFloat) {
    uint32_t_out = FLAMEGPU->agent("agent").transformReduce<float, uint32_t>("float", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceDouble) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<double, int32_t>("double", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReducechar) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<char, int32_t>("char", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceuchar) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<unsigned char, int32_t>("uchar", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceint16_t) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<int16_t, int32_t>("int16_t", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceuint16_t) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<uint16_t, int32_t>("uint16_t", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<int32_t, int32_t>("int32_t", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceuint32_t) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<uint32_t, int32_t>("uint32_t", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceint64_t) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<int64_t, int32_t>("int64_t", customTransform, customSum, 0);
}
FLAMEGPU_STEP_FUNCTION(step_transformReduceuint64_t) {
    int32_t_out = FLAMEGPU->agent("agent").transformReduce<uint64_t, int32_t>("uint64_t", customTransform, customSum, 0);
}

TEST_F(HostReductionTest, CustomTransformReduceFloat) {
    ms->model.addStepFunction(step_transformReduceFloat);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_real_distribution <float> dist(FLT_MIN, FLT_MAX);
    std::array<float, TEST_LEN> in;
    std::array<int, TEST_LEN> inTransform;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<float>("float", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<float, uint32_t>());
    EXPECT_EQ(uint32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<uint32_t>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceDouble) {
    ms->model.addStepFunction(step_transformReduceDouble);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_real_distribution <double> dist(DBL_MIN, DBL_MAX);
    std::array<double, TEST_LEN> in;
    std::array<int, TEST_LEN> inTransform;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<double>("double", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<double, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceChar) {
    ms->model.addStepFunction(step_transformReducechar);
    std::array<int, TEST_LEN> inTransform;
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
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<char, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceUnsignedChar) {
    ms->model.addStepFunction(step_transformReduceuchar);
    std::array<int, TEST_LEN> inTransform;
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UCHAR_MAX);
    std::array<unsigned char, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = static_cast<unsigned char>(dist(rd));
        instance.setVariable<unsigned char>("uchar", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<unsigned char, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceInt16) {
    ms->model.addStepFunction(step_transformReduceint16_t);
    std::array<int, TEST_LEN> inTransform;
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int16_t> dist(INT16_MIN, INT16_MAX);
    std::array<int16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int16_t>("int16_t", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<int16_t, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceUnsignedInt16) {
    ms->model.addStepFunction(step_transformReduceuint16_t);
    std::array<int, TEST_LEN> inTransform;
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint16_t> dist(0, UINT16_MAX);
    std::array<uint16_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint16_t>("uint16_t", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<uint16_t, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceInt32) {
    ms->model.addStepFunction(step_transformReduceint32_t);
    std::array<int, TEST_LEN> inTransform;
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int32_t> dist(INT32_MIN, INT32_MAX);
    std::array<int32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int32_t>("int32_t", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<int32_t, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceUnsignedInt32) {
    ms->model.addStepFunction(step_transformReduceuint32_t);
    std::array<int, TEST_LEN> inTransform;
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint32_t> dist(0, UINT32_MAX);
    std::array<uint32_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint32_t>("uint32_t", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<uint32_t, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceInt64) {
    ms->model.addStepFunction(step_transformReduceint64_t);
    std::array<int, TEST_LEN> inTransform;
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<int64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<int64_t>("int64_t", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<int64_t, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
TEST_F(HostReductionTest, CustomTransformReduceUnsignedInt64) {
    ms->model.addStepFunction(step_transformReduceuint64_t);
    std::array<int, TEST_LEN> inTransform;
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <uint64_t> dist(0, UINT64_MAX);
    std::array<uint64_t, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = dist(rd);
        instance.setVariable<uint64_t>("uint64_t", in[i]);
    }
    ms->run();
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_impl::unary_function<uint64_t, int>());
    EXPECT_EQ(int32_t_out, std::count(inTransform.begin(), inTransform.end(), static_cast<int>(1)));
}
#ifdef FLAMEGPU_USE_GLM
FLAMEGPU_CUSTOM_REDUCTION(customMax2_glm, a, b) {
    return glm::max(a, b);
}
FLAMEGPU_CUSTOM_TRANSFORM(customTransform_glm, a) {
    return a + glm::vec3(1, 2, 3);
}

FLAMEGPU_STEP_FUNCTION(step_transformReduce_glm) {
    vec3_t_out = FLAMEGPU->agent("agent").transformReduce<glm::vec3>("vec3", customTransform_glm, customMax2_glm, glm::vec3(0));
}
TEST_F(HostReductionTest, CustomTransformReduce_glm) {
    ms->model.addStepFunction(step_transformReduceint64_t);
    std::mt19937_64 rd;  // Seed does not matter
    std::uniform_int_distribution <int64_t> dist(INT64_MIN, INT64_MAX);
    std::array<glm::vec3, TEST_LEN> in;
    for (unsigned int i = 0; i < TEST_LEN; i++) {
        AgentVector::Agent instance = ms->population->at(i);
        in[i] = glm::vec3(dist(rd), dist(rd), dist(rd));
        instance.setVariable<glm::vec3>("vec3", in[i]);
    }
    ms->run();
    std::array<glm::vec3, TEST_LEN> inTransform;
    std::transform(in.begin(), in.end(), inTransform.begin(), customTransform_glm_impl::unary_function<glm::vec3, glm::vec3>());
    glm::vec3 test_result = std::reduce(inTransform.begin(), inTransform.end(), glm::vec3(0), customMax2_glm_impl::binary_function<glm::vec3>());
    EXPECT_EQ(vec3_t_out, test_result);
}
#else
TEST_F(HostReductionTest, DISABLED_CustomTransformReduce_glm) { }
#endif
}  // namespace test_host_reductions
}  // namespace flamegpu
