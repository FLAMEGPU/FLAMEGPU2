#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace tests {
namespace test_runplanvector {

TEST(TestRunPlanVector, constructor) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Declare a pointer
    flamegpu::RunPlanVector * plans = nullptr;
    // Use New
    const uint32_t initialLength = 4;
    EXPECT_NO_THROW(plans = new flamegpu::RunPlanVector(model, initialLength));
    EXPECT_NE(plans, nullptr);
    EXPECT_EQ(plans->size(), initialLength);
    // Run the destructor
    EXPECT_NO_THROW(delete plans);
    plans = nullptr;
}
// Test setting the random property seed
TEST(TestRunPlanVector, setRandomPropertySeed) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Create a vector of plans
    constexpr uint32_t totalPlans = 2;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Get the current random property seed. No sensible options to check this is an expected value.
    EXPECT_NO_THROW(plans.getRandomPropertySeed());
    // Set the seed to a new value, and check that it was applied.
    uint64_t newPropertySeed = 12;
    plans.setRandomPropertySeed(newPropertySeed);
    // Check the new seed was applied correctly.
    EXPECT_EQ(plans.getRandomPropertySeed(), newPropertySeed);
    // @todo - should check that the seed actually impacts the generatred random numbers. Seed, generated. Reseed the same, compare. reseed different, compare (Low risk of the same sequence being generated, so just make the sequence not trivially small)
}
// Exit function condition which leads to 0 steps being allowed for plans within a vector.
FLAMEGPU_EXIT_CONDITION(exitcond) {
    return flamegpu::EXIT;
}
TEST(TestRunPlanVector, setSteps) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Get the original value of steps, storing them for later.
    std::array<uint32_t, totalPlans> originalValues = {{}};
    for (uint32_t idx = 0; idx < totalPlans; idx++) {
        const auto &plan = plans[idx];
        originalValues[idx] = plan.getSteps();
    }
    // Set the number of steps
    const uint32_t newSteps = 12u;
    EXPECT_NO_THROW(plans.setSteps(newSteps));
    // For Check each plan against the previous value(s)
    for (uint32_t idx = 0; idx < totalPlans; idx++) {
        const auto &plan = plans[idx];
        EXPECT_EQ(plan.getSteps(), newSteps);
        EXPECT_NE(plan.getSteps(), originalValues[idx]);
    }
    // Expect an exception if setting the value to 0?
    EXPECT_THROW(plans.setSteps(0), exception::OutOfBoundsException);

    // If the model has an exit condition, then it will not throw.
    flamegpu::ModelDescription modelWithExit("modelWithExit");
    modelWithExit.addExitCondition(exitcond);
    flamegpu::RunPlanVector plansWithExit(modelWithExit, 1u);
    // Do not expect an exception iff allow_o_steps is set.
    EXPECT_NO_THROW(plansWithExit.setSteps(0));
}
TEST(TestRunPlanVector, setOutputSubdirectory) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Set the new value
    std::string newSubdir("test");
    EXPECT_NO_THROW(plans.setOutputSubdirectory(newSubdir));
    // For Check each plan against the previous value(s)
    for (uint32_t idx = 0; idx < totalPlans; idx++) {
        const auto &plan = plans[idx];
        EXPECT_EQ(plan.getOutputSubdirectory(), newSubdir);
        EXPECT_NE(plan.getOutputSubdirectory(), "");
    }
}

TEST(TestRunPlanVector, setProperty) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Add a few environment properties to the model.
    auto &environment = model.Environment();
    const float fOriginal = 1.0f;
    const int32_t iOriginal = 1;
    const std::array<uint32_t, 3> u3Original = {{0, 1, 2}};
    const std::array<double, 3> d3Original = {{0., 1., 2.}};
    environment.newProperty<float>("f", fOriginal);
    environment.newProperty<int32_t>("i", iOriginal);
    environment.newProperty<uint32_t, 3>("u3", u3Original);
    environment.newProperty<double, 3>("d3", d3Original);
#ifdef USE_GLM
    environment.newProperty<glm::ivec3>("ivec3", {});
    environment.newProperty<glm::ivec3, 2>("ivec32", {});
    environment.newProperty<glm::ivec3, 3>("ivec33", {});
#endif
    // Create a vector of plans
    constexpr uint32_t totalPlans = 2u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Uniformly set each property to a new value, then check it has been applied correctly.
    const float fNew = 2.0f;
    const int32_t iNew = 2;
    const std::array<uint32_t, 3> u3New = {{3u, 4u, 5u}};
    const std::array<double, 3> d3New = {{3., 4., 5.}};
    // void RunPlanVector::setProperty(const std::string &name, const T &value) {
    plans.setProperty("f", fNew);
    plans.setProperty("i", iNew);
    // Check setting full arrays
    // void RunPlanVector::setProperty(const std::string &name, const std::array<T, N> &value) {
    // Explicit type is required, to coerce the std::array. Might need partial template specialisation for  where the value is a stdarray of T?
    plans.setProperty<uint32_t, 3>("u3", u3New);
    // Check setting individual array elements
    // void RunPlanVector::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value) {
    plans.setProperty<double>("d3", 0, d3New[0]);
    plans.setProperty<double>("d3", 1, d3New[1]);
    plans.setProperty<double>("d3", 2, d3New[2]);
#ifdef USE_GLM
    const glm::ivec3 ivec3_1_check = glm::ivec3{ 1, 2, 3 };
    const std::array<glm::ivec3, 2> ivec3_2_check = { glm::ivec3{4, 5, 6}, glm::ivec3{7, 8, 9} };
    const std::array<glm::ivec3, 3> ivec3_3_check =
    { glm::ivec3{ 11, 12, 13 }, glm::ivec3{14, 15, 16}, glm::ivec3{17, 18, 19} };
    plans.setProperty<glm::ivec3>("ivec3", ivec3_1_check);
    plans.setProperty<glm::ivec3, 2>("ivec32", ivec3_2_check);
    plans.setProperty<glm::ivec3, 3>("ivec33", ivec3_3_check);
#endif
    // Check values are as expected by accessing the properties from each plan
    for (const auto &plan : plans) {
        EXPECT_EQ(plan.getProperty<float>("f"), fNew);
        EXPECT_EQ(plan.getProperty<int32_t>("i"), iNew);
        // Extra brackets allow template commas in macros.
        EXPECT_EQ((plan.getProperty<uint32_t, 3>("u3")), u3New);
        EXPECT_EQ((plan.getProperty<double, 3>("d3")), d3New);
#ifdef USE_GLM
        EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec3"), ivec3_1_check);
        EXPECT_EQ((plan.getProperty<glm::ivec3, 3>)("ivec33"), ivec3_3_check);
        EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec32", 0), ivec3_2_check[0]);
        EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec32", 1), ivec3_2_check[1]);
#endif
    }

    // Tests for exceptions
    // --------------------
    // Note litereals used must match the templated type not the incorrect types used, to appease MSVC warnings.
    // void RunPlanVector::setProperty(const std::string &name, const T &value)
    EXPECT_THROW(plans.setProperty<float>("does_not_exist", 1.f), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW(plans.setProperty<float>("i", 1.f), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW(plans.setProperty<uint32_t>("u3", 1u), flamegpu::exception::InvalidEnvPropertyType);
    // void RunPlanVector::setProperty(const std::string &name, const std::array<T, N> &value)
    // Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
    EXPECT_THROW((plans.setProperty<float, 3>("does_not_exist", {2.f, 2.f, 2.f})), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plans.setProperty<float, 3>("u3", {2.f, 2.f, 2.f})), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plans.setProperty<double, 2>("d3", {-2, 0})), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plans.setProperty<double, 4>("d3", {-2, 0, 2, 2})), flamegpu::exception::InvalidEnvPropertyType);
    // void RunPlanVector::setProperty(const std::string &name, const EnvironmentManager::size_type &index, const T &value)
    EXPECT_THROW((plans.setProperty<float>("does_not_exist", 0u, 3.f)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plans.setProperty<float>("u3", 0u, 3.f)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plans.setProperty<double>("d3", static_cast<EnvironmentManager::size_type>(-1), 3)), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setProperty<double>("d3", 4u, 3)), exception::OutOfBoundsException);
#ifdef USE_GLM
    EXPECT_THROW((plans.setProperty<glm::ivec3>)("ivec32", 3u, {}), exception::OutOfBoundsException);  // Out of bounds
    EXPECT_THROW((plans.setProperty<glm::ivec3>)("ivec33", 4u, {}), exception::OutOfBoundsException);  // Out of bounds
#endif
}
template<typename T>
double t_lerp(const T &_min, const T &_max, const double &a) {
    double min = static_cast<double>(_min);
    double max = static_cast<double>(_max);
    return min * (1.0 - a) + max * a;
}

// Check that all values set lie within the min and max inclusive
// @todo - should fp be [min, max) like when using RNG?
TEST(TestRunPlanVector, setPropertyUniformDistribution) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Add a few environment properties to the model.
    auto &environment = model.Environment();
    const float fOriginal = 0.0f;
    const int32_t iOriginal = 0;
    const std::array<uint32_t, 3> u3Original = {{0, 0, 0}};
    const std::array<float, 2> f2Original = { {12.0f, 13.0f} };
    environment.newProperty<float>("f", fOriginal);
    environment.newProperty<float>("fb", fOriginal);
    environment.newProperty<int32_t>("i", iOriginal);
    environment.newProperty<uint32_t, 3>("u3", u3Original);
    environment.newProperty<float, 2>("f2", f2Original);
    // Create a vector of plans
    constexpr uint32_t totalPlans = 10u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // No need to seed the random, as this is a LERP rather than a random distribution.

    // Uniformly set each property to a new value, then check it has been applied correctly.
    const float fMin = 1.f;
    const float fMax = 100.f;
    const float fbMin = 0.0f;  // Previous bug, where floating point types were being rounded to nearest int
    const float fbMax = 1.0f;
    const int32_t iMin = 1;
    const int32_t iMax = 100;
    const std::array<uint32_t, 3> u3Min = {{1u, 101u, 201u}};
    const std::array<uint32_t, 3> u3Max = {{100u, 200u, 300u}};
    const std::array<float, 2> f2Min = { {1.0f, 100.f} };
    const std::array<float, 2> f2Max = { {0.0f, -100.0f} };
    // void setPropertyUniformDistribution(const std::string &name, const T &min, const T &max);
    plans.setPropertyUniformDistribution("f", fMin, fMax);
    plans.setPropertyUniformDistribution("fb", fbMin, fbMax);
    plans.setPropertyUniformDistribution("i", iMin, iMax);
    // Check setting individual array elements
    // void setPropertyUniformDistribution(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &max);
    plans.setPropertyUniformDistribution("u3", 0, u3Min[0], u3Max[0]);
    plans.setPropertyUniformDistribution("u3", 1, u3Min[1], u3Max[1]);
    plans.setPropertyUniformDistribution("u3", 2, u3Min[2], u3Max[2]);
    plans.setPropertyUniformDistribution("f2", 0, f2Min[0], f2Max[0]);
    plans.setPropertyUniformDistribution("f2", 1, f2Min[1], f2Max[1]);
    // Check values are as expected by accessing the properties from each plan
    int i = 0;
    const double divisor = totalPlans - 1;
    for (const auto &plan : plans) {
        const double a = i++ / divisor;
        EXPECT_EQ(plan.getProperty<float>("f"), static_cast<float>(t_lerp(fMin, fMax, a)));
        EXPECT_EQ(plan.getProperty<float>("fb"), static_cast<float>(t_lerp(fbMin, fbMax, a)));
        const std::array<float, 2> f2FromPlan = plan.getProperty<float, 2>("f2");
        EXPECT_EQ(f2FromPlan[0], static_cast<float>(t_lerp(f2Min[0], f2Max[0], a)));
        EXPECT_EQ(f2FromPlan[1], static_cast<float>(t_lerp(f2Min[1], f2Max[1], a)));
        // Note integer values are rounded
        EXPECT_EQ(plan.getProperty<int32_t>("i"), static_cast<int32_t>(round(t_lerp(iMin, iMax, a))));
        const std::array<uint32_t, 3> u3FromPlan = plan.getProperty<uint32_t, 3>("u3");
        EXPECT_EQ(u3FromPlan[0], static_cast<uint32_t>(round(t_lerp(u3Min[0], u3Max[0], a))));
        EXPECT_EQ(u3FromPlan[1], static_cast<uint32_t>(round(t_lerp(u3Min[1], u3Max[1], a))));
        EXPECT_EQ(u3FromPlan[2], static_cast<uint32_t>(round(t_lerp(u3Min[2], u3Max[2], a))));
    }

    // Tests for exceptions
    // --------------------
    flamegpu::RunPlanVector singlePlanVector(model, 1);
    // Note literals used must match the templated type not the incorrect types used, to appease MSVC warnings.
    // void RunPlanVector::setPropertyUniformDistribution(const std::string &name, const T &min, const T &max)
    EXPECT_THROW((singlePlanVector.setPropertyUniformDistribution<float>("f", 1.f, 100.f)), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyUniformDistribution<float>("does_not_exist", 1.f, 100.f)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plans.setPropertyUniformDistribution<float>("i", 1.f, 100.f)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plans.setPropertyUniformDistribution<uint32_t>("u3", 1u, 100u)), flamegpu::exception::InvalidEnvPropertyType);
    // void RunPlanVector::setPropertyUniformDistribution(const std::string &name, const EnvironmentManager::size_type
    // Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
    EXPECT_THROW((singlePlanVector.setPropertyUniformDistribution<uint32_t>("u3", 0u, 1u, 100u)), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyUniformDistribution<float>("does_not_exist", 0u, 1.f, 100.f)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plans.setPropertyUniformDistribution<float>("u3", 0u, 1.f, 100.f)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plans.setPropertyUniformDistribution<uint32_t>("u3", static_cast<EnvironmentManager::size_type>(-1), 1u, 100u)), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyUniformDistribution<uint32_t>("u3", 4u, 1u, 100u)), exception::OutOfBoundsException);
}
// Checking for uniformity of distribution would require a very large samples size.
// As std:: is used, we trust the distribution is legit, and instead just check for min/max.
TEST(TestRunPlanVector, setPropertyUniformRandom) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Add a few environment properties to the model.
    auto &environment = model.Environment();
    const float fOriginal = 1.0f;
    const int32_t iOriginal = 1;
    const std::array<uint32_t, 3> u3Original = {{0, 1, 2}};
    environment.newProperty<float>("f", fOriginal);
    environment.newProperty<int32_t>("i", iOriginal);
    environment.newProperty<uint32_t, 3>("u3", u3Original);
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Seed the RunPlanVector RNG for a deterministic test.
    plans.setRandomPropertySeed(1u);

    // Uniformly set each property to a new value, then check it has been applied correctly.
    const float fMin = 1.f;
    const float fMax = 100.f;
    const int32_t iMin = 1;
    const int32_t iMax = 100;
    const std::array<uint32_t, 3> u3Min = {{1u, 101u, 201u}};
    const std::array<uint32_t, 3> u3Max = {{100u, 200u, 300u}};
    // void setPropertyUniformRandom(const std::string &name, const T &min, const T &Max);
    plans.setPropertyUniformRandom("f", fMin, fMax);
    plans.setPropertyUniformRandom("i", iMin, iMax);
    // Check setting individual array elements
    // void setPropertyUniformRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &min, const T &Max);
    plans.setPropertyUniformRandom("u3", 0, u3Min[0], u3Max[0]);
    plans.setPropertyUniformRandom("u3", 1, u3Min[1], u3Max[1]);
    plans.setPropertyUniformRandom("u3", 2, u3Min[2], u3Max[2]);
    EXPECT_THROW((plans.setPropertyUniformRandom("u3", 3, u3Min[0], u3Max[0])), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyUniformRandom("u3", static_cast<EnvironmentManager::size_type>(-1), u3Min[0], u3Max[0])), exception::OutOfBoundsException);
    // Check values are as expected by accessing the properties from each plan
    for (const auto &plan : plans) {
        // Floating point types are inclusive-exclusive [min, Max)
        EXPECT_GE(plan.getProperty<float>("f"), fMin);
        EXPECT_LT(plan.getProperty<float>("f"), fMax);
        // Integer types are mutually inclusive [min, Max]
        EXPECT_GE(plan.getProperty<int32_t>("i"), iMin);
        EXPECT_LE(plan.getProperty<int32_t>("i"), iMax);
        // Check array values are correct, Integers so mutually inclusive
        const std::array<uint32_t, 3> u3FromPlan = plan.getProperty<uint32_t, 3>("u3");
        EXPECT_GE(u3FromPlan[0], u3Min[0]);
        EXPECT_LE(u3FromPlan[0], u3Max[0]);
        EXPECT_GE(u3FromPlan[1], u3Min[1]);
        EXPECT_LE(u3FromPlan[1], u3Max[1]);
        EXPECT_GE(u3FromPlan[2], u3Min[2]);
        EXPECT_LE(u3FromPlan[2], u3Max[2]);
    }
}
// It's non trivial to check for correct distirbutions, and we rely on std:: so we are going to trust it works as intended.
// Instead, just check that the value is different than the original. As this is not guaranteed due to (seeded) RNG, just check that atleast one value is different.
// Real property types only. Non-reals are a static assert.
TEST(TestRunPlanVector, setPropertyNormalRandom) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Add a few environment properties to the model.
    auto &environment = model.Environment();
    const float fOriginal = 1.0f;
    const std::array<double, 3> d3Original = {{0., 1., 2.}};
    environment.newProperty<float>("f", fOriginal);
    environment.newProperty<double, 3>("d3", d3Original);
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Seed the RunPlanVector RNG for a deterministic test.
    plans.setRandomPropertySeed(1u);

    // Uniformly set each property to a new value, then check that atleast one of them is not the default.
    const float fMean = 1.f;
    const float fStddev = 100.f;
    const std::array<double, 3> d3Mean = {{1., 101., 201.}};
    const std::array<double, 3> d3Stddev = {{100., 200., 300.}};
    // void setPropertyNormalRandom(const std::string &name, const T &mean, const T &stddev);
    plans.setPropertyNormalRandom("f", fMean, fStddev);
    // Check setting individual array elements
    // void setPropertyNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev);
    plans.setPropertyNormalRandom("d3", 0, d3Mean[0], d3Stddev[0]);
    plans.setPropertyNormalRandom("d3", 1, d3Mean[1], d3Stddev[1]);
    plans.setPropertyNormalRandom("d3", 2, d3Mean[2], d3Stddev[2]);
    EXPECT_THROW((plans.setPropertyNormalRandom("d3", 3, d3Mean[0], d3Stddev[0])), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyNormalRandom("d3", static_cast<EnvironmentManager::size_type>(-1), d3Mean[0], d3Stddev[0])), exception::OutOfBoundsException);
    bool fAtleastOneNonDefault = false;
    std::array<bool, 3> d3AtleastOneNonDefault = {{false, false, false}};
    // Check values are as expected by accessing the properties from each plan
    for (const auto &plan : plans) {
        if (plan.getProperty<float>("f") != fOriginal) {
            fAtleastOneNonDefault = true;
        }
        const std::array<double, 3> d3FromPlan = plan.getProperty<double, 3>("d3");
        if (d3FromPlan[0] != d3Original[0]) {
            d3AtleastOneNonDefault[0] = true;
        }
        if (d3FromPlan[1] != d3Original[1]) {
            d3AtleastOneNonDefault[1] = true;
        }
        if (d3FromPlan[2] != d3Original[2]) {
            d3AtleastOneNonDefault[2] = true;
        }
    }
    // assert that atleast one of each value is non-default.
    EXPECT_TRUE(fAtleastOneNonDefault);
    EXPECT_TRUE(d3AtleastOneNonDefault[0]);
    EXPECT_TRUE(d3AtleastOneNonDefault[1]);
    EXPECT_TRUE(d3AtleastOneNonDefault[2]);
}
// It's non trivial to check for correct distirbutions, and we rely on std:: so we are going to trust it works as intended.
// Instead, just check that the value is different than the original. As this is not guaranteed due to (seeded) RNG, just check that atleast one value is different.
// Real property types only. Non-reals are a static assert.
TEST(TestRunPlanVector, setPropertyLogNormalRandom) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Add a few environment properties to the model.
    auto &environment = model.Environment();
    const float fOriginal = 1.0f;
    const std::array<double, 3> d3Original = {{0., 1., 2.}};
    environment.newProperty<float>("f", fOriginal);
    environment.newProperty<double, 3>("d3", d3Original);
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Seed the RunPlanVector RNG for a deterministic test.
    plans.setRandomPropertySeed(1u);

    // Uniformly set each property to a new value, then check that atleast one of them is not the default.
    const float fMean = 1.f;
    const float fStddev = 100.f;
    const std::array<double, 3> d3Mean = {{1., 101., 201.}};
    const std::array<double, 3> d3Stddev = {{100., 200., 300.}};
    // void RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const T &mean, const T &stddev) {
    plans.setPropertyLogNormalRandom("f", fMean, fStddev);
    // Check setting individual array elements
    // void RunPlanVector::setPropertyLogNormalRandom(const std::string &name, const EnvironmentManager::size_type &index, const T &mean, const T &stddev) {
    plans.setPropertyLogNormalRandom("d3", 0, d3Mean[0], d3Stddev[0]);
    plans.setPropertyLogNormalRandom("d3", 1, d3Mean[1], d3Stddev[1]);
    plans.setPropertyLogNormalRandom("d3", 2, d3Mean[2], d3Stddev[2]);
    bool fAtleastOneNonDefault = false;
    std::array<bool, 3> d3AtleastOneNonDefault = {{false, false, false}};
    // Check values are as expected by accessing the properties from each plan
    for (const auto &plan : plans) {
        if (plan.getProperty<float>("f") != fOriginal) {
            fAtleastOneNonDefault = true;
        }
        const std::array<double, 3> d3FromPlan = plan.getProperty<double, 3>("d3");
        if (d3FromPlan[0] != d3Original[0]) {
            d3AtleastOneNonDefault[0] = true;
        }
        if (d3FromPlan[1] != d3Original[1]) {
            d3AtleastOneNonDefault[1] = true;
        }
        if (d3FromPlan[2] != d3Original[2]) {
            d3AtleastOneNonDefault[2] = true;
        }
    }
    // assert that atleast one of each value is non-default.
    EXPECT_TRUE(fAtleastOneNonDefault);
    EXPECT_TRUE(d3AtleastOneNonDefault[0]);
    EXPECT_TRUE(d3AtleastOneNonDefault[1]);
    EXPECT_TRUE(d3AtleastOneNonDefault[2]);
}
// It's non trivial to check for correct distirbutions, and we rely on std:: so we are going to trust it works as intended.
// Instead, just check that the value is different than the original. As this is not guaranteed due to (seeded) RNG, just check that atleast one value is different.
// Real property types only. Non-reals are a static assert.
TEST(TestRunPlanVector, setPropertyRandom) {
    // @todo - test setPropertyRandom
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Add a few environment properties to the model.
    auto &environment = model.Environment();
    const float fOriginal = 1.0f;
    const std::array<double, 3> d3Original = {{0., 1., 2.}};
    environment.newProperty<float>("f", fOriginal);
    environment.newProperty<double, 3>("d3", d3Original);
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Seed the RunPlanVector RNG for a deterministic test.
    plans.setRandomPropertySeed(1u);

    // Uniformly set each property to a new value, then check that atleast one of them is not the default.
    const float fMin = 1.f;
    const float fMax = 100.f;
    const std::array<double, 3> d3Mean = {{1., 101., 201.}};
    const std::array<double, 3> d3Stddev = {{100., 200., 300.}};
    // void setPropertyRandom(const std::string &name, rand_dist &distribution);
    std::uniform_real_distribution<float> fdist(fMin, fMax);
    plans.setPropertyRandom<float>("f", fdist);
    // Check setting individual array elements
    // void setPropertyRandom(const std::string &name, const EnvironmentManager::size_type &index, rand_dist &distribution);
    std::normal_distribution<double> d3dist0(d3Mean[0], d3Stddev[0]);
    std::normal_distribution<double> d3dist1(d3Mean[1], d3Stddev[1]);
    std::normal_distribution<double> d3dist2(d3Mean[2], d3Stddev[2]);
    plans.setPropertyRandom<double>("d3", 0, d3dist0);
    plans.setPropertyRandom<double>("d3", 1, d3dist1);
    plans.setPropertyRandom<double>("d3", 2, d3dist2);
    std::array<bool, 3> d3AtleastOneNonDefault = {{false, false, false}};
    // Check values are as expected by accessing the properties from each plan
    for (const auto &plan : plans) {
        // Floating point types are inclusive-exclusive [min, Max)
        EXPECT_GE(plan.getProperty<float>("f"), fMin);
        EXPECT_LT(plan.getProperty<float>("f"), fMax);

        const std::array<double, 3> d3FromPlan = plan.getProperty<double, 3>("d3");
        if (d3FromPlan[0] != d3Original[0]) {
            d3AtleastOneNonDefault[0] = true;
        }
        if (d3FromPlan[1] != d3Original[1]) {
            d3AtleastOneNonDefault[1] = true;
        }
        if (d3FromPlan[2] != d3Original[2]) {
            d3AtleastOneNonDefault[2] = true;
        }
    }
    // assert that atleast one of each value is non-default.
    EXPECT_TRUE(d3AtleastOneNonDefault[0]);
    EXPECT_TRUE(d3AtleastOneNonDefault[1]);
    EXPECT_TRUE(d3AtleastOneNonDefault[2]);

    // Tests for exceptions
    // --------------------
    flamegpu::RunPlanVector singlePlanVector(model, 1);
    // Note litereals used must match the templated type not the incorrect types used, to appease MSVC warnings.
    // void RunPlanVector::setPropertyRandom(const std::string &name, rand_dist &distribution)
    EXPECT_THROW((singlePlanVector.setPropertyRandom<float>("f", fdist)), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyRandom<float>("does_not_exist", fdist)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plans.setPropertyRandom<double>("f", d3dist0)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plans.setPropertyRandom<double>("d3", d3dist0)), flamegpu::exception::InvalidEnvPropertyType);
    // void RunPlanVector::setPropertyRandom(const std::string &name, const EnvironmentManager::size_type &index, rand_dist &distribution)
    // Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
    EXPECT_THROW((singlePlanVector.setPropertyRandom<double>("d3", 0u, d3dist0)), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyRandom<float>("does_not_exist", 0u, fdist)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plans.setPropertyRandom<float>("d3", 0u, fdist)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plans.setPropertyRandom<double>("d3", static_cast<EnvironmentManager::size_type>(-1), d3dist0)), exception::OutOfBoundsException);
    EXPECT_THROW((plans.setPropertyRandom<double>("d3", 4u, d3dist0)), exception::OutOfBoundsException);
}
// Test getting the random property seed
TEST(TestRunPlanVector, getRandomPropertySeed) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // repeatedly create run vectors, and get the property seed. Once we've found 2 that are  different, stop.
    // If a maximum number of tries is reached, then we error.
    const uint32_t maxGenerations = 8;
    uint64_t prevSeed = 0;
    uint64_t seed = 0;
    for (uint32_t i = 0; i < maxGenerations; i++) {
        // Create the vector
        flamegpu::RunPlanVector plans(model, 1);
        seed = plans.getRandomPropertySeed();
        if (i > 0) {
            // the seed shouldn't be the same as the previous seed, but it might be so do not expect_.
            if (prevSeed != seed) {
                // Break out the loop
                break;
            }
        }
        prevSeed = seed;
    }
    EXPECT_NE(prevSeed, seed);
}
TEST(TestRunPlanVector, size) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create run plan vectors of a number of sizes and check the value
    flamegpu::RunPlanVector plans0(model, 0u);
    EXPECT_EQ(plans0.size(), 0);
    flamegpu::RunPlanVector plans1(model, 1u);
    EXPECT_EQ(plans1.size(), 1);
    flamegpu::RunPlanVector plans4(model, 4u);
    EXPECT_EQ(plans4.size(), 4);
    flamegpu::RunPlanVector plans64(model, 64u);
    EXPECT_EQ(plans64.size(), 64);
}
TEST(TestRunPlanVector, operatorAddition) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create multiple uniqe plans which can be used to check order of plans.
    flamegpu::RunPlan plan1(model);
    const uint64_t seed1 = 1u;
    plan1.setRandomSimulationSeed(seed1);
    flamegpu::RunPlan plan2(model);
    const uint64_t seed2 = 2u;
    plan2.setRandomSimulationSeed(seed2);
    flamegpu::RunPlan plan3(model);
    const uint64_t seed3 = 3u;
    plan3.setRandomSimulationSeed(seed3);
    flamegpu::RunPlan plan4(model);
    const uint64_t seed4 = 4u;
    plan4.setRandomSimulationSeed(seed4);
    // RunPlanVector operator+(const RunPlan& rhs) const;
    {
        flamegpu::RunPlanVector vec12 = plan1 + plan2;
        flamegpu::RunPlanVector vec123 = vec12 + plan3;
        EXPECT_EQ(vec123.size(), 3);
        EXPECT_EQ(vec123[0].getRandomSimulationSeed(), seed1);
        EXPECT_EQ(vec123[1].getRandomSimulationSeed(), seed2);
        EXPECT_EQ(vec123[2].getRandomSimulationSeed(), seed3);
        /* Disabled, as operator+ is always push_back for performance reasons.
        flamegpu::RunPlanVector vec312 = plan3 + vec12;
        EXPECT_EQ(vec312.size(), 3);
        EXPECT_EQ(vec312[0].getRandomSimulationSeed(), seed3);
        EXPECT_EQ(vec312[1].getRandomSimulationSeed(), seed1);
        EXPECT_EQ(vec312[2].getRandomSimulationSeed(), seed2);
        */
    }
    // RunPlanVector operator+(const RunPlanVector& rhs) const;
    {
        flamegpu::RunPlanVector vec12 = plan1 + plan2;
        flamegpu::RunPlanVector vec34 = plan3 + plan4;
        flamegpu::RunPlanVector vec1234 = vec12 + vec34;
        EXPECT_EQ(vec1234.size(), 4);
        EXPECT_EQ(vec1234[0].getRandomSimulationSeed(), seed1);
        EXPECT_EQ(vec1234[1].getRandomSimulationSeed(), seed2);
        EXPECT_EQ(vec1234[2].getRandomSimulationSeed(), seed3);
        EXPECT_EQ(vec1234[3].getRandomSimulationSeed(), seed4);
        /* Disabled, as operator+ is always push_back for performance reasons.
        flamegpu::RunPlanVector vec3412 = vec34 + vec12;
        EXPECT_EQ(vec3412.size(), 4);
        EXPECT_EQ(vec3412[0].getRandomSimulationSeed(), seed3);
        EXPECT_EQ(vec3412[1].getRandomSimulationSeed(), seed4);
        EXPECT_EQ(vec3412[2].getRandomSimulationSeed(), seed1);
        EXPECT_EQ(vec3412[3].getRandomSimulationSeed(), seed2);
        */
    }
    // RunPlanVector& operator+=(const RunPlan& rhs);
    {
        flamegpu::RunPlanVector vec123 = plan1 + plan2;
        vec123 += plan3;
        EXPECT_EQ(vec123.size(), 3);
        EXPECT_EQ(vec123[0].getRandomSimulationSeed(), seed1);
        EXPECT_EQ(vec123[1].getRandomSimulationSeed(), seed2);
        EXPECT_EQ(vec123[2].getRandomSimulationSeed(), seed3);
        // += cannot have a plan on the lhs and a plan vector on the right.
    }
    // RunPlanVector& operator+=(const RunPlanVector& rhs);
    {
        flamegpu::RunPlanVector vec1234 = plan1 + plan2;
        vec1234 += (plan3 + plan4);
        EXPECT_EQ(vec1234.size(), 4);
        EXPECT_EQ(vec1234[0].getRandomSimulationSeed(), seed1);
        EXPECT_EQ(vec1234[1].getRandomSimulationSeed(), seed2);
        EXPECT_EQ(vec1234[2].getRandomSimulationSeed(), seed3);
        EXPECT_EQ(vec1234[3].getRandomSimulationSeed(), seed4);
    }

    // Expected exceptions
    // -------------------
    // Adding runplans together which are not for the same model (actually environment) should throw.
    flamegpu::RunPlanVector planVector(model, 1);
    flamegpu::ModelDescription otherModel("other");
    otherModel.Environment().newProperty<float>("f", 1.0f);  // If both models have null environments they are compatible
    flamegpu::RunPlan otherPlan(otherModel);
    flamegpu::RunPlanVector otherPlanVector(otherModel, 1);
    EXPECT_THROW((plan1 + otherPlan), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((otherPlan + plan1), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((plan1 + otherPlanVector), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((otherPlanVector + plan1), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((planVector + otherPlan), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((otherPlan + planVector), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((planVector += otherPlan), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((otherPlanVector += plan1), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((planVector += otherPlanVector), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((otherPlanVector += planVector), flamegpu::exception::InvalidArgument);
}
// RunPlanVector operator*(const unsigned int& rhs) const;
TEST(TestRunPlanVector, operatorMultiplication) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    EXPECT_EQ(plans.size(), totalPlans);

    // Multiply the plan vector by a fixed size
    // RunPlanVector operator*(const unsigned int& rhs) const;
    const uint32_t mult = 2u;
    flamegpu::RunPlanVector morePlans = plans * mult;
    const uint32_t expectedSize = mult * totalPlans;
    EXPECT_EQ(morePlans.size(), expectedSize);

    // multiply a plan in-place
    // RunPlanVector& operator*=(const unsigned int& rhs);
    plans *= mult;
    EXPECT_EQ(plans.size(), expectedSize);
}
// operator[]
TEST(TestRunPlanVector, operatorSubscript) {
    // Define the simple model to use
    flamegpu::ModelDescription model("test");
    // Create a vector of plans
    constexpr uint32_t totalPlans = 4u;
    flamegpu::RunPlanVector plans(model, totalPlans);
    // Check that each in-range element can be accessed
    flamegpu::RunPlan * prevPtr = nullptr;
    for (uint32_t idx = 0; idx < totalPlans; idx++) {
        flamegpu::RunPlan * ptr = nullptr;
        EXPECT_NO_THROW(ptr = &plans[idx]);
        if (idx > 0) {
            EXPECT_NE(ptr, prevPtr);
        }
        prevPtr = ptr;
    }
}

/*
// setPropertyArray is only declared if SWIG is defined.
// This is not currently the case when building the test suite, so cannot be tested here.
TEST(TestRunPlanVector, setPropertyArray) {
}
*/


}  // namespace test_runplanvector
}  // namespace tests
}  // namespace flamegpu
