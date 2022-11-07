#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace tests {
namespace test_runplan {

TEST(TestRunPlan, constructor) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Declare a pointer
    flamegpu::RunPlan * plan = nullptr;
    // Use New
    EXPECT_NO_THROW(plan = new flamegpu::RunPlan(model));
    EXPECT_NE(plan, nullptr);
    // Run the destructor
    EXPECT_NO_THROW(delete plan);
    plan = nullptr;
}
TEST(TestRunPlan, setRandomSimulationSeed) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Get the original simulation seed. Cannot compare for an expected value as any uint64_t is potentially legitimate.
    EXPECT_NO_THROW(plan.getRandomSimulationSeed());
    // Set to max 32 bit value +1
    uint64_t newSimulationSeed = UINT_MAX + 1;
    plan.setRandomSimulationSeed(newSimulationSeed);
    // Get the value again, and expect it to be the set value. It is not guaranteed to not be the original random value.
    uint64_t simulationSeedUpdated = plan.getRandomSimulationSeed();
    EXPECT_EQ(newSimulationSeed, simulationSeedUpdated);
    // Set it again, this time passing a narrower number.
    uint32_t narrowSimulationSeed = 12u;
    plan.setRandomSimulationSeed(narrowSimulationSeed);
    // Get the seed again, into a narrow value
    uint32_t narrowSimulationSeedUpdated = static_cast<uint32_t>(plan.getRandomSimulationSeed());
    EXPECT_EQ(narrowSimulationSeed, narrowSimulationSeedUpdated);
}
TEST(TestRunPlan, setSteps) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Get the default value
    uint32_t steps = plan.getSteps();
    EXPECT_EQ(steps, 1u);
    // Set a new value
    uint32_t newSteps = 12u;
    plan.setSteps(newSteps);
    // Get the updated value and compare
    uint32_t updatedSteps = plan.getSteps();
    EXPECT_EQ(updatedSteps, newSteps);

    // Expected exception tests
    EXPECT_THROW(plan.setSteps(0u), exception::OutOfBoundsException);
}
TEST(TestRunPlan, setOutputSubdirectory) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Set the subdirectory to a non empty string
    std::string newSubdir("test");
    plan.setOutputSubdirectory(newSubdir);
    // Get the original value
    std::string updatedSubdir = plan.getOutputSubdirectory();
    // By default this is an empty string
    EXPECT_EQ(updatedSubdir, newSubdir);
}
TEST(TestRunPlan, setProperty) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Add some properties to the model, using a range of types.
    auto &environment = model.Environment();
    environment.newProperty<float>("f", 1.0f);
    environment.newProperty<int32_t>("i", -1);
    environment.newProperty<uint32_t>("u", 1u);
    environment.newProperty<float, 3>("f_a", {-1.0f, 0.0f, 1.0f});
    environment.newProperty<int32_t, 3>("i_a", {-1, 0, 1 });
    environment.newProperty<uint32_t, 3>("u_a", {0, 1, 2 });
#ifdef USE_GLM
    environment.newProperty<glm::ivec3>("ivec3", {});
    environment.newProperty<glm::ivec3, 2>("ivec32", {});
    environment.newProperty<glm::ivec3, 3>("ivec33", {});
#endif
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Set properties to new values
    // Compare the old and new values, to ensure that thy do not match
    // RunPlan::setProperty(const std::string &name, T value)
    plan.setProperty<float>("f", 2.0f);
    plan.setProperty<int32_t>("i", 2);
    plan.setProperty<uint32_t>("u", 2u);
    // Set arrays at once
    // RunPlan::setProperty(const std::string &name, const std::array<T, N> &value)
    plan.setProperty<float, 3>("f_a", {-2.0f, 0.0f, 2.0f});
    plan.setProperty<int32_t, 3>("i_a", {-2, 0, 2});
    // Set individual elements at a time
    // RunPlan::setProperty(const std::string &name, flamegpu::size_type index, T value)
    plan.setProperty<uint32_t>("u_a", 0, 3u);
    plan.setProperty<uint32_t>("u_a", 1, 4u);
    plan.setProperty<uint32_t>("u_a", 2, 5u);
#ifdef USE_GLM
    const glm::ivec3 ivec3_1_check = glm::ivec3{ 1, 2, 3 };
    const std::array<glm::ivec3, 2> ivec3_2_check = { glm::ivec3{4, 5, 6}, glm::ivec3{7, 8, 9} };
    const std::array<glm::ivec3, 3> ivec3_3_check =
    { glm::ivec3{ 11, 12, 13 }, glm::ivec3{14, 15, 16}, glm::ivec3{17, 18, 19} };
    plan.setProperty<glm::ivec3>("ivec3", ivec3_1_check);
    plan.setProperty<glm::ivec3, 2>("ivec32", ivec3_2_check);
    plan.setProperty<glm::ivec3, 3>("ivec33", ivec3_3_check);
#endif

    EXPECT_EQ(plan.getProperty<float>("f"), 2.0f);
    EXPECT_EQ(plan.getProperty<int32_t>("i"), 2);
    EXPECT_EQ(plan.getProperty<uint32_t>("u"), 2u);
    // extra brackets allow commas in macros.
    EXPECT_EQ((plan.getProperty<float, 3>("f_a")), (std::array<float, 3>{ -2.0f, 0.0f, 2.0f }));
    EXPECT_EQ((plan.getProperty<int32_t, 3>("i_a")), (std::array<int32_t, 3>{ -2, 0, 2 }));
    EXPECT_EQ(plan.getProperty<uint32_t>("u_a", 0), 3u);
    EXPECT_EQ(plan.getProperty<uint32_t>("u_a", 1), 4u);
    EXPECT_EQ(plan.getProperty<uint32_t>("u_a", 2), 5u);
#ifdef USE_GLM
    EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec3"), ivec3_1_check);
    EXPECT_EQ((plan.getProperty<glm::ivec3, 3>)("ivec33"), ivec3_3_check);
    EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec32", 0), ivec3_2_check[0]);
    EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec32", 1), ivec3_2_check[1]);
#endif

    // Update properties again (previous bug)
    // RunPlan::setProperty(const std::string &name, T value)
    plan.setProperty<float>("f", 3.0f);
    plan.setProperty<int32_t>("i", 3);
    plan.setProperty<uint32_t>("u", 3u);
    // Set arrays at once
    // RunPlan::setProperty(const std::string &name, const std::array<T, N> &value)
    plan.setProperty<float, 3>("f_a", { 3.0f, 0.0f, -3.0f });
    plan.setProperty<int32_t, 3>("i_a", { 3, 0, 5 });
    // Set individual elements at a time
    // RunPlan::setProperty(const std::string &name, flamegpu::size_type index, T value)
    plan.setProperty<uint32_t>("u_a", 0, 13u);
    plan.setProperty<uint32_t>("u_a", 1, 14u);
    plan.setProperty<uint32_t>("u_a", 2, 15u);
#ifdef USE_GLM
    plan.setProperty<glm::ivec3>("ivec3", glm::ivec3{ 31, 32, 33 });
    const std::array<glm::ivec3, 3> ivec3_3_check2 =
    { glm::ivec3{ 41, 42, 43 }, glm::ivec3{44, 45, 46}, glm::ivec3{47, 48, 49} };
    plan.setProperty<glm::ivec3, 3>("ivec33", ivec3_3_check2);
    plan.setProperty<glm::ivec3>("ivec32", 0, ivec3_2_check[1]);
    plan.setProperty<glm::ivec3>("ivec32", 1, ivec3_2_check[0]);
#endif

    EXPECT_EQ(plan.getProperty<float>("f"), 3.0f);
    EXPECT_EQ(plan.getProperty<int32_t>("i"), 3);
    EXPECT_EQ(plan.getProperty<uint32_t>("u"), 3u);
    // extra brackets allow commas in macros.
    EXPECT_EQ((plan.getProperty<float, 3>("f_a")), (std::array<float, 3>{ 3.0f, 0.0f, -3.0f }));
    EXPECT_EQ((plan.getProperty<int32_t, 3>("i_a")), (std::array<int32_t, 3>{ 3, 0, 5 }));
    EXPECT_EQ(plan.getProperty<uint32_t>("u_a", 0), 13u);
    EXPECT_EQ(plan.getProperty<uint32_t>("u_a", 1), 14u);
    EXPECT_EQ(plan.getProperty<uint32_t>("u_a", 2), 15u);
#ifdef USE_GLM
    EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec3"), glm::ivec3(31, 32, 33));
    EXPECT_EQ((plan.getProperty<glm::ivec3, 3>)("ivec33"), ivec3_3_check2);
    EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec32", 0), ivec3_2_check[1]);
    EXPECT_EQ(plan.getProperty<glm::ivec3>("ivec32", 1), ivec3_2_check[0]);
#endif


    // Tests for exceptions
    // --------------------
    // Note literals used must match the templated type not the incorrect types used, to appease MSVC warnings.
    // RunPlan::setProperty(const std::string &name, T value)
    EXPECT_THROW(plan.setProperty<float>("does_not_exist", 1.f), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW(plan.setProperty<float>("i", 1.f), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW(plan.setProperty<uint32_t>("u_a", 1u), flamegpu::exception::InvalidEnvPropertyType);
    // RunPlan::setProperty(const std::string &name, flamegpu::size_type index, T value)
    // Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
    EXPECT_THROW((plan.setProperty<float, 3>("does_not_exist", {2.f, 2.f, 2.f})), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plan.setProperty<float, 3>("u_a", {2.f, 2.f, 2.f})), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.setProperty<int32_t, 2>("i_a", {-2, 0})), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.setProperty<int32_t, 4>("i_a", {-2, 0, 2, 2})), flamegpu::exception::InvalidEnvPropertyType);
    // RunPlan::setProperty(const std::string &name, flamegpu::size_type index, T value)
    EXPECT_THROW((plan.setProperty<float>("does_not_exist", 0u, 3.f)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plan.setProperty<float>("u_a", 0u, 3.f)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.setProperty<int32_t>("i_a", static_cast<flamegpu::size_type>(-1), 3)), exception::OutOfBoundsException);
    EXPECT_THROW((plan.setProperty<int32_t>("i_a", 4u, 3)), exception::OutOfBoundsException);

    // RunPlan::getProperty(const std::string &name)
    EXPECT_THROW(plan.getProperty<float>("does_not_exist"), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW(plan.getProperty<float>("i"), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW(plan.getProperty<uint32_t>("u_a"), flamegpu::exception::InvalidEnvPropertyType);
    // RunPlan::getProperty(const std::string &name, flamegpu::size_type index)
    // Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
    EXPECT_THROW((plan.getProperty<float, 3>("does_not_exist")), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plan.getProperty<float, 3>("u_a")), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.getProperty<int32_t, 2>("i_a")), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.getProperty<int32_t, 4>("i_a")), flamegpu::exception::InvalidEnvPropertyType);
    // RunPlan::getProperty(const std::string &name, flamegpu::size_type index)
    EXPECT_THROW((plan.getProperty<float>("does_not_exist", 0u)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plan.getProperty<float>("u_a", 0u)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.getProperty<int32_t>("i_a", static_cast<flamegpu::size_type>(-1))), exception::OutOfBoundsException);
    EXPECT_THROW((plan.getProperty<int32_t>("i_a", 4u)), exception::OutOfBoundsException);
#ifdef USE_GLM
    EXPECT_THROW((plan.setProperty<glm::ivec3>)("ivec32", 3u, {}), exception::OutOfBoundsException);  // Out of bounds
    EXPECT_THROW((plan.setProperty<glm::ivec3>)("ivec33", 4u, {}), exception::OutOfBoundsException);  // Out of bounds
    EXPECT_THROW((plan.getProperty<glm::ivec3>)("ivec32", 3u), exception::OutOfBoundsException);  // Out of bounds
    EXPECT_THROW((plan.getProperty<glm::ivec3>)("ivec33", 4u), exception::OutOfBoundsException);  // Out of bounds
#endif
}
TEST(TestRunPlan, getRandomSimulationSeed) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Get the Simulation seed
    // As this is random, it could be any value. So get it twice, and make sure the same thing was returned?
    uint64_t simulationSeed = plan.getRandomSimulationSeed();
    uint64_t simulationSeedAgain = plan.getRandomSimulationSeed();
    EXPECT_EQ(simulationSeed, simulationSeedAgain);
}
TEST(TestRunPlan, getSteps) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Get the default value
    uint32_t steps = plan.getSteps();
    EXPECT_EQ(steps, 1u);
}
TEST(TestRunPlan, getOutputSubdirectory) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Get the default value
    std::string subdir = plan.getOutputSubdirectory();
    // By default this is an empty string
    EXPECT_EQ(subdir, "");
}
TEST(TestRunPlan, getProperty) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Add some properties to the model, using a range of types.
    auto &environment = model.Environment();
    environment.newProperty<float>("f", 1.0f);
    environment.newProperty<int32_t>("i", -1);
    environment.newProperty<uint32_t>("u", 1u);
    environment.newProperty<float, 3>("f_a", {-1.0f, 0.0f, 1.0f});
    environment.newProperty<int32_t, 3>("i_a", {-1, 0, 1 });
    environment.newProperty<uint32_t, 3>("u_a", {0, 1, 2 });
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Check that they match the original value when no overrides have been set.
    EXPECT_EQ(plan.getProperty<float>("f"), environment.getProperty<float>("f"));
    EXPECT_EQ(plan.getProperty<int32_t>("i"), environment.getProperty<int32_t>("i"));
    EXPECT_EQ(plan.getProperty<uint32_t>("u"), environment.getProperty<uint32_t>("u"));
    // extra brackets allow commas in macros.
    EXPECT_EQ((plan.getProperty<float, 3>("f_a")), (environment.getProperty<float, 3>("f_a")));
    EXPECT_EQ((plan.getProperty<int32_t, 3>("i_a")), (environment.getProperty<int32_t, 3>("i_a")));
    EXPECT_EQ((plan.getProperty<uint32_t, 3>("u_a")), (environment.getProperty<uint32_t, 3>("u_a")));

    // Tests for exceptions
    // --------------------
    // Note litereals used must match the templated type not the incorrect types used, to appease MSVC warnings.
    // RunPlan::getProperty(const std::string &name) const
    EXPECT_THROW(plan.getProperty<float>("does_not_exist"), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW(plan.getProperty<float>("i"), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW(plan.getProperty<uint32_t>("u_a"), flamegpu::exception::InvalidEnvPropertyType);
    // std::array<T, N> RunPlan::getProperty(const std::string &name) const
    // Extra brackets within the macro mean commas can be used due to how preproc tokenizers work
    EXPECT_THROW((plan.getProperty<float, 3>("does_not_exist")), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plan.getProperty<float, 3>("u_a")), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.getProperty<int32_t, 2>("i_a")), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.getProperty<int32_t, 4>("i_a")), flamegpu::exception::InvalidEnvPropertyType);
    // T RunPlan::getProperty(const std::string &name, flamegpu::size_type index) const
    EXPECT_THROW((plan.getProperty<float>("does_not_exist", 0u)), flamegpu::exception::InvalidEnvProperty);
    EXPECT_THROW((plan.getProperty<float>("u_a", 0u)), flamegpu::exception::InvalidEnvPropertyType);
    EXPECT_THROW((plan.getProperty<int32_t>("i_a", static_cast<flamegpu::size_type>(-1))), exception::OutOfBoundsException);
    EXPECT_THROW((plan.getProperty<int32_t>("i_a", 4u)), exception::OutOfBoundsException);
}
// @todo - This test could be a lot more thorough.
TEST(TestRunPlan, operatorAssignment) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create two separate RunPlans with unique values
    flamegpu::RunPlan plan1(model);
    const uint64_t seed1 = 1u;
    plan1.setRandomSimulationSeed(seed1);
    flamegpu::RunPlan plan2(model);
    const uint64_t seedB = 2u;
    plan1.setRandomSimulationSeed(seedB);
    // Verify properties are unique
    EXPECT_NE(plan1.getRandomSimulationSeed(), plan2.getRandomSimulationSeed());
    // use the assignment operator to set plan1=plan2, then check the unique value(s) are correct.
    plan1 = plan2;
    EXPECT_EQ(plan1.getRandomSimulationSeed(), plan2.getRandomSimulationSeed());
}
TEST(TestRunPlan, operatorAddition) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create multiple run plans and set unique values on each
    flamegpu::RunPlan plan1(model);
    const uint64_t seed1 = 1u;
    plan1.setRandomSimulationSeed(seed1);
    flamegpu::RunPlan plan2(model);
    const uint64_t seed2 = 2u;
    plan2.setRandomSimulationSeed(seed2);
    flamegpu::RunPlan plan3(model);
    const uint64_t seed3 = 3u;
    plan3.setRandomSimulationSeed(seed3);

    // RunPlanVector = RunPlan + RunPlan
    flamegpu::RunPlanVector vec12 = plan1 + plan2;
    EXPECT_EQ(vec12.size(), 2);
    EXPECT_EQ(vec12[0].getRandomSimulationSeed(), seed1);
    EXPECT_EQ(vec12[1].getRandomSimulationSeed(), seed2);

    /* Disabled for now, operator+ is always an append for performance reasons (currently) 
    // RunPlanVector = RunPlan + RunPlanVector
    // As an prepend?
    flamegpu::RunPlanVector vec312 = plan3 + vec12;
    EXPECT_EQ(vec312.size(), 3);
    EXPECT_EQ(vec312[0].getRandomSimulationSeed(), seed3);
    EXPECT_EQ(vec312[1].getRandomSimulationSeed(), seed1);
    EXPECT_EQ(vec312[2].getRandomSimulationSeed(), seed2); 
    */

    // Try with operators in the other order.
    // As an append.
    flamegpu::RunPlanVector vec123 = vec12 + plan3;
    EXPECT_EQ(vec123.size(), 3);
    EXPECT_EQ(vec123[0].getRandomSimulationSeed(), seed1);
    EXPECT_EQ(vec123[1].getRandomSimulationSeed(), seed2);
    EXPECT_EQ(vec123[2].getRandomSimulationSeed(), seed3);

    // Expected exceptions
    // -------------------
    // Adding runplans together which are not for the same model (actually environment) should throw.
    flamegpu::ModelDescription otherModel("other");
    otherModel.Environment().newProperty<float>("f", 1.0f);  // If both models have null environments they are compatible
    flamegpu::RunPlan otherPlan(otherModel);
    flamegpu::RunPlanVector otherPlanVector(otherModel, 1);
    EXPECT_THROW((plan1 + otherPlan), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((plan1 + otherPlanVector), flamegpu::exception::InvalidArgument);
    EXPECT_THROW((otherPlan + plan1), flamegpu::exception::InvalidArgument);
}
// RunPLanVector = RunPlan * uint32_t
TEST(TestRunPlan, operatorMultiplication) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an individual run plan.
    flamegpu::RunPlan plan(model);
    // Set a value to a non default value to allow comparison
    const uint64_t newSimulationSeed = 12u;
    plan.setRandomSimulationSeed(newSimulationSeed);
    // Create a RunPlanVector of N elemets
    const uint32_t N = 4u;
    flamegpu::RunPlanVector plans = plan * N;
    EXPECT_EQ(plans.size(), N);
    // Compare each element
    for (const auto &p : plans) {
        EXPECT_EQ(p.getRandomSimulationSeed(),  newSimulationSeed);
        // @todo - compare more than one part? Operator== might be easier / cleaner
    }
}
/*
getPropertyArray/setPropertyArray are only declared if SWIG is defined.
// This is not currently the case when building the test suite, so these cannot be tested here.
TEST(TestRunPlan, getPropertyArray) {
}
TEST(TestRunPlan, setPropertyArray) {
}
*/

}  // namespace test_runplan
}  // namespace tests
}  // namespace flamegpu
