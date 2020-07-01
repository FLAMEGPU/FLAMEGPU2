#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

namespace test_sub_environment_description {

FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}
TEST(SubEnvironmentDescriptionTest, InvalidNames) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().add<float>("a", 0);
        m2.Environment().add<float, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().add<float>("b", 0);
        m.Environment().add<float, 2>("b2", {});
    }
    auto &sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("c", "b"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("c", "b2"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a", "c"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "c"), InvalidEnvProperty);
    EXPECT_NO_THROW(senv.mapProperty("a2", "b2"));
    EXPECT_NO_THROW(senv.mapProperty("a", "b"));
}
TEST(SubEnvironmentDescriptionTest, TypesDoNotMatch) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().add<float>("a", 0);
        m2.Environment().add<int, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().add<unsigned int>("b", 0);
        m.Environment().add<float, 2>("b2", {});
    }
    auto &sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("a", "b"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b2"), InvalidEnvProperty);
}
TEST(SubEnvironmentDescriptionTest, ElementsDoNotMatch) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().add<float>("a", 0);
        m2.Environment().add<float, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().add<float>("b", 0);
        m.Environment().add<float, 2>("b2", {});
    }
    auto &sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("a", "b2"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b"), InvalidEnvProperty);
    EXPECT_NO_THROW(senv.mapProperty("a2", "b2"));
    EXPECT_NO_THROW(senv.mapProperty("a", "b"));
}
TEST(SubEnvironmentDescriptionTest, IsConstWrong) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().add<float>("a", 0);
        m2.Environment().add<float, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().add<float>("b", 0, true);
        m.Environment().add<float, 2>("b2", {}, true);
    }
    auto &sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("a", "b"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b2"), InvalidEnvProperty);
}
TEST(SubEnvironmentDescriptionTest, AlreadyBound) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().add<float>("a", 0);
        m2.Environment().add<float, 2>("a2", {});
        m2.Environment().add<float>("a_", 0);
        m2.Environment().add<float, 2>("a2_", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().add<float>("b", 0);
        m.Environment().add<float, 2>("b2", {});
        m.Environment().add<float>("b_", 0);
        m.Environment().add<float, 2>("b2_", {});
    }
    // Missing exit condition
    auto &sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_NO_THROW(senv.mapProperty("a", "b"));
    EXPECT_NO_THROW(senv.mapProperty("a2", "b2"));
    EXPECT_THROW(senv.mapProperty("a", "b_"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b2_"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a_", "b"), InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2_", "b2"), InvalidEnvProperty);
}
};  // namespace test_sub_environment_description
