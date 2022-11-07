#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_sub_environment_description {

FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}
TEST(SubEnvironmentDescriptionTest, InvalidNames) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newProperty<float>("a", 0);
        m2.Environment().newProperty<float, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newProperty<float>("b", 0);
        m.Environment().newProperty<float, 2>("b2", {});
    }
    auto sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("c", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("c", "b2"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a", "c"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "c"), exception::InvalidEnvProperty);
    EXPECT_NO_THROW(senv.mapProperty("a2", "b2"));
    EXPECT_NO_THROW(senv.mapProperty("a", "b"));
}
TEST(SubEnvironmentDescriptionTest, TypesDoNotMatch) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newProperty<float>("a", 0);
        m2.Environment().newProperty<int, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newProperty<unsigned int>("b", 0);
        m.Environment().newProperty<float, 2>("b2", {});
    }
    auto sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("a", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b2"), exception::InvalidEnvProperty);
}
TEST(SubEnvironmentDescriptionTest, ElementsDoNotMatch) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newProperty<float>("a", 0);
        m2.Environment().newProperty<float, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newProperty<float>("b", 0);
        m.Environment().newProperty<float, 2>("b2", {});
    }
    auto sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("a", "b2"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b"), exception::InvalidEnvProperty);
    EXPECT_NO_THROW(senv.mapProperty("a2", "b2"));
    EXPECT_NO_THROW(senv.mapProperty("a", "b"));
}
TEST(SubEnvironmentDescriptionTest, IsConstWrong) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newProperty<float>("a", 0);
        m2.Environment().newProperty<float, 2>("a2", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newProperty<float>("b", 0, true);
        m.Environment().newProperty<float, 2>("b2", {}, true);
    }
    auto sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapProperty("a", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b2"), exception::InvalidEnvProperty);
}
TEST(SubEnvironmentDescriptionTest, AlreadyBound) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newProperty<float>("a", 0);
        m2.Environment().newProperty<float, 2>("a2", {});
        m2.Environment().newProperty<float>("a_", 0);
        m2.Environment().newProperty<float, 2>("a2_", {});
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newProperty<float>("b", 0);
        m.Environment().newProperty<float, 2>("b2", {});
        m.Environment().newProperty<float>("b_", 0);
        m.Environment().newProperty<float, 2>("b2_", {});
    }
    // Missing exit condition
    auto sm = m.newSubModel("sub", m2);
    auto &senv = sm.SubEnvironment();
    EXPECT_NO_THROW(senv.mapProperty("a", "b"));
    EXPECT_NO_THROW(senv.mapProperty("a2", "b2"));
    EXPECT_THROW(senv.mapProperty("a", "b_"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2", "b2_"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a_", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapProperty("a2_", "b2"), exception::InvalidEnvProperty);
}
TEST(SubEnvironmentDescriptionTest, Macro_InvalidNames) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<float>("a");
        m2.Environment().newMacroProperty<float, 2, 3, 4, 5>("a2");
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<float>("b");
        m.Environment().newMacroProperty<float, 2, 3, 4, 5>("b2");
    }
    auto sm = m.newSubModel("sub", m2);
    auto& senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapMacroProperty("c", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("c", "b2"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a", "c"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a2", "c"), exception::InvalidEnvProperty);
    EXPECT_NO_THROW(senv.mapMacroProperty("a2", "b2"));
    EXPECT_NO_THROW(senv.mapMacroProperty("a", "b"));
}
TEST(SubEnvironmentDescriptionTest, Macro_TypesDoNotMatch) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<float>("a");
        m2.Environment().newMacroProperty<int, 2, 3, 4>("a2");
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<unsigned int>("b");
        m.Environment().newMacroProperty<float, 2, 3, 4>("b2");
    }
    auto sm = m.newSubModel("sub", m2);
    auto& senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapMacroProperty("a", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a2", "b2"), exception::InvalidEnvProperty);
}
TEST(SubEnvironmentDescriptionTest, Macro_DimensionsDoNotMatch) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<float, 4, 3, 2, 1>("a");
        m2.Environment().newMacroProperty<float, 1, 2, 3, 4>("a2");
        m2.Environment().newMacroProperty<float, 1, 2, 3>("a3");
        m2.Environment().newMacroProperty<float, 2, 3, 4>("a4");
        m2.Environment().newMacroProperty<float>("a5");
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<float, 4, 3, 2, 1>("b");
        m.Environment().newMacroProperty<float, 1, 2, 3, 4>("b2");
    }
    auto sm = m.newSubModel("sub", m2);
    auto& senv = sm.SubEnvironment();
    EXPECT_THROW(senv.mapMacroProperty("a", "b2"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a", "b3"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a", "b4"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a", "b5"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a2", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a3", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a4", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a5", "b"), exception::InvalidEnvProperty);
    EXPECT_NO_THROW(senv.mapMacroProperty("a2", "b2"));
    EXPECT_NO_THROW(senv.mapMacroProperty("a", "b"));
}
TEST(SubEnvironmentDescriptionTest, Macro_AlreadyBound) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<float>("a");
        m2.Environment().newMacroProperty<float, 2>("a2");
        m2.Environment().newMacroProperty<float>("a_");
        m2.Environment().newMacroProperty<float, 2>("a2_");
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<float>("b");
        m.Environment().newMacroProperty<float, 2>("b2");
        m.Environment().newMacroProperty<float>("b_");
        m.Environment().newMacroProperty<float, 2>("b2_");
    }
    // Missing exit condition
    auto sm = m.newSubModel("sub", m2);
    auto& senv = sm.SubEnvironment();
    EXPECT_NO_THROW(senv.mapMacroProperty("a", "b"));
    EXPECT_NO_THROW(senv.mapMacroProperty("a2", "b2"));
    EXPECT_THROW(senv.mapMacroProperty("a", "b_"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a2", "b2_"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a_", "b"), exception::InvalidEnvProperty);
    EXPECT_THROW(senv.mapMacroProperty("a2_", "b2"), exception::InvalidEnvProperty);
}
};  // namespace test_sub_environment_description
}  // namespace flamegpu
