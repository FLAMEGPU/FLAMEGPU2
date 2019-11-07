#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

namespace test_sub_agent_description {

FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}
TEST(SubAgentDescriptionTest, RequiresExitCondition) {
    ModelDescription sm("sub");
    {
        // Define SubModel
        sm.newAgent("a");
    }
    ModelDescription m("host");
    auto &ma = m.newAgent("b");
    {
        // Define Model
        ma.newVariable<float>("b_float");
        ma.newVariable<unsigned int>("b_uint");
        ma.newState("b");
    }
    // Missing exit condition
    EXPECT_THROW(m.newSubModel("sub", sm), InvalidSubModel);
    sm.addExitCondition(ExitAlways);
    m.newSubModel("sub", sm);
    // Submodel name already exists
    ModelDescription sm2("sub2");
    {
        // Define SubModel
        sm2.newAgent("a");
    }
    EXPECT_THROW(m.newSubModel("sub", sm2), InvalidSubModelName);
}
TEST(SubAgentDescriptionTest, InvalidAgentName) {
    ModelDescription sm("sub");
    sm.addExitCondition(ExitAlways);
    {
        // Define SubModel
        sm.newAgent("a");
    }
    ModelDescription m("host");
    {
       m.newAgent("b");
    }
    auto &smd = m.newSubModel("sub", sm);
    // Invalid agent
    EXPECT_THROW(smd.bindAgent("c", "b", false, false), InvalidSubAgentName);
    EXPECT_THROW(smd.bindAgent("a", "c", false, false), InvalidAgentName);
    // Good
    EXPECT_NO_THROW(smd.bindAgent("a", "b", false, false));
}
TEST(SubAgentDescriptionTest, InvalidAgentState) {
    ModelDescription sm("sub");
    sm.addExitCondition(ExitAlways);
    {
        // Define SubModel
        auto &a = sm.newAgent("a");
        a.newVariable<float>("a_float");
        a.newVariable<unsigned int>("a_uint");
        a.newState("a");
        a.newState("a2");
    }
    ModelDescription m("host");
    auto &ma = m.newAgent("b");
    {
        // Define Model
        ma.newVariable<float>("b_float");
        ma.newVariable<unsigned int>("b_uint");
        ma.newState("b");
        ma.newState("b2");
    }
    auto &smd = m.newSubModel("sub", sm);
    auto &agent_map = smd.bindAgent("a", "b", false, false);
    // Invalid name
    EXPECT_THROW(agent_map.mapState("c", "b"), InvalidAgentState);
    EXPECT_THROW(agent_map.mapState("a", "c"), InvalidAgentState);
    // Good
    EXPECT_NO_THROW(agent_map.mapState("a", "b"));
    // Already Bound
    EXPECT_THROW(agent_map.mapState("a2", "b"), InvalidAgentState);
    EXPECT_THROW(agent_map.mapState("a", "b2"), InvalidAgentState);
    // Good
    EXPECT_NO_THROW(agent_map.mapState("a2", "b2"));
}
TEST(SubAgentDescriptionTest, InvalidAgentVariable) {
    ModelDescription sm("sub");
    sm.addExitCondition(ExitAlways);
    {
        // Define SubModel
        auto &a = sm.newAgent("a");
        a.newVariable<float>("a_float");
        a.newVariable<unsigned int>("a_uint");
        a.newVariable<unsigned int, 2>("a_uint2");
        a.newVariable<float>("a_float2");
        a.newState("a");
    }
    ModelDescription m("host");
    auto &ma = m.newAgent("b");
    {
        // Define Model
        ma.newVariable<float>("b_float");
        ma.newVariable<unsigned int>("b_uint");
        ma.newVariable<unsigned int, 2>("b_uint2");
        ma.newVariable<float>("b_float2");
        ma.newState("b");
    }
    auto &smd = m.newSubModel("sub", sm);
    auto &agent_map = smd.bindAgent("a", "b", false, false);
    // Bad name
    EXPECT_THROW(agent_map.mapVariable("c", "b_float"), InvalidAgentVar);
    EXPECT_THROW(agent_map.mapVariable("a_float", "c"), InvalidAgentVar);
    // Bad data type
    EXPECT_THROW(agent_map.mapVariable("a_uint", "b_float"), InvalidAgentVar);
    EXPECT_THROW(agent_map.mapVariable("a_float", "a_uint"), InvalidAgentVar);
    // Bad array length
    EXPECT_THROW(agent_map.mapVariable("a_uint", "b_uint2"), InvalidAgentVar);
    EXPECT_THROW(agent_map.mapVariable("a_uint2", "b_uint"), InvalidAgentVar);
    // Good
    EXPECT_NO_THROW(agent_map.mapVariable("a_float", "b_float"));
    EXPECT_NO_THROW(agent_map.mapVariable("a_uint", "b_uint"));
    EXPECT_NO_THROW(agent_map.mapVariable("a_uint2", "b_uint2"));
    // Already bound
    EXPECT_THROW(agent_map.mapVariable("a_float2", "b_float"), InvalidAgentVar);
    EXPECT_THROW(agent_map.mapVariable("a_float", "b_float2"), InvalidAgentVar);
    // Good
    EXPECT_NO_THROW(agent_map.mapVariable("a_float2", "b_float2"));
}
TEST(SubAgentDescriptionTest, AlreadyBound) {
    ModelDescription sm("sub");
    sm.addExitCondition(ExitAlways);
    {
        // Define SubModel
        sm.newAgent("a");
        sm.newAgent("a2");
    }
    ModelDescription m("host");
    {
        m.newAgent("b");
        m.newAgent("b2");
    }
    auto &smd = m.newSubModel("sub", sm);
    // Good
    EXPECT_NO_THROW(smd.bindAgent("a", "b", false, false));
    // Already Bound
    EXPECT_THROW(smd.bindAgent("a2", "b", false, false), InvalidAgentName);
    EXPECT_THROW(smd.bindAgent("a", "b2", false, false), InvalidSubAgentName);
    // Good
    EXPECT_NO_THROW(smd.bindAgent("a2", "b2", false, false));
}
};  // namespace test_sub_agent_description
