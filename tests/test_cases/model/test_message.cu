#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace test_message {
    const char *MODEL_NAME = "Model";
    const char *MESSAGE_NAME1 = "Message1";
    const char *VARIABLE_NAME1 = "Var1";
    const char *VARIABLE_NAME2 = "Var2";
    const char *VARIABLE_NAME3 = "Var3";
    const char *VARIABLE_NAME4 = "Var4";

TEST(MessageDescriptionTest, variables) {
    ModelDescription _m(MODEL_NAME);
    MsgBruteForce::Description &m = _m.newMessage(MESSAGE_NAME1);
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME2));
    EXPECT_EQ(m.getVariablesCount(), 0u);
    m.newVariable<float>(VARIABLE_NAME1);
    EXPECT_EQ(m.getVariablesCount(), 1u);
    m.newVariable<int16_t>(VARIABLE_NAME2);
    EXPECT_EQ(m.getVariablesCount(), 2u);
    // Cannot create variable with same name
    EXPECT_THROW(m.newVariable<int64_t>(VARIABLE_NAME1), InvalidMessageVar);
    // Variable have the right name
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME2));
    // Returned variable data is same
    EXPECT_EQ(sizeof(float), m.getVariableSize(VARIABLE_NAME1));
    EXPECT_EQ(std::type_index(typeid(float)), m.getVariableType(VARIABLE_NAME1));
    EXPECT_EQ(sizeof(int16_t), m.getVariableSize(VARIABLE_NAME2));
    EXPECT_EQ(std::type_index(typeid(int16_t)), m.getVariableType(VARIABLE_NAME2));
}
TEST(MessageDescriptionTest, variables_array) {
    ModelDescription _m(MODEL_NAME);
    MsgBruteForce::Description &m = _m.newMessage(MESSAGE_NAME1);
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME2));
    EXPECT_EQ(m.getVariablesCount(), 0u);
    m.newVariable<float>(VARIABLE_NAME1);
    EXPECT_EQ(m.getVariablesCount(), 1u);
    m.newVariable<int16_t>(VARIABLE_NAME3);
    EXPECT_EQ(m.getVariablesCount(), 2u);
    m.newVariable<int16_t>(VARIABLE_NAME2);
    EXPECT_EQ(m.getVariablesCount(), 3u);
    // Cannot create variable with same name
    EXPECT_THROW(m.newVariable<int64_t>(VARIABLE_NAME1), InvalidMessageVar);
    // Cannot create array of length 0 (disabled, blocked at compilation with static_assert)
    // auto newVarArray0 = &MessageDescription::newVariable<int64_t, 0>;  // Use function ptr, can't do more than 1 template arg inside macro
    // EXPECT_THROW((m.*newVarArray0)(VARIABLE_NAME4), InvalidMessageVar);
    // Variable have the right name
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME2));
    // Returned variable data is same
    EXPECT_EQ(sizeof(float), m.getVariableSize(VARIABLE_NAME1));
    EXPECT_EQ(sizeof(int16_t), m.getVariableSize(VARIABLE_NAME2));
    EXPECT_EQ(std::type_index(typeid(float)), m.getVariableType(VARIABLE_NAME1));
    EXPECT_EQ(std::type_index(typeid(int16_t)), m.getVariableType(VARIABLE_NAME2));
}

FLAMEGPU_AGENT_FUNCTION(NoInput, MsgNone, MsgSpatial3D) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(NoOutput, MsgSpatial2D, MsgNone) {
    return ALIVE;
}

TEST(MessageDescriptionTest, CorrectMessageTypeBound1) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent("foo");
    AgentFunctionDescription &fo = a.newFunction("bar", NoInput);
    LayerDescription &lo = m.newLayer("foo2");
    lo.addAgentFunction(fo);
    EXPECT_THROW(CUDASimulation c(m), InvalidMessageType);
}
TEST(MessageDescriptionTest, CorrectMessageTypeBound2) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent("foo");
    AgentFunctionDescription &fo = a.newFunction("bar", NoOutput);
    LayerDescription &lo = m.newLayer("foo2");
    lo.addAgentFunction(fo);
    EXPECT_THROW(CUDASimulation c(m), InvalidMessageType);
}
TEST(MessageDescriptionTest, CorrectMessageTypeBound3) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent("foo");
    AgentFunctionDescription &fo = a.newFunction("bar", NoInput);
    MsgBruteForce::Description &md = m.newMessage<MsgBruteForce>("foo2");
    EXPECT_THROW(fo.setMessageOutput(md), InvalidMessageType);
    EXPECT_THROW(fo.setMessageInput(md), InvalidMessageType);
}
TEST(MessageDescriptionTest, CorrectMessageTypeBound4) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent("foo");
    AgentFunctionDescription &fo = a.newFunction("bar", NoOutput);
    MsgBruteForce::Description &md = m.newMessage<MsgBruteForce>("foo2");
    EXPECT_THROW(fo.setMessageOutput(md), InvalidMessageType);
    EXPECT_THROW(fo.setMessageInput(md), InvalidMessageType);
}

}  // namespace test_message
