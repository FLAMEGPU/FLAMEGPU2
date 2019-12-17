#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_message {
    const char *MODEL_NAME = "Model";
    const char *MESSAGE_NAME1 = "Message1";
    const char *VARIABLE_NAME1 = "Var1";
    const char *VARIABLE_NAME2 = "Var2";
    const char *VARIABLE_NAME3 = "Var3";

TEST(MessageDescriptionTest, variables) {
    ModelDescription _m(MODEL_NAME);
    MessageDescription &m = _m.newMessage(MESSAGE_NAME1);
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME2));
    EXPECT_EQ(m.getVariablesCount(), 0u);
    m.newVariable<float>(VARIABLE_NAME1);
    EXPECT_EQ(m.getVariablesCount(), 1u);
    m.newVariable<int16_t>(VARIABLE_NAME2);
    EXPECT_EQ(m.getVariablesCount(), 2u);
    // Cannot create variable with same name
    EXPECT_THROW(m.newVariable<int64_t>(VARIABLE_NAME1), InvalidAgentVar);
    auto newVarArray3 = &MessageDescription::newVariable<int64_t, 3>;  // Use function ptr, can't do more than 1 template arg inside macro
    EXPECT_THROW((m.*newVarArray3)(VARIABLE_NAME1), InvalidAgentVar);
    // Variable have the right name
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME2));
    // Returned variable data is same
    EXPECT_EQ(1u, m.getVariableLength(VARIABLE_NAME1));
    EXPECT_EQ(1u, m.getVariableLength(VARIABLE_NAME2));
    EXPECT_EQ(sizeof(float), m.getVariableSize(VARIABLE_NAME1));
    EXPECT_EQ(sizeof(int16_t), m.getVariableSize(VARIABLE_NAME2));
    EXPECT_EQ(std::type_index(typeid(float)), m.getVariableType(VARIABLE_NAME1));
    EXPECT_EQ(std::type_index(typeid(int16_t)), m.getVariableType(VARIABLE_NAME2));
}
TEST(MessageDescriptionTest, variables_array) {
    ModelDescription _m(MODEL_NAME);
    MessageDescription &m = _m.newMessage(MESSAGE_NAME1);
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(m.hasVariable(VARIABLE_NAME2));
    EXPECT_EQ(m.getVariablesCount(), 0u);
    m.newVariable<float, 2>(VARIABLE_NAME1);
    EXPECT_EQ(m.getVariablesCount(), 1u);
    m.newVariable<int16_t>(VARIABLE_NAME3);
    EXPECT_EQ(m.getVariablesCount(), 2u);
    m.newVariable<int16_t, 56>(VARIABLE_NAME2);
    EXPECT_EQ(m.getVariablesCount(), 3u);
    // Cannot create variable with same name
    EXPECT_THROW(m.newVariable<int64_t>(VARIABLE_NAME1), InvalidAgentVar);
    auto newVarArray3 = &MessageDescription::newVariable<int64_t, 3>;  // Use function ptr, can't do more than 1 template arg inside macro
    EXPECT_THROW((m.*newVarArray3)(VARIABLE_NAME1), InvalidAgentVar);
    // Variable have the right name
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME1));
    EXPECT_TRUE(m.hasVariable(VARIABLE_NAME2));
    // Returned variable data is same
    EXPECT_EQ(2u, m.getVariableLength(VARIABLE_NAME1));
    EXPECT_EQ(56u, m.getVariableLength(VARIABLE_NAME2));
    EXPECT_EQ(sizeof(float), m.getVariableSize(VARIABLE_NAME1));
    EXPECT_EQ(sizeof(int16_t), m.getVariableSize(VARIABLE_NAME2));
    EXPECT_EQ(std::type_index(typeid(float)), m.getVariableType(VARIABLE_NAME1));
    EXPECT_EQ(std::type_index(typeid(int16_t)), m.getVariableType(VARIABLE_NAME2));
}

}  // namespace test_message
