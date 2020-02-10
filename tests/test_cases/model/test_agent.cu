#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_agent {
const char *MODEL_NAME = "Model";
const char *AGENT_NAME1 = "Agent1";
const char *AGENT_NAME2 = "Agent2";
const char *VARIABLE_NAME1 = "Var1";
const char *VARIABLE_NAME2 = "Var2";
const char *VARIABLE_NAME3 = "Var3";
const char *VARIABLE_NAME4 = "Var4";
const char *FUNCTION_NAME1 = "Func1";
const char *FUNCTION_NAME2 = "Func2";
const char *STATE_NAME1 = "State1";
const char *STATE_NAME2 = "State2";

FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn2, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}

TEST(AgentDescriptionTest, functions) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasFunction(FUNCTION_NAME1));
    EXPECT_FALSE(a.hasFunction(FUNCTION_NAME2));
    EXPECT_EQ(a.getFunctionsCount(), 0u);
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    EXPECT_EQ(a.getFunctionsCount(), 1u);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    EXPECT_EQ(a.getFunctionsCount(), 2u);
    // Cannot create function with same name
    EXPECT_THROW(a.newFunction(FUNCTION_NAME1, agent_fn1), InvalidAgentFunc);
    EXPECT_THROW(a.newFunction(FUNCTION_NAME1, agent_fn2), InvalidAgentFunc);
    EXPECT_THROW(a.newFunction(FUNCTION_NAME2, agent_fn2), InvalidAgentFunc);
    // Functions have the right name
    EXPECT_TRUE(a.hasFunction(FUNCTION_NAME1));
    EXPECT_TRUE(a.hasFunction(FUNCTION_NAME2));
    // Returned function data is same
    EXPECT_EQ(f1, a.getFunction(FUNCTION_NAME1));
    EXPECT_EQ(f2, a.getFunction(FUNCTION_NAME2));
    EXPECT_EQ(f1, a.Function(FUNCTION_NAME1));
    EXPECT_EQ(f2, a.Function(FUNCTION_NAME2));
    EXPECT_EQ(f1.getName(), FUNCTION_NAME1);
    EXPECT_EQ(f2.getName(), FUNCTION_NAME2);
    {
        AgentFunctionWrapper *_a = &agent_function_wrapper<agent_fn1_impl, MsgNone, MsgNone>;
        EXPECT_EQ(f1.getFunctionPtr(), _a);
        AgentFunctionWrapper *_b = &agent_function_wrapper<agent_fn2_impl, MsgNone, MsgNone>;
        EXPECT_EQ(f2.getFunctionPtr(), _b);
    }
}
TEST(AgentDescriptionTest, variables) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME2));
    EXPECT_EQ(a.getVariablesCount(), 0u);
    a.newVariable<float>(VARIABLE_NAME1);
    EXPECT_EQ(a.getVariablesCount(), 1u);
    a.newVariable<int16_t>(VARIABLE_NAME2);
    EXPECT_EQ(a.getVariablesCount(), 2u);
    // Cannot create variable with same name
    EXPECT_THROW(a.newVariable<int64_t>(VARIABLE_NAME1), InvalidAgentVar);
    auto newVarArray3 = &AgentDescription::newVariable<int64_t, 3>;  // Use function ptr, can't do more than 1 template arg inside macro
    EXPECT_THROW((a.*newVarArray3)(VARIABLE_NAME1), InvalidAgentVar);
    // Variable have the right name
    EXPECT_TRUE(a.hasVariable(VARIABLE_NAME1));
    EXPECT_TRUE(a.hasVariable(VARIABLE_NAME2));
    // Returned variable data is same
    EXPECT_EQ(1u, a.getVariableLength(VARIABLE_NAME1));
    EXPECT_EQ(1u, a.getVariableLength(VARIABLE_NAME2));
    EXPECT_EQ(sizeof(float), a.getVariableSize(VARIABLE_NAME1));
    EXPECT_EQ(sizeof(int16_t), a.getVariableSize(VARIABLE_NAME2));
    EXPECT_EQ(std::type_index(typeid(float)), a.getVariableType(VARIABLE_NAME1));
    EXPECT_EQ(std::type_index(typeid(int16_t)), a.getVariableType(VARIABLE_NAME2));
}
TEST(AgentDescriptionTest, variables_array) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME2));
    EXPECT_EQ(a.getVariablesCount(), 0u);
    a.newVariable<float, 2>(VARIABLE_NAME1);
    EXPECT_EQ(a.getVariablesCount(), 1u);
    a.newVariable<int16_t>(VARIABLE_NAME3);
    EXPECT_EQ(a.getVariablesCount(), 2u);
    a.newVariable<int16_t, 56>(VARIABLE_NAME2);
    EXPECT_EQ(a.getVariablesCount(), 3u);
    // Cannot create variable with same name
    EXPECT_THROW(a.newVariable<int64_t>(VARIABLE_NAME1), InvalidAgentVar);
    auto newVarArray3 = &AgentDescription::newVariable<int64_t>;  // Use function ptr, can't do more than 1 template arg inside macro
    EXPECT_THROW((a.*newVarArray3)(VARIABLE_NAME1), InvalidAgentVar);
    // Cannot create array of length 0 (disabled, blocked at compilation with static_assert)
    // auto newVarArray0 = &AgentDescription::newVariable<int64_t, 0>;  // Use function ptr, can't do more than 1 template arg inside macro
    // EXPECT_THROW((a.*newVarArray0)(VARIABLE_NAME4), InvalidAgentVar);
    // Variable have the right name
    EXPECT_TRUE(a.hasVariable(VARIABLE_NAME1));
    EXPECT_TRUE(a.hasVariable(VARIABLE_NAME2));
    // Returned variable data is same
    EXPECT_EQ(2u, a.getVariableLength(VARIABLE_NAME1));
    EXPECT_EQ(56u, a.getVariableLength(VARIABLE_NAME2));
    EXPECT_EQ(sizeof(float), a.getVariableSize(VARIABLE_NAME1));
    EXPECT_EQ(sizeof(int16_t), a.getVariableSize(VARIABLE_NAME2));
    EXPECT_EQ(std::type_index(typeid(float)), a.getVariableType(VARIABLE_NAME1));
    EXPECT_EQ(std::type_index(typeid(int16_t)), a.getVariableType(VARIABLE_NAME2));
}
TEST(AgentDescriptionTest, states) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasState(STATE_NAME1));
    EXPECT_FALSE(a.hasState(STATE_NAME2));
    EXPECT_EQ(a.getStatesCount(), 1u);  // Initially just default state
    a.newState(STATE_NAME1);
    EXPECT_EQ(a.getStatesCount(), 1u);  // Remains 1 state after first set
    a.newState(STATE_NAME2);
    EXPECT_EQ(a.getStatesCount(), 2u);
    // Cannot create state with same name
    EXPECT_THROW(a.newState(STATE_NAME1), InvalidStateName);
    EXPECT_THROW(a.newState(STATE_NAME2), InvalidStateName);
    // States have the right name
    EXPECT_TRUE(a.hasState(STATE_NAME1));
    EXPECT_TRUE(a.hasState(STATE_NAME2));
}
TEST(AgentDescriptionTest, initial_state1) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    // Initial state starts out default
    EXPECT_EQ(a.getInitialState(), ModelData::DEFAULT_STATE);
    // Initial state changes when first state added
    a.newState(STATE_NAME1);
    EXPECT_EQ(a.getInitialState(), STATE_NAME1);
    // Initial state does not change when next state added
    a.newState(STATE_NAME2);
    EXPECT_EQ(a.getInitialState(), STATE_NAME1);
}
TEST(AgentDescriptionTest, initial_state2) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    // Initial state starts out default
    EXPECT_EQ(a.getInitialState(), ModelData::DEFAULT_STATE);
    // Initial state changes when first state added
    a.newState(ModelData::DEFAULT_STATE);
    EXPECT_EQ(a.getStatesCount(), 1u);  // Remains 1 state after first set
    EXPECT_EQ(a.getInitialState(), ModelData::DEFAULT_STATE);
    // Initial state does not change when next state added
    a.newState(STATE_NAME2);
    EXPECT_EQ(a.getInitialState(), ModelData::DEFAULT_STATE);
    EXPECT_EQ(a.getStatesCount(), 2u);  // Increases to 2 state
}
TEST(AgentDescriptionTest, agent_outputs) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    AgentDescription &b = m.newAgent(AGENT_NAME2);
    EXPECT_EQ(a.getAgentOutputsCount(), 0u);
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    // Count increases as we set values
    f1.setAgentOutput(a);
    EXPECT_EQ(a.getAgentOutputsCount(), 1u);
    f2.setAgentOutput(a);
    EXPECT_EQ(a.getAgentOutputsCount(), 2u);
    // Replacing value doesnt break the count
    f2.setAgentOutput(a);
    EXPECT_EQ(a.getAgentOutputsCount(), 2u);
    f2.setAgentOutput(b);
    EXPECT_EQ(a.getAgentOutputsCount(), 1u);
}

}  // namespace test_agent
