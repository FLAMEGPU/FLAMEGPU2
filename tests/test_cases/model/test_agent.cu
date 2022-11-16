#include <fstream>
#include <string>
#include <iostream>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

// Putting the tests inside the namespace is the minimal effort method of switching over
namespace flamegpu {
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

FLAMEGPU_AGENT_FUNCTION(agent_fn1, MessageNone, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn2, MessageNone, MessageNone) {
    // do nothing
    return ALIVE;
}

TEST(AgentDescriptionTest, functions) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasFunction(FUNCTION_NAME1));
    EXPECT_FALSE(a.hasFunction(FUNCTION_NAME2));
    EXPECT_EQ(a.getFunctionsCount(), 0u);
    AgentFunctionDescription f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    EXPECT_EQ(a.getFunctionsCount(), 1u);
    AgentFunctionDescription f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    EXPECT_EQ(a.getFunctionsCount(), 2u);
    // Cannot create function with same name
    EXPECT_THROW(a.newFunction(FUNCTION_NAME1, agent_fn1), exception::InvalidAgentFunc);
    EXPECT_THROW(a.newFunction(FUNCTION_NAME1, agent_fn2), exception::InvalidAgentFunc);
    EXPECT_THROW(a.newFunction(FUNCTION_NAME2, agent_fn2), exception::InvalidAgentFunc);
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
        AgentFunctionWrapper *_a = &agent_function_wrapper<agent_fn1_impl, MessageNone, MessageNone>;
        EXPECT_EQ(f1.getFunctionPtr(), _a);
        AgentFunctionWrapper *_b = &agent_function_wrapper<agent_fn2_impl, MessageNone, MessageNone>;
        EXPECT_EQ(f2.getFunctionPtr(), _b);
    }
}
TEST(AgentDescriptionTest, variables) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME2));
    // When created, agent has 1 internal variable _id
    EXPECT_EQ(a.getVariablesCount(), 1u);
    a.newVariable<float>(VARIABLE_NAME1);
    EXPECT_EQ(a.getVariablesCount(), 2u);
    a.newVariable<int16_t>(VARIABLE_NAME2);
    EXPECT_EQ(a.getVariablesCount(), 3u);
    // Cannot create variable with same name
    EXPECT_THROW(a.newVariable<int64_t>(VARIABLE_NAME1), exception::InvalidAgentVar);
    auto newVarArray3 = &AgentDescription::newVariable<int64_t, 3>;  // Use function ptr, can't do more than 1 template arg inside macro
    EXPECT_THROW((a.*newVarArray3)(VARIABLE_NAME1, { }), exception::InvalidAgentVar);
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
#ifdef USE_GLM
    // Can create variable with GLM types
    a.newVariable<glm::vec3>("vec3");
    a.newVariable<glm::uvec4>("uvec4");
    EXPECT_EQ(a.getVariablesCount(), 5u);
    EXPECT_EQ(3u, a.getVariableLength("vec3"));
    EXPECT_EQ(4u, a.getVariableLength("uvec4"));
    EXPECT_EQ(sizeof(float), a.getVariableSize("vec3"));
    EXPECT_EQ(sizeof(unsigned int), a.getVariableSize("uvec4"));
    EXPECT_EQ(std::type_index(typeid(float)), a.getVariableType("vec3"));
    EXPECT_EQ(std::type_index(typeid(unsigned int)), a.getVariableType("uvec4"));
#endif
}
TEST(AgentDescriptionTest, variables_array) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME1));
    EXPECT_FALSE(a.hasVariable(VARIABLE_NAME2));
    // When created, agent has 1 internal variable _id
    EXPECT_EQ(a.getVariablesCount(), 1u);
    a.newVariable<float, 2>(VARIABLE_NAME1);
    EXPECT_EQ(a.getVariablesCount(), 2u);
    a.newVariable<int16_t>(VARIABLE_NAME3);
    EXPECT_EQ(a.getVariablesCount(), 3u);
    a.newVariable<int16_t, 56>(VARIABLE_NAME2);
    EXPECT_EQ(a.getVariablesCount(), 4u);
    // Cannot create variable with same name
    EXPECT_THROW(a.newVariable<int64_t>(VARIABLE_NAME1), exception::InvalidAgentVar);
    // auto newVarArray3 = &AgentDescription::newVariable<int64_t, 1>;  // Use function ptr, can't do more than 1 template arg inside macro
    // EXPECT_THROW((a.*newVarArray3)(VARIABLE_NAME1, {}), exception::InvalidAgentVar);
    EXPECT_THROW(a.newVariable<int64_t>(VARIABLE_NAME1, 0), exception::InvalidAgentVar);
    // Cannot create array of length 0 (disabled, blocked at compilation with static_assert)
    // auto newVarArray0 = &AgentDescription::newVariable<int64_t, 0>;  // Use function ptr, can't do more than 1 template arg inside macro
    // EXPECT_THROW((a.*newVarArray0)(VARIABLE_NAME4), exception::InvalidAgentVar);
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
#ifdef USE_GLM
    // Can create variable array with GLM types
    a.newVariable<glm::vec3, 5>("vec3_5");
    a.newVariable<glm::uvec4, 2>("uvec4_2");
    EXPECT_EQ(a.getVariablesCount(), 6u);
    EXPECT_EQ(5 * 3u, a.getVariableLength("vec3_5"));
    EXPECT_EQ(2 * 4u, a.getVariableLength("uvec4_2"));
    EXPECT_EQ(sizeof(float), a.getVariableSize("vec3_5"));
    EXPECT_EQ(sizeof(unsigned int), a.getVariableSize("uvec4_2"));
    EXPECT_EQ(std::type_index(typeid(float)), a.getVariableType("vec3_5"));
    EXPECT_EQ(std::type_index(typeid(unsigned int)), a.getVariableType("uvec4_2"));
#endif
}
TEST(AgentDescriptionTest, states) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
    EXPECT_FALSE(a.hasState(STATE_NAME1));
    EXPECT_FALSE(a.hasState(STATE_NAME2));
    EXPECT_EQ(a.getStatesCount(), 1u);  // Initially just default state
    a.newState(STATE_NAME1);
    EXPECT_EQ(a.getStatesCount(), 1u);  // Remains 1 state after first set
    a.newState(STATE_NAME2);
    EXPECT_EQ(a.getStatesCount(), 2u);
    // Cannot create state with same name
    EXPECT_THROW(a.newState(STATE_NAME1), exception::InvalidStateName);
    EXPECT_THROW(a.newState(STATE_NAME2), exception::InvalidStateName);
    // States have the right name
    EXPECT_TRUE(a.hasState(STATE_NAME1));
    EXPECT_TRUE(a.hasState(STATE_NAME2));
}
TEST(AgentDescriptionTest, initial_state1) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
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
    AgentDescription a = m.newAgent(AGENT_NAME1);
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
    AgentDescription a = m.newAgent(AGENT_NAME1);
    AgentDescription b = m.newAgent(AGENT_NAME2);
    EXPECT_EQ(a.getAgentOutputsCount(), 0u);
    AgentFunctionDescription f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
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
TEST(AgentDescriptionTest, reserved_name) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
    EXPECT_THROW(a.newVariable<int>("_"), exception::ReservedName);
    EXPECT_THROW(a.newVariable<int>("name"), exception::ReservedName);
    EXPECT_THROW(a.newVariable<int>("state"), exception::ReservedName);
    EXPECT_THROW(a.newVariable<int>("nAme"), exception::ReservedName);
    EXPECT_THROW(a.newVariable<int>("sTate"), exception::ReservedName);
    auto array_version = &AgentDescription::newVariable<int, 3>;
    EXPECT_THROW((a.*array_version)("_", {}), exception::ReservedName);
    EXPECT_THROW((a.*array_version)("name", {}), exception::ReservedName);
    EXPECT_THROW((a.*array_version)("state", {}), exception::ReservedName);
    EXPECT_THROW((a.*array_version)("nAme", {}), exception::ReservedName);
    EXPECT_THROW((a.*array_version)("sTate", {}), exception::ReservedName);
}
const char* rtc_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_filefunc, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") + 1);
    return flamegpu::ALIVE;
}
)###";
TEST(AgentDescriptionTest, rtc_function_from_file) {
    const std::string test_file_name = "test_rtcfunc_file";
    // Create RTC function inside file
    std::ofstream out(test_file_name);
    ASSERT_TRUE(out.is_open());
    out << std::string(rtc_agent_func);
    out.close();
    // Add it to a model
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
    a.newVariable<int>("x");
    EXPECT_NO_THROW(a.newRTCFunctionFile("rtc_test_filefunc", test_file_name));
    m.newLayer().addAgentFunction(AGENT_NAME1, "rtc_test_filefunc");
    // Create and step the model without error
    AgentVector pop(a, 10);
    for (unsigned int i = 0; i < pop.size(); ++i)
      pop[i].setVariable<int>("x", static_cast<int>(i));
    CUDASimulation sim(m);
    sim.setPopulationData(pop);
    EXPECT_NO_THROW(sim.step());
    // Check results are as expected
    AgentVector pop_out(a, 10);
    sim.getPopulationData(pop_out);
    EXPECT_EQ(pop.size(), pop_out.size());
    for (unsigned int i = 0; i < pop_out.size(); ++i) {
        EXPECT_EQ(pop_out[i].getVariable<int>("x"), static_cast<int>(i + 1));
    }
    // Cleanup the file we created
    ASSERT_EQ(::remove(test_file_name.c_str()), 0);
}
TEST(AgentDescriptionTest, rtc_function_from_file_missing) {
    const std::string test_file_name = "test_rtcfunc_file2";
    // Add it to a model
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME1);
    a.newVariable<int>("x");
    EXPECT_THROW(a.newRTCFunctionFile("test_rtcfunc_file2", test_file_name), exception::InvalidFilePath);
}

}  // namespace test_agent
}  // namespace flamegpu
