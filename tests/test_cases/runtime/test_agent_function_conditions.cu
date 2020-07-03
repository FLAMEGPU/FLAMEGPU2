/**
* Tests of Agent State Transitions
*
* Tests cover:
* > two functions (in seperate layers) partition agents into two new states
*/

#include <array>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


namespace test_agent_function_conditions {
    const unsigned int AGENT_COUNT = 1000;
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *FUNCTION_NAME1 = "Function1";
    const char *FUNCTION_NAME2 = "Function2";
    const char *STATE1 = "Start";
    const char *STATE2 = "End";
    const char *STATE3 = "End2";
    FLAMEGPU_AGENT_FUNCTION(NullFn1, MsgNone, MsgNone) {
        FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") + 1);
        FLAMEGPU->setVariable<int, 4>("y", 0, 3);
        FLAMEGPU->setVariable<int, 4>("y", 1, 4);
        FLAMEGPU->setVariable<int, 4>("y", 2, 5);
        FLAMEGPU->setVariable<int, 4>("y", 3, 6);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(NullFn2, MsgNone, MsgNone) {
        FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") - 1);
        FLAMEGPU->setVariable<int, 4>("y", 0, 23);
        FLAMEGPU->setVariable<int, 4>("y", 1, 24);
        FLAMEGPU->setVariable<int, 4>("y", 2, 25);
        FLAMEGPU->setVariable<int, 4>("y", 3, 26);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION_CONDITION(Condition1) {
        return FLAMEGPU->getVariable<int>("x") == 1;
    }
    FLAMEGPU_AGENT_FUNCTION_CONDITION(Condition2) {
        return FLAMEGPU->getVariable<int>("x") != 1;
    }
    TEST(TestAgentFunctionConditions, SplitAgents) {
        const std::array<int, 4> ARRAY_REFERENCE = { 13, 14, 15, 16 };
        const std::array<int, 4> ARRAY_REFERENCE2 = { 23, 24, 25, 26 };
        const std::array<int, 4> ARRAY_REFERENCE3 = { 3, 4, 5, 6 };
        ModelDescription m(MODEL_NAME);
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<int>("x");
        a.newVariable<int, 4>("y");
        a.newState(STATE1);
        a.newState(STATE2);
        a.newState(STATE3);
        AgentFunctionDescription &af1 = a.newFunction(FUNCTION_NAME1, NullFn1);
        af1.setInitialState(STATE1);
        af1.setEndState(STATE2);
        af1.setFunctionCondition(Condition1);
        AgentFunctionDescription &af2 = a.newFunction(FUNCTION_NAME2, NullFn2);
        af2.setInitialState(STATE1);
        af2.setEndState(STATE3);
        af2.setFunctionCondition(Condition2);
        LayerDescription &l1 = m.newLayer();
        l1.addAgentFunction(af1);
        LayerDescription &l2 = m.newLayer();
        l2.addAgentFunction(af2);
        AgentPopulation pop(a, AGENT_COUNT * 2);
        for (unsigned int i = 0; i < AGENT_COUNT * 2; ++i) {
            AgentInstance ai = pop.getNextInstance(STATE1);
            unsigned int val = i % 2;  // 0, 1, 0, 1, etc
            ai.setVariable<int>("x", val);
            ai.setVariable<int, 4>("y", ARRAY_REFERENCE);
        }
        CUDAAgentModel c(m);
        c.setPopulationData(pop);
        c.step();
        c.getPopulationData(pop);
        EXPECT_EQ(pop.getCurrentListSize(STATE1), 0u);
        EXPECT_EQ(pop.getCurrentListSize(STATE2), AGENT_COUNT);
        EXPECT_EQ(pop.getCurrentListSize(STATE3), AGENT_COUNT);
        // Check val of agents in STATE2 state
        for (unsigned int j = 0; j < pop.getCurrentListSize(STATE2); ++j) {
            AgentInstance ai = pop.getInstanceAt(j, STATE2);
            EXPECT_EQ(ai.getVariable<int>("x"), 2);
            auto test = ai.getVariable<int, 4>("y");
            ASSERT_EQ(test, ARRAY_REFERENCE3);
        }
        // Check val of agents in STATE3 state
        for (unsigned int j = 0; j < pop.getCurrentListSize(STATE3); ++j) {
            AgentInstance ai = pop.getInstanceAt(j, STATE3);
            EXPECT_EQ(ai.getVariable<int>("x"), -1);
            auto test = ai.getVariable<int, 4>("y");
            ASSERT_EQ(test, ARRAY_REFERENCE2);
        }
    }
    FLAMEGPU_AGENT_FUNCTION_CONDITION(AllFail) {
        return false;
    }
    TEST(TestAgentFunctionConditions, AllDisabled) {
        // Tests for a bug created by #342, fixed by #343
        // If all agents got disabled by an agent function condition, they would not get re-enabled
        // This would lead to an exception later on
        ModelDescription m(MODEL_NAME);
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<int>("x");
        a.newVariable<int, 4>("y");
        AgentFunctionDescription &af1 = a.newFunction(FUNCTION_NAME1, NullFn1);
        af1.setFunctionCondition(AllFail);
        AgentFunctionDescription &af2 = a.newFunction(FUNCTION_NAME2, NullFn2);
        LayerDescription &l1 = m.newLayer();
        l1.addAgentFunction(af1);
        LayerDescription &l2 = m.newLayer();
        l2.addAgentFunction(af2);
        // Create a bunch of empty agents
        AgentPopulation pop(a, AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getNextInstance();
        }
        CUDAAgentModel c(m);
        c.setPopulationData(pop);
        EXPECT_NO_THROW(c.step());
        EXPECT_NO_THROW(c.step());
        EXPECT_NO_THROW(c.step());
    }
}  // namespace test_agent_function_conditions
