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
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(NullFn2, MsgNone, MsgNone) {
        FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") - 1);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION_CONDITION(Condition1) {
        return FLAMEGPU->getVariable<int>("x") == 1;
    }
    FLAMEGPU_AGENT_FUNCTION_CONDITION(Condition2) {
        return FLAMEGPU->getVariable<int>("x") != 1;
    }
    TEST(TestAgentFunctionConditions, SplitAgents) {
        ModelDescription m(MODEL_NAME);
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<int>("x");
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
        }
        // Check val of agents in STATE3 state
        for (unsigned int j = 0; j < pop.getCurrentListSize(STATE3); ++j) {
            AgentInstance ai = pop.getInstanceAt(j, STATE3);
            EXPECT_EQ(ai.getVariable<int>("x"), -1);
        }
    }


}  // namespace test_agent_function_conditions
