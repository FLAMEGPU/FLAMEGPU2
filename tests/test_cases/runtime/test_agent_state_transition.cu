/**
* Tests of Agent State Transitions
*
* Tests cover:
* > src: 0, dest: 10 (TODO)
* > src: 10, dest: 10 (TODO)
* > src: 10, dest: 0
*/

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


namespace test_agent_state_transitions {
    const unsigned int AGENT_COUNT = 10;
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *FUNCTION_NAME1 = "Function1";
    const char *FUNCTION_NAME2 = "Function2";
    const char *START_STATE = "Start";
    const char *END_STATE = "End";
    const char *END_STATE2 = "End2";
    const char *LAYER_NAME1 = "Layer1";
    const char *LAYER_NAME2 = "Layer2";
FLAMEGPU_AGENT_FUNCTION(AgentGood, MsgNone, MsgNone) {
    FLAMEGPU->setVariable("x", 11);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentBad, MsgNone, MsgNone) {
    FLAMEGPU->setVariable("x", 13);
    return ALIVE;
}
TEST(TestAgentStateTransitions, DISABLED_Src_0_Dest_10) {
    printf("Todo: This test requires a way to create agents on the host (or maybe agent function conditions).\n");
}
TEST(TestAgentStateTransitions, DISABLED_Src_10_Dest_10) {
    printf("Todo: This test requires a way to create agents on the host (or maybe agent function conditions).\n");
}
TEST(TestAgentStateTransitions, Src_10_Dest_0) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newState(START_STATE);
    a.newState(END_STATE);
    a.newState(END_STATE2);
    a.setInitialState(START_STATE);
    a.newVariable<int>("x");
    AgentFunctionDescription &af1 = a.newFunction(FUNCTION_NAME1, AgentGood);
    af1.setInitialState(START_STATE);
    af1.setEndState(END_STATE);
    AgentFunctionDescription &af2 = a.newFunction(FUNCTION_NAME2, AgentBad);
    af2.setInitialState(END_STATE);
    af2.setEndState(END_STATE2);
    LayerDescription &lo1 = m.newLayer(LAYER_NAME1);
    LayerDescription &lo2 = m.newLayer(LAYER_NAME2);
    lo1.addAgentFunction(af2);
    lo2.addAgentFunction(af1);
    AgentPopulation pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance(START_STATE);
        ai.setVariable<int>("x", 12);
    }
    CUDAAgentModel c(m);
    c.setPopulationData(pop);
    // Step 1, all agents go from Start->End state, and value become 11
    c.step();
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(START_STATE), 0u);
    EXPECT_EQ(pop.getCurrentListSize(END_STATE), AGENT_COUNT);
    EXPECT_EQ(pop.getCurrentListSize(END_STATE2), 0u);
    for (unsigned int i = 0; i < pop.getCurrentListSize(END_STATE); ++i) {
        AgentInstance ai = pop.getInstanceAt(i, END_STATE);
        ASSERT_EQ(ai.getVariable<int>("x"), 11);
    }
    // Step 2, all agents go from End->End2 state, and value become 13
    c.step();
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(START_STATE), 0u);
    EXPECT_EQ(pop.getCurrentListSize(END_STATE), 0u);
    EXPECT_EQ(pop.getCurrentListSize(END_STATE2), AGENT_COUNT);
    for (unsigned int i = 0; i < pop.getCurrentListSize(END_STATE2); ++i) {
        AgentInstance ai = pop.getInstanceAt(i, END_STATE2);
        ASSERT_EQ(ai.getVariable<int>("x"), 13);
    }
}
}  // namespace test_agent_state_transitions
