/**
* Tests of Agent State Transitions
*
* Tests cover:
* > src: 0, dest: 10
* > src: 10, dest: 0
* > src: 10, dest: 10 (This complicated test also serves to demonstrate that agent function conditions work)
*/

#include <array>

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
FLAMEGPU_AGENT_FUNCTION(AgentDecrement, MsgNone, MsgNone) {
    unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    FLAMEGPU->setVariable("x", x == 0 ? 0 : x - 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentNull, MsgNone, MsgNone) {
    FLAMEGPU->setVariable("x", UINT_MAX);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION_CONDITION(Zero_X) {
    // Agent's only transition when counter reaches zero
    return FLAMEGPU->getVariable<unsigned int>("x") == 0;
}
TEST(TestAgentStateTransitions, Src_0_Dest_10) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newState(START_STATE);
    a.newState(END_STATE);
    a.setInitialState(START_STATE);
    a.newVariable<int>("x");
    AgentFunctionDescription &af1 = a.newFunction(FUNCTION_NAME1, AgentGood);
    af1.setInitialState(START_STATE);
    af1.setEndState(END_STATE);
    LayerDescription &lo1 = m.newLayer(LAYER_NAME1);
    lo1.addAgentFunction(af1);
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
    for (unsigned int i = 0; i < pop.getCurrentListSize(END_STATE); ++i) {
        AgentInstance ai = pop.getInstanceAt(i, END_STATE);
        ASSERT_EQ(ai.getVariable<int>("x"), 11);
    }
    // Step 2, no agents in start state, nothing changes
    c.step();
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(START_STATE), 0u);
    EXPECT_EQ(pop.getCurrentListSize(END_STATE), AGENT_COUNT);
    for (unsigned int i = 0; i < pop.getCurrentListSize(END_STATE); ++i) {
        AgentInstance ai = pop.getInstanceAt(i, END_STATE);
        ASSERT_EQ(ai.getVariable<int>("x"), 11);
    }
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
TEST(TestAgentStateTransitions, Src_10_Dest_10) {
    // Init agents with two vars x and y
    // 3 round, 10 agents per round, with x and y value all set to 1 * round no
    // Each round 10 agents move from start to end state
    // Confirm why values are as expected
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newState(START_STATE);
    a.newState(END_STATE);
    a.setInitialState(START_STATE);
    a.newVariable<unsigned int>("x");
    a.newVariable<unsigned int>("y");
    AgentFunctionDescription &af1 = a.newFunction(FUNCTION_NAME1, AgentDecrement);
    af1.setInitialState(START_STATE);
    af1.setEndState(START_STATE);
    // Does nothing, just serves to demonstrate conditional state transition
    AgentFunctionDescription &af2 = a.newFunction(FUNCTION_NAME2, AgentNull);
    af2.setInitialState(START_STATE);
    af2.setEndState(END_STATE);
    af2.setFunctionCondition(Zero_X);
    LayerDescription &lo1 = m.newLayer(LAYER_NAME1);
    LayerDescription &lo2 = m.newLayer(LAYER_NAME2);
    lo1.addAgentFunction(af1);
    lo2.addAgentFunction(af2);
    // Init pop
    const unsigned int ROUNDS = 3;
    AgentPopulation pop(a, ROUNDS * AGENT_COUNT);
    for (unsigned int i = 0; i < ROUNDS * AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance(START_STATE);
        unsigned int val = 1 + (i % ROUNDS);  // 1, 2, 3, 1, 2, 3 etc
        ai.setVariable<unsigned int>("x", val);
        ai.setVariable<unsigned int>("y", val);
    }
    CUDAAgentModel c(m);
    c.setPopulationData(pop);

    // Step 1, all agents go from Start->End state, and value become 11
    for (unsigned int i = 1; i <= ROUNDS; i++) {
        // printf("Round: %d\n", i);
        std::array<unsigned int, ROUNDS + 1> out;
        memset(out.data(), 0, sizeof(unsigned int) * (ROUNDS + 1));
        c.step();
        c.getPopulationData(pop);
        EXPECT_EQ(pop.getCurrentListSize(START_STATE), (ROUNDS - i) * AGENT_COUNT);
        EXPECT_EQ(pop.getCurrentListSize(END_STATE), i * AGENT_COUNT);
        // Check val of agents in start state
        for (unsigned int j = 0; j < pop.getCurrentListSize(START_STATE); ++j) {
            AgentInstance ai = pop.getInstanceAt(j, START_STATE);
            unsigned int y = ai.getVariable<unsigned int>("y");
            out[y]++;
        }
        // printf("Start hist: [%u, %u, %u, %u]\n", out[0], out[1], out[2], out[3]);
        EXPECT_EQ(out[0], 0u);
        for (unsigned int j = 1+i; j <= ROUNDS; j++) {
            EXPECT_EQ(out[j], AGENT_COUNT);
        }
        for (unsigned int j = ROUNDS - i ; j < 1 + i; j++) {
            EXPECT_EQ(out[j], 0u);
        }
        // Check val of agents in end state
        memset(out.data(), 0, sizeof(unsigned int) * (ROUNDS + 1));
        for (unsigned int j = 0; j < pop.getCurrentListSize(END_STATE); ++j) {
            AgentInstance ai = pop.getInstanceAt(j, END_STATE);
            unsigned int y = ai.getVariable<unsigned int>("y");
            out[y]++;
        }
        // printf("End hist: [%u, %u, %u, %u]\n", out[0], out[1], out[2], out[3]);
        for (unsigned int j = 1; j < i + 1; j++) {
            EXPECT_EQ(out[j], AGENT_COUNT);
        }
        for (unsigned int j = i + 1; j <= ROUNDS; j++) {
            EXPECT_EQ(out[j], 0u);
        }
    }
}

}  // namespace test_agent_state_transitions
