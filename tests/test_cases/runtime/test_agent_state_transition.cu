/**
* Tests of Agent State Transitions
*
* Tests cover:
* > src: 0, dest: 10
* > src: 10, dest: 0
* > src: 10, dest: 10 (This complicated test also serves to demonstrate that agent function conditions work)
*/

#include <array>

#include "flamegpu/flamegpu.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace flamegpu {


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
    FLAMEGPU->setVariable<int, 4>("y", 0, 23);
    FLAMEGPU->setVariable<int, 4>("y", 1, 24);
    FLAMEGPU->setVariable<int, 4>("y", 2, 25);
    FLAMEGPU->setVariable<int, 4>("y", 3, 26);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentBad, MsgNone, MsgNone) {
    FLAMEGPU->setVariable("x", 13);
    FLAMEGPU->setVariable<int, 4>("y", 0, 3);
    FLAMEGPU->setVariable<int, 4>("y", 1, 4);
    FLAMEGPU->setVariable<int, 4>("y", 2, 5);
    FLAMEGPU->setVariable<int, 4>("y", 3, 6);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentDecrement, MsgNone, MsgNone) {
    unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    FLAMEGPU->setVariable("x", x == 0 ? 0 : x - 1);
    FLAMEGPU->setVariable<int, 4>("z", 0, 23);
    FLAMEGPU->setVariable<int, 4>("z", 1, 24);
    FLAMEGPU->setVariable<int, 4>("z", 2, 25);
    FLAMEGPU->setVariable<int, 4>("z", 3, 26);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentNull, MsgNone, MsgNone) {
    FLAMEGPU->setVariable("x", UINT_MAX);
    FLAMEGPU->setVariable<int, 4>("z", 0, 3);
    FLAMEGPU->setVariable<int, 4>("z", 1, 4);
    FLAMEGPU->setVariable<int, 4>("z", 2, 5);
    FLAMEGPU->setVariable<int, 4>("z", 3, 6);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION_CONDITION(Zero_X) {
    // Agent's only transition when counter reaches zero
    return FLAMEGPU->getVariable<unsigned int>("x") == 0;
}
TEST(TestAgentStateTransitions, Src_0_Dest_10) {
    const std::array<int, 4> ARRAY_REFERENCE = { 13, 14, 15, 16 };
    const std::array<int, 4> ARRAY_REFERENCE2 = { 23, 24, 25, 26 };
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newState(START_STATE);
    a.newState(END_STATE);
    a.setInitialState(START_STATE);
    a.newVariable<int>("x");
    a.newVariable<int, 4>("y");
    AgentFunctionDescription &af1 = a.newFunction(FUNCTION_NAME1, AgentGood);
    af1.setInitialState(START_STATE);
    af1.setEndState(END_STATE);
    LayerDescription &lo1 = m.newLayer(LAYER_NAME1);
    lo1.addAgentFunction(af1);
    AgentVector pop(a, AGENT_COUNT);
    for (AgentVector::Agent ai : pop) {
        ai.setVariable<int>("x", 12);
        ai.setVariable<int, 4>("y", ARRAY_REFERENCE);
    }
    CUDASimulation c(m);
    c.setPopulationData(pop, START_STATE);
    // Step 1, all agents go from Start->End state, and value become 11
    c.step();

    AgentVector pop_START_STATE(a);
    AgentVector pop_END_STATE(a);
    c.getPopulationData(pop_START_STATE, START_STATE);
    c.getPopulationData(pop_END_STATE, END_STATE);
    EXPECT_EQ(pop_START_STATE.size(), 0u);
    EXPECT_EQ(pop_END_STATE.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_END_STATE) {
        ASSERT_EQ(ai.getVariable<int>("x"), 11);
        auto test = ai.getVariable<int, 4>("y");
        ASSERT_EQ(test, ARRAY_REFERENCE2);
    }
    // Step 2, no agents in start state, nothing changes
    c.step();
    c.getPopulationData(pop_START_STATE, START_STATE);
    c.getPopulationData(pop_END_STATE, END_STATE);
    EXPECT_EQ(pop_START_STATE.size(), 0u);
    EXPECT_EQ(pop_END_STATE.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_END_STATE) {
        ASSERT_EQ(ai.getVariable<int>("x"), 11);
        auto test = ai.getVariable<int, 4>("y");
        ASSERT_EQ(test, ARRAY_REFERENCE2);
    }
}
TEST(TestAgentStateTransitions, Src_10_Dest_0) {
    const std::array<int, 4> ARRAY_REFERENCE = { 13, 14, 15, 16 };
    const std::array<int, 4> ARRAY_REFERENCE2 = { 23, 24, 25, 26 };
    const std::array<int, 4> ARRAY_REFERENCE3 = { 3, 4, 5, 6 };
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newState(START_STATE);
    a.newState(END_STATE);
    a.newState(END_STATE2);
    a.setInitialState(START_STATE);
    a.newVariable<int>("x");
    a.newVariable<int, 4>("y");
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
    AgentVector pop(a, AGENT_COUNT);
    for (AgentVector::Agent ai : pop) {
        ai.setVariable<int>("x", 12);
        ai.setVariable<int, 4>("y", ARRAY_REFERENCE);
    }
    CUDASimulation c(m);
    c.setPopulationData(pop, START_STATE);
    // Step 1, all agents go from Start->End state, and value become 11
    c.step();
    AgentVector pop_START_STATE(a);
    AgentVector pop_END_STATE(a);
    AgentVector pop_END_STATE2(a);
    c.getPopulationData(pop_START_STATE, START_STATE);
    c.getPopulationData(pop_END_STATE, END_STATE);
    c.getPopulationData(pop_END_STATE2, END_STATE2);
    EXPECT_EQ(pop_START_STATE.size(), 0u);
    EXPECT_EQ(pop_END_STATE.size(), AGENT_COUNT);
    EXPECT_EQ(pop_END_STATE2.size(), 0u);
    for (AgentVector::Agent ai : pop_END_STATE) {
        ASSERT_EQ(ai.getVariable<int>("x"), 11);
        auto test = ai.getVariable<int, 4>("y");
        ASSERT_EQ(test, ARRAY_REFERENCE2);
    }
    // Step 2, all agents go from End->End2 state, and value become 13
    c.step();
    c.getPopulationData(pop_START_STATE, START_STATE);
    c.getPopulationData(pop_END_STATE, END_STATE);
    c.getPopulationData(pop_END_STATE2, END_STATE2);
    EXPECT_EQ(pop_START_STATE.size(), 0u);
    EXPECT_EQ(pop_END_STATE.size(), 0u);
    EXPECT_EQ(pop_END_STATE2.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_END_STATE2) {
        ASSERT_EQ(ai.getVariable<int>("x"), 13);
        auto test = ai.getVariable<int, 4>("y");
        ASSERT_EQ(test, ARRAY_REFERENCE3);
    }
}
TEST(TestAgentStateTransitions, Src_10_Dest_10) {
    const std::array<int, 4> ARRAY_REFERENCE = { 13, 14, 15, 16 };
    const std::array<int, 4> ARRAY_REFERENCE2 = { 23, 24, 25, 26 };
    const std::array<int, 4> ARRAY_REFERENCE3 = { 3, 4, 5, 6 };
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
    a.newVariable<int, 4>("z");
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
    AgentVector pop(a, ROUNDS * AGENT_COUNT);
    for (unsigned int i = 0; i < ROUNDS * AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        unsigned int val = 1 + (i % ROUNDS);  // 1, 2, 3, 1, 2, 3 etc
        ai.setVariable<unsigned int>("x", val);
        ai.setVariable<unsigned int>("y", val);
        ai.setVariable<int, 4>("z", ARRAY_REFERENCE);
    }
    CUDASimulation c(m);
    c.setPopulationData(pop, START_STATE);

    AgentVector pop_START_STATE(a);
    AgentVector pop_END_STATE(a);
    // Step 1, all agents go from Start->End state, and value become 11
    for (unsigned int i = 1; i <= ROUNDS; i++) {
        // printf("Round: %d\n", i);
        std::array<unsigned int, ROUNDS + 1> out;
        memset(out.data(), 0, sizeof(unsigned int) * (ROUNDS + 1));
        c.step();
        c.getPopulationData(pop_START_STATE, START_STATE);
        c.getPopulationData(pop_END_STATE, END_STATE);
        EXPECT_EQ(pop_START_STATE.size(), (ROUNDS - i) * AGENT_COUNT);
        EXPECT_EQ(pop_END_STATE.size(), i * AGENT_COUNT);
        // Check val of agents in start state
        for (AgentVector::Agent ai : pop_START_STATE) {
            unsigned int y = ai.getVariable<unsigned int>("y");
            out[y]++;
            auto test = ai.getVariable<int, 4>("z");
            ASSERT_EQ(test, ARRAY_REFERENCE2);
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
        for (AgentVector::Agent ai : pop_END_STATE) {
            unsigned int y = ai.getVariable<unsigned int>("y");
            out[y]++;
            auto test = ai.getVariable<int, 4>("z");
            ASSERT_EQ(test, ARRAY_REFERENCE3);
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
}  // namespace flamegpu
