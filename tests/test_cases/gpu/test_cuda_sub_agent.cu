#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

/***
 * Things to test on SubAgents
 * Simple:
 *  - Data can be passed into submodel and back out with only expected changes taking place
 * Agent death:
 *  - Layer before SubModel
 *  - In SubModel
 * Agent birth:
 *  - Layer before SubModel
 *  - In SubModel
 * AgentFunction Condition:
 *  - Layer before SubModel
 *  - In SubModel
 * Agent State Transitions
 *  - Layer before SubModel
 *    - Unmapped -> Unmapped
 *    - Mapped -> Mapped
 *    - Mapped -> Unmapped (Not currently supported)
 *    - Unmapped -> Mapped (Not currently supported)
 *    - (Due to a branch in the handling, these should be repeated with a full list transition to an empty state, and otherwise)
 *  - In SubModel
 *    - See above, although this is likely less error prone
 *    - Probably more of a case, to test this with a submodel of the submodel
 */
namespace test_cuda_sub_agent {
    const unsigned int AGENT_COUNT = 100;
    const char *SUB_MODEL_NAME = "SubModel";
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *AGENT_VAR1_NAME = "AVar1";
    const char *AGENT_VAR2_NAME = "AVar2";
    const char *AGENT_VAR_i = "i";

FLAMEGPU_AGENT_FUNCTION(AddOne, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AddTen, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + 10);
    const unsigned int v2 = FLAMEGPU->getVariable<unsigned int>("AVar2");
    FLAMEGPU->setVariable<unsigned int>("AVar2", v2 - 10);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(KillEven, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    FLAMEGPU->setVariable<unsigned int>("i", v * 3);
    if (FLAMEGPU->getVariable<unsigned int>("AVar2") == UINT_MAX) {
        // First iteration
        if (v % 4 == 0)
            return DEAD;
    } else {
        // Second iteration
        if (v % 2 == 0)
            return DEAD;
    }
    return ALIVE;
}
FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}
TEST(TestCUDASubAgent, Simple) {
    // Tests whether a sub model is capable of changing an agents variable
    // Agents in same named state, with matching variables
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newFunction("", AddOne);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define bModel
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newFunction("", AddTen);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        // Vars all default init
    }
    // Init Model
    CUDAAgentModel c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop);
    // Run Model
    c.step();
    // Check result
    // Mapped var = init + af + submodel af + af
    const unsigned int mapped_result = 1 + 10 + 1 + 10;
    // Unmapped var = init + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 10 - 10;
    c.getPopulationData(pop);
    for (int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), mapped_result);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result);
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af + submodel af + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af
    const unsigned int unmapped_result2 = unmapped_result - 10 - 10;
    c.getPopulationData(pop);
    for (int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), mapped_result2);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2);
    }
}
TEST(TestCUDASubAgent, AgentDeath_BeforeSubModel) {
    // Tests whether a sub model is capable of changing an agents variable
    // Agents in same named state, with matching variables
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newFunction("", AddOne);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define bModel
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("1", KillEven).setAllowAgentDeath(true);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(KillEven);
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        // Other vars all default init
    }
    // Init Model
    CUDAAgentModel c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop);
    // Run Model
    c.step();
    // Check result
    // Mapped var = init + af + submodel af + af
    const unsigned int mapped_result = + 10 + 1 + 10;
    // Unmapped var = init + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 10 - 10;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT*0.75));
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 3, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/3;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 4, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result);
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af + submodel af + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af
    const unsigned int unmapped_result2 = unmapped_result - 10 - 10;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), AGENT_COUNT/2);
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/9;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 2, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2);
    }
}

TEST(TestCUDASubAgent, AgentDeath_InSubModel) { }
};  // namespace test_cuda_sub_agent
