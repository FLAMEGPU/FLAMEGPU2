#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

/***
 * Things to test on SubAgents
 * Simple:
 *  - Data can be passed into submodel and back out with only expected changes taking place
 *  - SubModel can have unmapped default vars
 *  - Main model can have unmapped vars
 * Agent death:
 *  - Layer before SubModel
 *  - In SubModel
 *  - In SubModel (nested version)
 * Agent birth:
 *  - Layer before SubModel
 *  - In SubModel
 *  - In SubModel (nested version)
 * AgentFunction Condition:
 *  - Layer before SubModel
 *  - In SubModel
 *  - In SubModel (nested version)
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
    const char *PROXY_SUB_MODEL_NAME = "ProxySubModel";
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *AGENT_VAR1_NAME = "AVar1";
    const char *AGENT_VAR2_NAME = "AVar2";
    const char *SUB_VAR1_NAME = "SubVar1";
    const char *AGENT_VAR_i = "i";
    const char *AGENT_VAR_t = "t";
FLAMEGPU_AGENT_FUNCTION(AddT, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    const unsigned int t = FLAMEGPU->getVariable<unsigned int>("t");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + t);
    FLAMEGPU->setVariable<unsigned int>("t", t + 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AddOne, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    const unsigned int sub_v = FLAMEGPU->getVariable<unsigned int>("SubVar1");
    if (sub_v == 12) {
        // sub_v should always be it's default value 12 if created in submodel, we never change it
        FLAMEGPU->setVariable<unsigned int>("AVar1", v + 1);
    } else if (sub_v == 599) {
        // sub_v Agents created byproxysubmodel or above will have this value, so original agents set this
        FLAMEGPU->setVariable<unsigned int>("AVar1", v + 1);
    } else {
        FLAMEGPU->setVariable<unsigned int>("AVar1", v + 100000);
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AddOne2, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AddTen, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + 10);
    const unsigned int v2 = FLAMEGPU->getVariable<unsigned int>("AVar2");
    FLAMEGPU->setVariable<unsigned int>("AVar2", v2 - 1000);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(KillEven, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    FLAMEGPU->setVariable<unsigned int>("i", v * 3);
    if (FLAMEGPU->getVariable<unsigned int>("AVar2") > UINT_MAX-1000) {
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
FLAMEGPU_AGENT_FUNCTION(BirthEven, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    FLAMEGPU->setVariable<unsigned int>("i", v * 3);
    if (FLAMEGPU->getVariable<unsigned int>("AVar2") > UINT_MAX - 2000) {
        // First iteration
        if (v % 4 == 0) {
            FLAMEGPU->agent_out.setVariable("i", v * 3);
            FLAMEGPU->agent_out.setVariable("AVar2", 4000 + v);
        }
    } else if (FLAMEGPU->getVariable<unsigned int>("AVar2") > UINT_MAX - 4000) {
        // Second iteration
        if ((v / 3) % 4 == 0) {
            FLAMEGPU->agent_out.setVariable("i", v * 3);
            FLAMEGPU->agent_out.setVariable("AVar2", 4000 + v);
        }
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION_CONDITION(AllowEven) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    // First iteration
    if (v % 4 == 0) {
        return true;
    }
    return false;
}
FLAMEGPU_AGENT_FUNCTION(UpdateId100, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    FLAMEGPU->setVariable<unsigned int>("i", v + 100);
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
        a.newVariable<unsigned int>(AGENT_VAR_t, 1);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        a.newFunction("1", AddT);
        a.newFunction("", AddOne);
        sm.newLayer().addAgentFunction(AddT);
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
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
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
    const unsigned int mapped_result = 1 + 10 + 1 + 1 + 10;
    // Unmapped var = init + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), mapped_result);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result);
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af + submodel af + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 2 + 10;  // NOTE: THIS SHOULD BE mapped_reuslt + 1 + 10, however unmapped vars aren't reset between submodel executions
    // Unmapped var = unmapped_result + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), mapped_result2);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2);
    }
}
TEST(TestCUDASubAgent, AgentDeath_BeforeSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
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
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT*0.75));    // if AGENT_COUNT > 1000 this test will fail
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 3, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/3;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 4, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result - __i);
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af + submodel af + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), AGENT_COUNT/2);
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/9;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 2, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2 - __i);
    }
}
TEST(TestCUDASubAgent, AgentDeath_InSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(AGENT_VAR2_NAME, 0);
        a.newVariable<unsigned int>(AGENT_VAR_i, 0);
        a.newFunction("1", KillEven).setAllowAgentDeath(true);
        sm.newLayer().addAgentFunction(KillEven);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define bModel
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("", AddOne2);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddOne2);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    const unsigned int mapped_result = 1 + 10;
    // Unmapped var = init + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT*0.75));    // if AGENT_COUNT > 1000 this test will fail
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 3, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/3;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 4, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result - __i);
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af + submodel af + af
    const unsigned int mapped_result2 = mapped_result + 1 + 10;
    // Unmapped var = unmapped_result + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), AGENT_COUNT/2);
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/9;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 2, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2 - __i);
    }
}
TEST(TestCUDASubAgent, AgentDeath_InNestedSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(AGENT_VAR2_NAME, 0);
        a.newVariable<unsigned int>(AGENT_VAR_i, 0);
        a.newFunction("1", KillEven).setAllowAgentDeath(true);
        sm.newLayer().addAgentFunction(KillEven);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription psm(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = psm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(AGENT_VAR2_NAME, 0);
        a.newVariable<unsigned int>(AGENT_VAR_i, 0);
        auto &smd = psm.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        psm.newLayer().addSubModel("sub");
        psm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define bModel
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("", AddOne2);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("proxysub", psm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddOne2);
        m.newLayer().addSubModel("proxysub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    const unsigned int mapped_result = 1 + 10;
    // Unmapped var = init + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT*0.75));    // if AGENT_COUNT > 1000 this test will fail
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 3, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/3;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 4, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result - __i);
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af + submodel af + af
    const unsigned int mapped_result2 = mapped_result + 1 + 10;
    // Unmapped var = unmapped_result + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), AGENT_COUNT/2);
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/9;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 2, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2 - __i);
    }
}

TEST(TestCUDASubAgent, AgentBirth_BeforeSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        a.newFunction("", AddOne);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("1", BirthEven).setAgentOutput(ma);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addAgentFunction(BirthEven);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    // Mapped var = init + af +  af + submodel + af
    const unsigned int mapped_result = +10 + 1 + 10;
    // Unmapped var = init + af + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT*1.25));    // if AGENT_COUNT != 1000 this test may fail
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 3, 0u);  // Var divides cleanly by 3
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_avar2 < 4000) {
            // Agent was born
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
            // Agent var name 1 was init to default 1, submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 12u);
            EXPECT_EQ(_avar2 - 3000, __i);  // AGENT_VAR2_NAME is the expected value 4000 + i - 1000
        } else {
            // Agent is from init
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
        }
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    // Init AGENT_COUNT + 25% 1st birth + 25% 2nd birth (25% of init only)
    EXPECT_EQ(pop.getCurrentListSize(), AGENT_COUNT * 1.25 + AGENT_COUNT * 0.25);
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 9
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_avar2 < 2000) {
            // Agent was born step 1
            const unsigned int __i = _i / 9;  // Calculate original value of AGENT_VAR_i
            // Agent var name 1 was 22 previous step, +10 from addten() submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 33u);
            EXPECT_EQ(_avar2 - 1000, __i);  // AGENT_VAR2_NAME is the expected value (3000 + i) - 1000 - 1000
        } else if (_avar2 < 4000) {
            // Agent was born step 2
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
            // Agent var name 1 was default 1, submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 12u);
            EXPECT_EQ(_avar2 - 3000, __i);  // AGENT_VAR2_NAME is the expected value 4000 + i - 1000
        } else {
            // Agent is from init
            const unsigned int __i = _i / 9;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2 - __i);
        }
    }
}
TEST(TestCUDASubAgent, AgentBirth_InSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        a.newVariable<unsigned int>(AGENT_VAR2_NAME, 12);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        a.newVariable<unsigned int>(AGENT_VAR_i);
        a.newFunction("1", BirthEven).setAgentOutput(a);
        a.newFunction("2", AddOne);
        sm.newLayer().addAgentFunction(BirthEven);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>("default_main_agent_var", 25);
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 10000);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    // Mapped var = init + af +  af + submodel + af
    const unsigned int mapped_result = +10 + 1 + 10;
    // Unmapped var = init + af + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT*1.25));    // if AGENT_COUNT != 1000 this test may fail
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<unsigned int>("default_main_agent_var"), 25u);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 3, 0u);  // Var divides cleanly by 3
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_avar2 < 4000) {
            // Agent was born
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
                                              // Agent var name 1 was init to default 1, submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 12u);
            EXPECT_EQ(_avar2 - 3000, __i);  // AGENT_VAR2_NAME is the expected value 4000 + i - 1000
        } else {
            // Agent is from init
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
        }
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    // Init AGENT_COUNT + 25% 1st birth + 25% 2nd birth (25% of init only)
    EXPECT_EQ(pop.getCurrentListSize(), AGENT_COUNT * 1.25 + AGENT_COUNT * 0.25);
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 9
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_avar2 < 2000) {
            // Agent was born step 1
            const unsigned int __i = _i / 9;  // Calculate original value of AGENT_VAR_i
                                              // Agent var name 1 was 22 previous step, +10 from addten() submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 33u);
            EXPECT_EQ(_avar2 - 1000, __i);  // AGENT_VAR2_NAME is the expected value (3000 + i) - 1000 - 1000
        } else if (_avar2 < 4000) {
            // Agent was born step 2
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
                                              // Agent var name 1 was default 1, submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 12u);
            EXPECT_EQ(_avar2 - 3000, __i);  // AGENT_VAR2_NAME is the expected value 4000 + i - 1000
        } else {
            // Agent is from init
            const unsigned int __i = _i / 9;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2 - __i);
        }
    }
}
TEST(TestCUDASubAgent, AgentBirth_InNestedSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        a.newVariable<unsigned int>(AGENT_VAR2_NAME, 12);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        a.newVariable<unsigned int>(AGENT_VAR_i);
        a.newFunction("1", BirthEven).setAgentOutput(a);
        a.newFunction("2", AddOne);
        sm.newLayer().addAgentFunction(BirthEven);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription psm(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = psm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 599);
        a.newVariable<unsigned int>(AGENT_VAR2_NAME, 599);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 599);
        a.newVariable<unsigned int>(AGENT_VAR_i);
        auto &smd = psm.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        psm.newLayer().addSubModel("sub");
        psm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>("default_main_agent_var", 25);
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 10000);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("proxysub", psm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("proxysub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    // Mapped var = init + af +  af + submodel + af
    const unsigned int mapped_result = +10 + 1 + 10;
    // Unmapped var = init + af + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT*1.25));    // if AGENT_COUNT != 1000 this test may fail
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<unsigned int>("default_main_agent_var"), 25u);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 3, 0u);  // Var divides cleanly by 3
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_avar2 < 4000) {
            // Agent was born
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
                                              // Agent var name 1 was init to default 1, submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 12u);
            EXPECT_EQ(_avar2 - 3000, __i);  // AGENT_VAR2_NAME is the expected value 4000 + i - 1000
        } else {
            // Agent is from init
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
        }
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    // Init AGENT_COUNT + 25% 1st birth + 25% 2nd birth (25% of init only)
    EXPECT_EQ(pop.getCurrentListSize(), AGENT_COUNT * 1.25 + AGENT_COUNT * 0.25);
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 9
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_avar2 < 2000) {
            // Agent was born step 1
            const unsigned int __i = _i / 9;  // Calculate original value of AGENT_VAR_i
                                              // Agent var name 1 was 22 previous step, +10 from addten() submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 33u);
            EXPECT_EQ(_avar2 - 1000, __i);  // AGENT_VAR2_NAME is the expected value (3000 + i) - 1000 - 1000
        } else if (_avar2 < 4000) {
            // Agent was born step 2
            const unsigned int __i = _i / 3;  // Calculate original value of AGENT_VAR_i
                                              // Agent var name 1 was default 1, submodel added +1, +10 from addten()
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 12u);
            EXPECT_EQ(_avar2 - 3000, __i);  // AGENT_VAR2_NAME is the expected value 4000 + i - 1000
        } else {
            // Agent is from init
            const unsigned int __i = _i / 9;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2 - __i);
        }
    }
}

TEST(TestCUDASubAgent, AgentFunctionCondition_BeforeSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        a.newFunction("", AddOne);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("1", UpdateId100).setFunctionCondition(AllowEven);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addAgentFunction(UpdateId100);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    // Mapped var = init + af +  af + submodel + af
    const unsigned int mapped_result = +10 + 1 + 10;
    // Unmapped var = init + af + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT));
    unsigned int pass_count = 0;
    unsigned int fail_count = 0;
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i >= AGENT_COUNT) {
            // Agent passed condition
            const unsigned int __i = _i - 100;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
            pass_count++;
        } else {
            // Agent failed condition
            const unsigned int __i = _i;
            EXPECT_NE(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
            fail_count++;
        }
    }
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.75), fail_count);
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT));
    pass_count = 0;
    fail_count = 0;
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i >= AGENT_COUNT + 100) {
            // Agent passed condition (same agents pass both times)
            const unsigned int __i = _i - 200;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(_avar2, unmapped_result2 - __i);
            pass_count++;
        } else {
            // Agent failed condition
            const unsigned int __i = _i;
            EXPECT_NE(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(_avar2, unmapped_result2 - __i);
            fail_count++;
        }
    }
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.75), fail_count);
}
TEST(TestCUDASubAgent, AgentFunctionCondition_InSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR_i);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        a.newFunction("1", UpdateId100).setFunctionCondition(AllowEven);
        a.newFunction("", AddOne);
        sm.newLayer().addAgentFunction(UpdateId100);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    // Mapped var = init + af +  af + submodel + af
    const unsigned int mapped_result = +10 + 1 + 10;
    // Unmapped var = init + af + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT));
    unsigned int pass_count = 0;
    unsigned int fail_count = 0;
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i >= AGENT_COUNT) {
            // Agent passed condition
            const unsigned int __i = _i - 100;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
            pass_count++;
        } else {
            // Agent failed condition
            const unsigned int __i = _i;
            EXPECT_NE(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
            fail_count++;
        }
    }
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.75), fail_count);
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT));
    pass_count = 0;
    fail_count = 0;
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i >= AGENT_COUNT + 100) {
            // Agent passed condition (same agents pass both times)
            const unsigned int __i = _i - 200;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(_avar2, unmapped_result2 - __i);
            pass_count++;
        } else {
            // Agent failed condition
            const unsigned int __i = _i;
            EXPECT_NE(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(_avar2, unmapped_result2 - __i);
            fail_count++;
        }
    }
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.75), fail_count);
}
TEST(TestCUDASubAgent, AgentFunctionCondition_InNestedSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR_i);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        a.newFunction("1", UpdateId100).setFunctionCondition(AllowEven);
        a.newFunction("", AddOne);
        sm.newLayer().addAgentFunction(UpdateId100);
        sm.newLayer().addAgentFunction(AddOne);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription psm(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = psm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR_i);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 12);
        auto &smd = psm.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        psm.newLayer().addSubModel("sub");
        psm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max());
        ma.newVariable<unsigned int>(AGENT_VAR_i);
        ma.newFunction("2", AddTen);
        auto &smd = m.newSubModel("proxysub", psm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("proxysub");
        m.newLayer().addAgentFunction(AddTen);
    }
    // Init Agents
    AgentPopulation pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
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
    // Mapped var = init + af +  af + submodel + af
    const unsigned int mapped_result = +10 + 1 + 10;
    // Unmapped var = init + af + af + af
    const unsigned int unmapped_result = std::numeric_limits<unsigned int>::max() - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT));
    unsigned int pass_count = 0;
    unsigned int fail_count = 0;
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i >= AGENT_COUNT) {
            // Agent passed condition
            const unsigned int __i = _i - 100;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
            pass_count++;
        } else {
            // Agent failed condition
            const unsigned int __i = _i;
            EXPECT_NE(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result);
            EXPECT_EQ(_avar2, unmapped_result - __i);
            fail_count++;
        }
    }
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.75), fail_count);
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.getCurrentListSize(), static_cast<unsigned int>(AGENT_COUNT));
    pass_count = 0;
    fail_count = 0;
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        const unsigned int _avar2 = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i >= AGENT_COUNT + 100) {
            // Agent passed condition (same agents pass both times)
            const unsigned int __i = _i - 200;  // Calculate original value of AGENT_VAR_i
            EXPECT_EQ(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(_avar2, unmapped_result2 - __i);
            pass_count++;
        } else {
            // Agent failed condition
            const unsigned int __i = _i;
            EXPECT_NE(__i % 4, 0u);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
            EXPECT_EQ(_avar2, unmapped_result2 - __i);
            fail_count++;
        }
    }
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.getCurrentListSize() * 0.75), fail_count);
}
};  // namespace test_cuda_sub_agent
