#include "flamegpu/flame_api.h"

#include "gtest/gtest.h"
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
 *    - Mapped -> Unmapped
 *    - Unmapped -> Mapped
 *    - (Due to a branch in the handling, these should be repeated with a full list transition to an empty state, and otherwise)
 *  - In SubModel
 *    - See above, although this is likely less error prone
 *    - Mapped -> Unmapped -> Mapped
 *    - Probably more of a case, to test these with a submodel of the submodel
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

    const char *MAPPED_STATE1 = "mapped1";
    const char *MAPPED_STATE2 = "mapped2";
    const char *UNMAPPED_STATE1 = "unmapped1";
    const char *UNMAPPED_STATE2 = "unmapped2";
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
FLAMEGPU_AGENT_FUNCTION(AddSubVar, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    const unsigned int sub_v = FLAMEGPU->getVariable<unsigned int>("SubVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + sub_v);
    FLAMEGPU->setVariable<unsigned int>("SubVar1", sub_v * 2);
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
FLAMEGPU_HOST_FUNCTION(HostBirth) {
    auto a = FLAMEGPU->agent(AGENT_NAME).newAgent();
    a.setVariable<unsigned int>(AGENT_VAR1_NAME, 5);
    a.setVariable<unsigned int>(AGENT_VAR2_NAME, 500);
}
FLAMEGPU_HOST_FUNCTION(HostBirth2) {
    auto a = FLAMEGPU->agent(AGENT_NAME).newAgent();
    a.setVariable<unsigned int>(AGENT_VAR1_NAME, 5);
}
FLAMEGPU_AGENT_FUNCTION(HostBirthUpdate, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    if (v == 5) {
        FLAMEGPU->setVariable<unsigned int>("AVar1", 500);
    }
    return ALIVE;
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    // Init Model
    CUDASimulation c(m);
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
    for (AgentVector::Agent ai : pop) {
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), mapped_result);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result);
    }
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af + submodel af + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 1 + 10;
    // Unmapped var = unmapped_result + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    for (AgentVector::Agent ai : pop) {
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT*0.75));    // if AGENT_COUNT > 1000 this test will fail
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(pop.size(), AGENT_COUNT/2);
    for (AgentVector::Agent ai : pop) {
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT*0.75));    // if AGENT_COUNT > 1000 this test will fail
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(pop.size(), AGENT_COUNT/2);
    for (AgentVector::Agent ai : pop) {
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT*0.75));    // if AGENT_COUNT > 1000 this test will fail
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(pop.size(), AGENT_COUNT/2);
    for (AgentVector::Agent ai : pop) {
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR_i);
        EXPECT_EQ(_i % 9, 0u);  // Var divides cleanly by 3
        const unsigned int __i = _i/9;  // Calculate original value of AGENT_VAR_i
        EXPECT_NE(__i % 2, 0u);  // Agent doesn't have original AGENT_VAR_i that was supposed to be killed
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), __i + mapped_result2);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), unmapped_result2 - __i);
    }
}

TEST(TestCUDASubAgent, DeviceAgentBirth_BeforeSubModel) {
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT*1.25));    // if AGENT_COUNT != 1000 this test may fail
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(pop.size(), AGENT_COUNT * 1.25 + AGENT_COUNT * 0.25);
    for (AgentVector::Agent ai : pop) {
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
TEST(TestCUDASubAgent, DeviceAgentBirth_InSubModel) {
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT*1.25));    // if AGENT_COUNT != 1000 this test may fail
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(pop.size(), AGENT_COUNT * 1.25 + AGENT_COUNT * 0.25);
    for (AgentVector::Agent ai : pop) {
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
TEST(TestCUDASubAgent, DeviceAgentBirth_InNestedSubModel) {
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT*1.25));    // if AGENT_COUNT != 1000 this test may fail
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(pop.size(), AGENT_COUNT * 1.25 + AGENT_COUNT * 0.25);
    for (AgentVector::Agent ai : pop) {
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

TEST(TestCUDASubAgent, HostAgentBirth_BeforeSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        a.newFunction("", HostBirthUpdate);
        sm.newLayer().addAgentFunction(HostBirthUpdate);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, 2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addHostFunction(HostBirth);
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, 1);
    pop[0].setVariable<unsigned int>(AGENT_VAR2_NAME, 4);  // 1 agent, default values except var2

    // Init Model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop);
    EXPECT_EQ(pop.size(), 2u);
    for (AgentVector::Agent ai : pop) {
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i == 500) {
            // agent was born
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 500u);
        } else if (_i == 4) {
            // Agent is from init
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 1u);
        } else {
            ASSERT_TRUE(false);  // This means AGENT_VAR2_NAME is wrong
        }
    }
}
TEST(TestCUDASubAgent, HostAgentBirth_InSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        sm.newLayer().addHostFunction(HostBirth2);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 1);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, 2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, 1);
    pop[0].setVariable<unsigned int>(AGENT_VAR2_NAME, 4);  // 1 agent, default values except var2

    // Init Model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop);
    EXPECT_EQ(pop.size(), 2u);
    for (AgentVector::Agent ai : pop) {
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i == 2u) {
            // agent was born
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 5u);
        } else if (_i == 4u) {
            // Agent is from init
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 1u);
        } else {
            ASSERT_TRUE(false);  // This means AGENT_VAR2_NAME is wrong
        }
    }
}
TEST(TestCUDASubAgent, HostAgentBirth_InNestedSubModel) {
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
        sm.newLayer().addHostFunction(HostBirth2);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription psm(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = psm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME, 0);
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
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, 2);
        auto &smd = m.newSubModel("proxysub", psm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("proxysub");
    }
    // Init Agents
    AgentVector pop(ma, 1);
    pop[0].setVariable<unsigned int>(AGENT_VAR2_NAME, 4);  // 1 agent, default values except var2

    // Init Model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop);
    EXPECT_EQ(pop.size(), 2u);
    for (AgentVector::Agent ai : pop) {
        const unsigned int _i = ai.getVariable<unsigned int>(AGENT_VAR2_NAME);
        if (_i == 2u) {
            // agent was born
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 5u);
        } else if (_i == 4u) {
            // Agent is from init
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), 1u);
        } else {
            ASSERT_TRUE(false);  // This means AGENT_VAR2_NAME is wrong
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT));
    unsigned int pass_count = 0;
    unsigned int fail_count = 0;
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.75), fail_count);
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT));
    pass_count = 0;
    fail_count = 0;
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.75), fail_count);
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT));
    unsigned int pass_count = 0;
    unsigned int fail_count = 0;
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.75), fail_count);
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT));
    pass_count = 0;
    fail_count = 0;
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.75), fail_count);
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
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>(AGENT_VAR_i, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, static_cast<unsigned int>(i));
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, std::numeric_limits<unsigned int>::max() - i);
        // Other vars all default init
    }
    // Init Model
    CUDASimulation c(m);
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
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT));
    unsigned int pass_count = 0;
    unsigned int fail_count = 0;
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.75), fail_count);
    // Run Model
    c.step();
    // Check result
    // Mapped var = mapped_result + af +  af + submodel + af
    const unsigned int mapped_result2 = mapped_result + 10 + 1 + 10;
    // Unmapped var = unmapped_result + af + af + af
    const unsigned int unmapped_result2 = unmapped_result - 1000 - 1000;
    c.getPopulationData(pop);
    EXPECT_EQ(pop.size(), static_cast<unsigned int>(AGENT_COUNT));
    pass_count = 0;
    fail_count = 0;
    for (AgentVector::Agent ai : pop) {
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
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.25), pass_count);
    EXPECT_EQ(static_cast<unsigned int>(pop.size() * 0.75), fail_count);
}

TEST(TestCUDASubAgent, AgentStateTransition_UnmapToUnmap_BeforeSubModel) {
    // Transitioning a mapped agent between unmapped states should not impact the mapped states within the submodel
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newFunction("", AddOne2);
        sm.newLayer().addAgentFunction(AddOne2);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(UNMAPPED_STATE1);
        ma.newState(UNMAPPED_STATE2);
        ma.newState(MAPPED_STATE1);
        auto &unmapped_fn = ma.newFunction("", AddTen);
        unmapped_fn.setInitialState(UNMAPPED_STATE1);
        unmapped_fn.setEndState(UNMAPPED_STATE2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, UNMAPPED_STATE1);
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_UNMAPPED_STATE1(ma);
    AgentVector pop_UNMAPPED_STATE2(ma);
    AgentVector pop_MAPPED_STATE1(ma);
    c.getPopulationData(pop_UNMAPPED_STATE1, UNMAPPED_STATE1);
    c.getPopulationData(pop_UNMAPPED_STATE2, UNMAPPED_STATE2);
    c.getPopulationData(pop_MAPPED_STATE1, MAPPED_STATE1);
    ASSERT_EQ(pop_UNMAPPED_STATE1.size(), 0u);
    ASSERT_EQ(pop_UNMAPPED_STATE2.size(), AGENT_COUNT);
    ASSERT_EQ(pop_MAPPED_STATE1.size(), AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        // Unmapped agents only call AddTen()
        AgentVector::Agent ai = pop_UNMAPPED_STATE2[i];
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000);
        // Mapped agents only call AddOne2()
        AgentVector::Agent ai2 = pop_MAPPED_STATE1[i];
        EXPECT_EQ(ai2.getVariable<unsigned int>(AGENT_VAR1_NAME), ai2.getVariable<unsigned int>("id") + 1);
        EXPECT_EQ(ai2.getVariable<unsigned int>(AGENT_VAR2_NAME), ai2.getVariable<unsigned int>("id"));
    }
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToMap_BeforeSubModel) {
    // Transitioning a mapped agent between mapped states should impact the mapped states within the submodel
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("", AddOne2);
        af.setInitialState(MAPPED_STATE2);
        af.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AddOne2);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        ma.newState(MAPPED_STATE2);
        auto &unmapped_fn = ma.newFunction("", AddTen);
        unmapped_fn.setInitialState(MAPPED_STATE1);
        unmapped_fn.setEndState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_MAPPED_STATE1(ma);
    AgentVector pop_MAPPED_STATE2(ma);
    c.getPopulationData(pop_MAPPED_STATE1, MAPPED_STATE1);
    c.getPopulationData(pop_MAPPED_STATE2, MAPPED_STATE2);
    ASSERT_EQ(pop_MAPPED_STATE1.size(), 0u);
    ASSERT_EQ(pop_MAPPED_STATE2.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_MAPPED_STATE2) {
        // Mapped agents call AddTen() and AddOne2()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 1);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000);
    }
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToUnmap_BeforeSubModel) {
    // Transitioning a mapped agent from an mapped to unmapped state should cause the submodel to have no impact
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("", AddOne2);
        af.setInitialState(MAPPED_STATE2);
        af.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AddOne2);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE2);
        ma.newState(UNMAPPED_STATE1);
        auto &unmapped_fn = ma.newFunction("", AddTen);
        unmapped_fn.setInitialState(MAPPED_STATE2);
        unmapped_fn.setEndState(UNMAPPED_STATE1);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE2);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_MAPPED_STATE2(ma);
    AgentVector pop_UNMAPPED_STATE1(ma);
    c.getPopulationData(pop_MAPPED_STATE2, MAPPED_STATE2);
    c.getPopulationData(pop_UNMAPPED_STATE1, UNMAPPED_STATE1);
    ASSERT_EQ(pop_MAPPED_STATE2.size(), 0u);
    ASSERT_EQ(pop_UNMAPPED_STATE1.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_UNMAPPED_STATE1) {
        // Mapped agents only call AddTen()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000);
    }
}
TEST(TestCUDASubAgent, AgentStateTransition_UnmapToMap_BeforeSubModel) {
    // Transitioning a mapped agent from an unmapped to mapped state should impact the mapped states within the submodel
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("", AddOne2);
        af.setInitialState(MAPPED_STATE2);
        af.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AddOne2);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(UNMAPPED_STATE1);
        ma.newState(MAPPED_STATE2);
        auto &unmapped_fn = ma.newFunction("", AddTen);
        unmapped_fn.setInitialState(UNMAPPED_STATE1);
        unmapped_fn.setEndState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, UNMAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_UNMAPPED_STATE1(ma);
    AgentVector pop_MAPPED_STATE2(ma);
    c.getPopulationData(pop_UNMAPPED_STATE1, UNMAPPED_STATE1);
    c.getPopulationData(pop_MAPPED_STATE2, MAPPED_STATE2);
    ASSERT_EQ(pop_UNMAPPED_STATE1.size(), 0u);
    ASSERT_EQ(pop_MAPPED_STATE2.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_MAPPED_STATE2) {
        // Mapped agents call AddTen() and AddOne2()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 1);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000);
    }
}
FLAMEGPU_INIT_FUNCTION(InitCreateAgents_AgentStateTransition_UnmapToUnmap_InSubModel) {
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        auto a = FLAMEGPU->agent(AGENT_NAME, UNMAPPED_STATE1).newAgent();
        a.setVariable<unsigned int>(AGENT_VAR1_NAME, UINT_MAX/2);
    }
}
TEST(TestCUDASubAgent, AgentStateTransition_UnmapToUnmap_InSubModel) {
    // The presence of unmapped agents within a submodel will not impact mapped agents
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(UNMAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE2);
        af.setEndState(MAPPED_STATE2);
        auto &af2 = a.newFunction("a2", AddOne2);
        af2.setInitialState(UNMAPPED_STATE1);
        af2.setEndState(UNMAPPED_STATE1);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.newLayer().addAgentFunction(AGENT_NAME, "a2");
        sm.addInitFunction(InitCreateAgents_AgentStateTransition_UnmapToUnmap_InSubModel);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE2);
        auto &unmapped_fn = ma.newFunction("", AddTen);
        unmapped_fn.setInitialState(MAPPED_STATE2);
        unmapped_fn.setEndState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE2);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE2);
    ASSERT_EQ(pop.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop) {
        // Mapped agents call AddTen() and AddOne2()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 1);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000);
    }
    // Sub model unmapped agents have no effect
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToMap_InSubModel) {
    // Agents can move between mapped stats within a submodel, same asthough they were in main model
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE1);
        af.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        ma.newState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_MAPPED_STATE1(ma);
    AgentVector pop_MAPPED_STATE2(ma);
    c.getPopulationData(pop_MAPPED_STATE1, MAPPED_STATE1);
    c.getPopulationData(pop_MAPPED_STATE2, MAPPED_STATE2);
    ASSERT_EQ(pop_MAPPED_STATE1.size(), 0u);
    ASSERT_EQ(pop_MAPPED_STATE2.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_MAPPED_STATE2) {
        // Mapped agents only call AddOne2()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 1);
    }
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToUnmap_InSubModel) {
    // Agents which are in an unmapped state when submodel exits "die"
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(UNMAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE1);
        af.setEndState(UNMAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE1);
    ASSERT_EQ(pop.size(), 0u);
}
TEST(TestCUDASubAgent, AgentStateTransition_UnmapToMap_InSubModel) {
    // Agents created within the submodel will enter the main model (with default values to unmapped vars)
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(UNMAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(UNMAPPED_STATE1);
        af.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.addInitFunction(InitCreateAgents_AgentStateTransition_UnmapToUnmap_InSubModel);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 12);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, 13);
        ma.newVariable<unsigned int>("id", UINT_MAX/2);
        ma.newState(MAPPED_STATE2);
        auto &af = ma.newFunction("a1", AddTen);
        af.setInitialState(MAPPED_STATE2);
        af.setEndState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(af);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(af);
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE2);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE2);
    ASSERT_EQ(pop.size(), AGENT_COUNT*2);
    unsigned int original = 0;
    unsigned int fromsub = 0;
    for (AgentVector::Agent ai : pop) {
        if (ai.getVariable<unsigned int>(AGENT_VAR1_NAME) < 2*AGENT_COUNT) {
            // Original agents call AddTen() twice
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 10);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000 - 1000);
            original++;
        } else {
            // Submodel new agents call AddTen() and AddOne2() once each
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 1);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), 13u - 1000u);
            fromsub++;
        }
    }
    ASSERT_EQ(original, AGENT_COUNT);
    ASSERT_EQ(fromsub, AGENT_COUNT);
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToUnmapToMap_InSubModel) {
    // Agents can move between mapped and unmapped states in submodels without losing their main model variable values
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(UNMAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE1);
        af.setEndState(MAPPED_STATE1);
        auto &af2 = a.newFunction("a2", AddOne2);
        af2.setInitialState(MAPPED_STATE1);
        af2.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.newLayer().addAgentFunction(AGENT_NAME, "a2");
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        ma.newState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, 12+i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_MAPPED_STATE1(ma);
    AgentVector pop_MAPPED_STATE2(ma);
    c.getPopulationData(pop_MAPPED_STATE1, MAPPED_STATE1);
    c.getPopulationData(pop_MAPPED_STATE2, MAPPED_STATE2);
    ASSERT_EQ(pop_MAPPED_STATE1.size(), 0u);
    ASSERT_EQ(pop_MAPPED_STATE2.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_MAPPED_STATE2) {
        // Mapped agents call AddOne2() twice
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 1 + 1);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") + 12);  // This is original unchanged value
    }
}
TEST(TestCUDASubAgent, AgentStateTransition_UnmapToUnmap_InNestedSubModel) {
    // The presence of unmapped agents within a submodel will not impact mapped agents
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(UNMAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE2);
        af.setEndState(MAPPED_STATE2);
        auto &af2 = a.newFunction("a2", AddOne2);
        af2.setInitialState(UNMAPPED_STATE1);
        af2.setEndState(UNMAPPED_STATE1);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.newLayer().addAgentFunction(AGENT_NAME, "a2");
        sm.addInitFunction(InitCreateAgents_AgentStateTransition_UnmapToUnmap_InSubModel);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription proxy(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = proxy.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE2);
        auto &smd = proxy.newSubModel("proxy", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        proxy.newLayer().addSubModel(smd);
        proxy.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE2);
        auto &unmapped_fn = ma.newFunction("", AddTen);
        unmapped_fn.setInitialState(MAPPED_STATE2);
        unmapped_fn.setEndState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", proxy);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(AddTen);
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE2);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE2);
    ASSERT_EQ(pop.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop) {
        // Mapped agents call AddTen() and AddOne2()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 1);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000);
    }
    // Sub model unmapped agents have no effect
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToMap_InNestedSubModel) {
    // Agents can move between mapped stats within a submodel, same asthough they were in main model
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE1);
        af.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription proxy(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = proxy.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &smd = proxy.newSubModel("proxy", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        proxy.newLayer().addSubModel(smd);
        proxy.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        ma.newState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", proxy);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_MAPPED_STATE1(ma);
    AgentVector pop_MAPPED_STATE2(ma);
    c.getPopulationData(pop_MAPPED_STATE1, MAPPED_STATE1);
    c.getPopulationData(pop_MAPPED_STATE2, MAPPED_STATE2);
    ASSERT_EQ(pop_MAPPED_STATE1.size(), 0u);
    ASSERT_EQ(pop_MAPPED_STATE2.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_MAPPED_STATE2) {
        // Mapped agents only call AddOne2()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 1);
    }
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToUnmap_InNestedSubModel) {
    // Agents which are in an unmapped state when submodel exits "die"
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(UNMAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE1);
        af.setEndState(UNMAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription proxy(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = proxy.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &smd = proxy.newSubModel("proxy", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        proxy.newLayer().addSubModel(smd);
        proxy.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        auto &smd = m.newSubModel("sub", proxy);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE1);
    ASSERT_EQ(pop.size(), 0u);
}
TEST(TestCUDASubAgent, AgentStateTransition_UnmapToMap_InNestedSubModel) {
    // Agents created within the submodel will enter the main model (with default values to unmapped vars)
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(UNMAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(UNMAPPED_STATE1);
        af.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.addInitFunction(InitCreateAgents_AgentStateTransition_UnmapToUnmap_InSubModel);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription proxy(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = proxy.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE2);
        auto &smd = proxy.newSubModel("proxy", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        proxy.newLayer().addSubModel(smd);
        proxy.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME, 12);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME, 13);
        ma.newVariable<unsigned int>("id", UINT_MAX/2);
        ma.newState(MAPPED_STATE2);
        auto &af = ma.newFunction("a1", AddTen);
        af.setInitialState(MAPPED_STATE2);
        af.setEndState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", proxy);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addAgentFunction(af);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(af);
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE2);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE2);
    ASSERT_EQ(pop.size(), AGENT_COUNT*2);
    unsigned int original = 0;
    unsigned int fromsub = 0;
    for (AgentVector::Agent ai : pop) {
        if (ai.getVariable<unsigned int>(AGENT_VAR1_NAME) < 2*AGENT_COUNT) {
            // Original agents call AddTen() twice
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 10);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") - 1000 - 1000);
            original++;
        } else {
            // Submodel new agents call AddTen() and AddOne2() once each
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 10 + 1);
            EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), 13u - 1000u);
            fromsub++;
        }
    }
    ASSERT_EQ(original, AGENT_COUNT);
    ASSERT_EQ(fromsub, AGENT_COUNT);
}
TEST(TestCUDASubAgent, AgentStateTransition_MapToUnmapToMap_InNestedSubModel) {
    // Agents can move between mapped and unmapped states in submodels without losing their main model variable values
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(UNMAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &af = a.newFunction("a1", AddOne2);
        af.setInitialState(MAPPED_STATE1);
        af.setEndState(MAPPED_STATE1);
        auto &af2 = a.newFunction("a2", AddOne2);
        af2.setInitialState(MAPPED_STATE1);
        af2.setEndState(MAPPED_STATE2);
        sm.newLayer().addAgentFunction(AGENT_NAME, "a1");
        sm.newLayer().addAgentFunction(AGENT_NAME, "a2");
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription proxy(PROXY_SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = proxy.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(MAPPED_STATE1);
        a.newState(MAPPED_STATE2);
        auto &smd = proxy.newSubModel("proxy", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        proxy.newLayer().addSubModel(smd);
        proxy.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>(AGENT_VAR2_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        ma.newState(MAPPED_STATE2);
        auto &smd = m.newSubModel("sub", proxy);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
        ai.setVariable<unsigned int>(AGENT_VAR2_NAME, 12+i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model
    c.step();
    // Check result
    AgentVector pop_MAPPED_STATE1(ma);
    AgentVector pop_MAPPED_STATE2(ma);
    c.getPopulationData(pop_MAPPED_STATE1, MAPPED_STATE1);
    c.getPopulationData(pop_MAPPED_STATE2, MAPPED_STATE2);
    ASSERT_EQ(pop_MAPPED_STATE1.size(), 0u);
    ASSERT_EQ(pop_MAPPED_STATE2.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop_MAPPED_STATE2) {
        // Mapped agents call AddOne2() twice
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 1 + 1);
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR2_NAME), ai.getVariable<unsigned int>("id") + 12);  // This is original unchanged value
    }
}

TEST(TestCUDASubAgent, UnmappedAgentStatesDontPersistBetweenSubmodelRuns) {
    // Transitioning a mapped agent to and unmapped state, will not allow it to return to mapped state in next step of parent model
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newState(UNMAPPED_STATE1);
        a.newState(MAPPED_STATE1);
        auto &af1 = a.newFunction("a1", AddOne2);
        af1.setInitialState(UNMAPPED_STATE1);
        af1.setEndState(MAPPED_STATE1);
        auto &af2 = a.newFunction("a2", AddOne2);
        af2.setInitialState(MAPPED_STATE1);
        af2.setEndState(UNMAPPED_STATE1);
        sm.newLayer().addAgentFunction(af1);
        sm.newLayer().addAgentFunction(af2);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>("id");
        ma.newState(MAPPED_STATE1);
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    ASSERT_NE(AGENT_COUNT, 0u);  // This constant should not be 0
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop, MAPPED_STATE1);
    // Run Model (agents move into UNMAPPED_STATE1 and die)
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE1);
    ASSERT_EQ(pop.size(), 0u);
    // Run Model (agents nolonger exist, so can't move from UNMAPPED_STATE1 back to MAPPED_STATE1)
    c.step();
    // Check result
    c.getPopulationData(pop, MAPPED_STATE1);
    ASSERT_EQ(pop.size(), 0u);
}
TEST(TestCUDASubAgent, UnmappedVariablesResetToDefaultBetweenSubmodelRuns) {
    // Agents that rely on unmapped subagent variables will receive the default value each time the submodel is called by the parent model
    ModelDescription sm(SUB_MODEL_NAME);
    {
        // Define SubModel
        auto &a = sm.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>(SUB_VAR1_NAME, 1);
        a.newVariable<unsigned int>(AGENT_VAR1_NAME);
        a.newFunction("", AddSubVar);
        sm.newLayer().addAgentFunction(AddSubVar);
        sm.addExitCondition(ExitAlways);
    }
    ModelDescription m(MODEL_NAME);
    auto &ma = m.newAgent(AGENT_NAME);
    {
        // Define Model
        ma.newVariable<unsigned int>(AGENT_VAR1_NAME);
        ma.newVariable<unsigned int>("id");
        auto &smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
        m.newLayer().addSubModel("sub");
    }
    // Init Agents
    AgentVector pop(ma, static_cast<unsigned int>(AGENT_COUNT));
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("id", i);
        ai.setVariable<unsigned int>(AGENT_VAR1_NAME, i);
    }
    // Init Model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.applyConfig();
    c.setPopulationData(pop);
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop);
    ASSERT_EQ(pop.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop) {
        // Mapped agents only call AddSubVar()
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 1);
    }
    // Run Model
    c.step();
    // Check result
    c.getPopulationData(pop);
    ASSERT_EQ(pop.size(), AGENT_COUNT);
    for (AgentVector::Agent ai : pop) {
        // Mapped agents only call AddSubVar() (with default value of subvar)
        EXPECT_EQ(ai.getVariable<unsigned int>(AGENT_VAR1_NAME), ai.getVariable<unsigned int>("id") + 1 + 1);  // If unmapped subvar persists would be + 1 + 2
    }
}

FLAMEGPU_AGENT_FUNCTION(CopyID, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<id_t>("id_copy", FLAMEGPU->getID());
    return ALIVE;
}
FLAMEGPU_EXIT_CONDITION(AlwaysExit) {
    return EXIT;
}
TEST(TestCUDASubAgent, AgentID_BindsID) {
    // Define a model whereby, the submodel copies agent id to agent var
    // Step model
    // Get agent pop and check agent var value matches ID
    ModelDescription submodel("subm");
    AgentDescription& agent1 = submodel.newAgent("agent");
    agent1.newVariable<id_t>("id_copy");
    AgentFunctionDescription& af1 = agent1.newFunction("copy_id", CopyID);
    submodel.newLayer().addAgentFunction(af1);
    submodel.addExitCondition(AlwaysExit);

    ModelDescription model("subm");
    AgentDescription& agent2 = model.newAgent("agent");
    agent2.newVariable<id_t>("id_copy");
    SubModelDescription& smd = model.newSubModel("sm", submodel);
    smd.bindAgent("agent", "agent", true);
    model.newLayer().addSubModel(smd);

    CUDASimulation sim(model);
    AgentVector pop_in(agent2, 100);
    sim.setPopulationData(pop_in);
    sim.step();
    AgentVector pop_out(agent2, 100);
    sim.getPopulationData(pop_out);

    for (auto p : pop_out) {
        EXPECT_NE(p.getID(), ID_NOT_SET);
        EXPECT_EQ(p.getID(), p.getVariable<id_t>("id_copy"));
    }
}
// Submodel unbound agents id counter resets
FLAMEGPU_AGENT_FUNCTION(BirthAndCopyID, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<id_t>("id_copy", FLAMEGPU->agent_out.getID());
    return ALIVE;
}
TEST(TestCUDASubAgent, AgentID_ResetsOunboundAgentID) {
    // Define a model whereby, the submodel every agent births an agent (of a second type not bound to parent model) and copies that agent's id to agent var
    // Step model
    // Get agent pop
    // Step model
    // Get agent pop and check set of copied IDs match the first pop's ids
    ModelDescription submodel("subm");
    AgentDescription& baby = submodel.newAgent("baby");
    AgentDescription& agent1 = submodel.newAgent("agent");
    agent1.newVariable<id_t>("id_copy");
    AgentFunctionDescription& af1 = agent1.newFunction("birth_and_copy_id", BirthAndCopyID);
    af1.setAgentOutput(baby);
    submodel.newLayer().addAgentFunction(af1);
    submodel.addExitCondition(AlwaysExit);

    ModelDescription model("subm");
    AgentDescription& agent2 = model.newAgent("agent");
    agent2.newVariable<id_t>("id_copy");
    SubModelDescription& smd = model.newSubModel("sm", submodel);
    smd.bindAgent("agent", "agent", true);
    model.newLayer().addSubModel(smd);

    CUDASimulation sim(model);
    AgentVector pop_in(agent2, 100);
    sim.setPopulationData(pop_in);
    sim.step();
    AgentVector pop_out1(agent2, 100);
    sim.getPopulationData(pop_out1);
    sim.setPopulationData(pop_in);  // Reset the copy_id back
    sim.step();
    AgentVector pop_out2(agent2, 100);
    sim.getPopulationData(pop_out2);
    sim.step();                       // Attempt it without resetting copy_id back too
    AgentVector pop_out3(agent2, 100);
    sim.getPopulationData(pop_out3);

    std::set<id_t> ids1, ids2, ids3;
    for (unsigned int i = 0; i < pop_in.size(); ++i) {
        ids1.insert(pop_out1[i].getVariable<id_t>("id_copy"));
        ids2.insert(pop_out2[i].getVariable<id_t>("id_copy"));
        ids3.insert(pop_out3[i].getVariable<id_t>("id_copy"));
    }
    EXPECT_EQ(ids1.size(), pop_in.size());
    ASSERT_EQ(ids1, ids2);
    ASSERT_EQ(ids1, ids3);
}

};  // namespace test_cuda_sub_agent
