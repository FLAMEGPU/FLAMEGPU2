#include <random>
#include <ctime>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_message_brute_force {
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *MESSAGE_NAME = "Message";
    const char *IN_FUNCTION_NAME = "InFunction";
    const char *OUT_FUNCTION_NAME = "OutFunction";
    const char *IN_LAYER_NAME = "InLayer";
    const char *OUT_LAYER_NAME = "OutLayer";
    const unsigned int AGENT_COUNT = 128;

FLAMEGPU_AGENT_FUNCTION(OutFunction, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutFunction_Optional, MsgNone, MsgBruteForce) {
    const int x = FLAMEGPU->getVariable<int>("x");
    if (x) FLAMEGPU->message_out.setVariable("x", x);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunction, MsgBruteForce, MsgNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunction2, MsgBruteForce, MsgNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + 1);
    return ALIVE;
}

/**
 * Test whether a group of agents can output unique messages which can then all be read back by the same (all) agents
 */
TEST(TestMessage_BruteForce, Mandatory1) {
    ModelDescription m(MODEL_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    msg.newVariable<int>("x");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(msg);

    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        const int x = dist(rng);
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        ai.setVariable<int>("x", x);
        ai.setVariable<int>("sum", 0);
        ai.setVariable<int>("product", 1);
    }
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    CUDAAgentModel c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
        ASSERT_EQ(ai.getVariable<int>("product"), product);
    }
}
/**
 * Ensures messages are correct on 2nd step
 */
#include "flamegpu/gpu/CUDAScanCompaction.h"
TEST(TestMessage_BruteForce, Mandatory2) {
    ModelDescription m(MODEL_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    msg.newVariable<int>("x");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction2);
    fi.setMessageInput(msg);

    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        const int x = dist(rng);
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        sum += (x + 1);
        product *= (x + 1);
        product = product > 1000000 ? 1 : product;
        ai.setVariable<int>("x", x);
        ai.setVariable<int>("sum", 0);
        ai.setVariable<int>("product", 1);
    }
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    CUDAAgentModel c(m);
    c.SimulationConfig().steps = 2;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
        ASSERT_EQ(ai.getVariable<int>("product"), product);
    }
}


/**
 * Test whether a group of agents can optionally output unique messages which can then all be read back by the same agents
 */
TEST(TestMessage_BruteForce, Optional1) {
    ModelDescription m(MODEL_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    msg.newVariable<int>("x");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction_Optional);
    fo.setMessageOutputOptional(true);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(msg);

    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        const int x = dist(rng);
        if (x) {
            sum += x;
            product *= x;
            product = product > 1000000 ? 1 : product;
        }
        ai.setVariable<int>("x", x);
        ai.setVariable<int>("sum", 0);
        ai.setVariable<int>("product", 1);
    }
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    CUDAAgentModel c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
        ASSERT_EQ(ai.getVariable<int>("product"), product);
    }
}
TEST(TestMessage_BruteForce, Optional2) {
    ModelDescription m(MODEL_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    msg.newVariable<int>("x");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction_Optional);
    fo.setMessageOutputOptional(true);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction2);
    fi.setMessageInput(msg);

    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        const int x = dist(rng);
        if (x) {
            sum += x;
            product *= x;
            product = product > 1000000 ? 1 : product;
        }
        ai.setVariable<int>("x", x);
        ai.setVariable<int>("sum", 0);
        ai.setVariable<int>("product", 1);
    }

    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const int x = ai.getVariable<int>("x");
        if (x + 1) {
            sum += (x + 1);
            product *= (x + 1);
            product = product > 1000000 ? 1 : product;
        }
    }
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    CUDAAgentModel c(m);
    c.SimulationConfig().steps = 2;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
        ASSERT_EQ(ai.getVariable<int>("product"), product);
    }
}

TEST(TestMessage_BruteForce, reserved_name) {
    ModelDescription m(MODEL_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    EXPECT_THROW(msg.newVariable<int>("_"), ReservedName);
}


}  // namespace test_message_brute_force
