#include <random>
#include <ctime>

#include "flamegpu/flamegpu.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace flamegpu {


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
FLAMEGPU_AGENT_FUNCTION(OutFunction_OptionalNone, MsgNone, MsgBruteForce) {
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
FLAMEGPU_AGENT_FUNCTION(InFunctionNone, MsgBruteForce, MsgNone) {
    for (auto &message : FLAMEGPU->message_in) {
        FLAMEGPU->setVariable("message_read", message.getVariable<unsigned int>("index_times_3"));
    }
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
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (AgentVector::Agent ai : pop) {
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
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
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
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (AgentVector::Agent ai : pop) {
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
    CUDASimulation c(m);
    c.SimulationConfig().steps = 2;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
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
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (AgentVector::Agent ai : pop) {
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
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
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
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (AgentVector::Agent ai : pop) {
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

    for (AgentVector::Agent ai : pop) {
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
    CUDASimulation c(m);
    c.SimulationConfig().steps = 2;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
        ASSERT_EQ(ai.getVariable<int>("product"), product);
    }
}
// Test optional message output, wehre no messages are output.
TEST(TestMessage_BruteForce, OptionalNone) {
    ModelDescription m(MODEL_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    msg.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction_OptionalNone);
    fo.setMessageOutputOptional(true);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunctionNone);
    fi.setMessageInput(msg);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);

    // Generate an arbitrary population.
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
    }

    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has seen no messages.
    for (AgentVector::Agent ai : pop) {
        // unsigned int index = ai.getVariable<unsigned int>("index");
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        // no messages should have been read.
        EXPECT_EQ(UINT_MAX, message_read);
    }
}

TEST(TestMessage_BruteForce, reserved_name) {
    ModelDescription m(MODEL_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    EXPECT_THROW(msg.newVariable<int>("_"), ReservedName);
}

FLAMEGPU_AGENT_FUNCTION(countBF, MsgBruteForce, MsgNone) {
    unsigned int count = 0;
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3 Moore neighbourhood
    for (const auto &message : FLAMEGPU->message_in) {
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return ALIVE;
}
TEST(TestMessage_BruteForce, ReadEmpty) {
// What happens if we read a message list before it has been output?
    ModelDescription model("Model");
    {   // Location message
        MsgBruteForce::Description &message = model.newMessage<MsgBruteForce>("location");
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<unsigned int>("count", 0);  // Count the number of messages read
        agent.newFunction("in", countBF).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(countBF);
    }
    // Create 1 agent
    AgentVector pop_in(model.Agent("agent"), 1);
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop_in);
    // Execute model
    EXPECT_NO_THROW(cuda_model.step());
    // Check result
    AgentVector pop_out(model.Agent("agent"), 1);
    pop_out[0].setVariable<unsigned int>("count", 1);
    cuda_model.getPopulationData(pop_out);
    EXPECT_EQ(pop_out.size(), 1u);
    auto ai = pop_out[0];
    EXPECT_EQ(ai.getVariable<unsigned int>("count"), 0u);
}

}  // namespace test_message_brute_force
}  // namespace flamegpu
