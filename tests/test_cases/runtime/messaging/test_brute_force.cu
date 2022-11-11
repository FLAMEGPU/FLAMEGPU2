#include <random>
#include <ctime>

#include "flamegpu/flamegpu.h"

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

FLAMEGPU_AGENT_FUNCTION(OutFunction, MessageNone, MessageBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutFunction_Optional, MessageNone, MessageBruteForce) {
    const int x = FLAMEGPU->getVariable<int>("x");
    if (x) FLAMEGPU->message_out.setVariable("x", x);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutFunction_OptionalNone, MessageNone, MessageBruteForce) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunction, MessageBruteForce, MessageNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        product = product < -1000000 ? -1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunction2, MessageBruteForce, MessageNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        product = product < -1000000 ? -1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunctionNone, MessageBruteForce, MessageNone) {
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
    MessageBruteForce::Description message = m.newMessage(MESSAGE_NAME);
    message.newVariable<int>("x");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);

    std::mt19937_64 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        product = product < -1000000 ? -1 : product;
        ai.setVariable<int>("x", x);
        ai.setVariable<int>("sum", 0);
        ai.setVariable<int>("product", 1);
    }
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
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
TEST(TestMessage_BruteForce, Mandatory2) {
    ModelDescription m(MODEL_NAME);
    MessageBruteForce::Description message = m.newMessage(MESSAGE_NAME);
    message.newVariable<int>("x");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction2);
    fi.setMessageInput(message);

    std::mt19937_64 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    int product = 1;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        ai.setVariable<int>("x", x);
        ai.setVariable<int>("sum", 0);
        ai.setVariable<int>("product", 1);
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        product = product < -1000000 ? -1 : product;
    }
    for (AgentVector::Agent ai : pop) {
        const int x = ai.getVariable<int>("x");
        sum += (x + 1);
        product *= (x + 1);
        product = product > 1000000 ? 1 : product;
        product = product < -1000000 ? -1 : product;
    }
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
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
    MessageBruteForce::Description message = m.newMessage(MESSAGE_NAME);
    message.newVariable<int>("x");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction_Optional);
    fo.setMessageOutputOptional(true);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);

    std::mt19937_64 rng(static_cast<unsigned int>(time(nullptr)));
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
            product = product < -1000000 ? -1 : product;
        }
        ai.setVariable<int>("x", x);
        ai.setVariable<int>("sum", 0);
        ai.setVariable<int>("product", 1);
    }
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
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
    MessageBruteForce::Description message = m.newMessage(MESSAGE_NAME);
    message.newVariable<int>("x");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction_Optional);
    fo.setMessageOutputOptional(true);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction2);
    fi.setMessageInput(message);

    std::mt19937_64 rng(static_cast<unsigned int>(time(nullptr)));
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
            product = product < -1000000 ? -1 : product;
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
            product = product < -1000000 ? -1 : product;
        }
    }
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
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
    MessageBruteForce::Description message = m.newMessage(MESSAGE_NAME);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction_OptionalNone);
    fo.setMessageOutputOptional(true);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunctionNone);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
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
    MessageBruteForce::Description message = m.newMessage(MESSAGE_NAME);
    EXPECT_THROW(message.newVariable<int>("_"), exception::ReservedName);
}

FLAMEGPU_AGENT_FUNCTION(countBF, MessageBruteForce, MessageNone) {
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
        MessageBruteForce::Description message = model.newMessage<MessageBruteForce>("location");
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<unsigned int>("count", 0);  // Count the number of messages read
        agent.newFunction("in", countBF).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(countBF);
    }
    // Create 1 agent
    AgentVector pop_in(model.Agent("agent"), 1);
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(pop_in);
    // Execute model
    EXPECT_NO_THROW(cudaSimulation.step());
    // Check result
    AgentVector pop_out(model.Agent("agent"), 1);
    pop_out[0].setVariable<unsigned int>("count", 1);
    cudaSimulation.getPopulationData(pop_out);
    EXPECT_EQ(pop_out.size(), 1u);
    auto ai = pop_out[0];
    EXPECT_EQ(ai.getVariable<unsigned int>("count"), 0u);
}

FLAMEGPU_AGENT_FUNCTION(ArrayOut, MessageNone, MessageBruteForce) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int>("index", index);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, index * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, index * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, index * 11);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn, MessageBruteForce, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<unsigned int>("index") == my_index) {
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
            break;
        }
    }
    return ALIVE;
}
TEST(TestMessage_BruteForce, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageBruteForce::Description message = m.newMessage<MessageBruteForce>(MESSAGE_NAME);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("index");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}
// Test getting and setting the persistent flag state of the message list description
TEST(TestMessage_BruteForce, getSetPersistent) {
    ModelDescription m(MODEL_NAME);
    MessageBruteForce::Description message = m.newMessage(MESSAGE_NAME);
    // message lists should be non-persistent by default
    EXPECT_EQ(message.getPersistent(), false);
    // Settiog the persistent value ot true should not throw
    EXPECT_NO_THROW(message.setPersistent(true));
    // The value should now be true
    EXPECT_EQ(message.getPersistent(), true);
    // Set it to true again, to make sure it isn't an invert
    EXPECT_NO_THROW(message.setPersistent(true));
    EXPECT_EQ(message.getPersistent(), true);
    // And flip it back to false for good measure
    EXPECT_NO_THROW(message.setPersistent(false));
    EXPECT_EQ(message.getPersistent(), false);
}
// Initialise a population for persistnece testing methdos
FLAMEGPU_INIT_FUNCTION(InitPopulationEvenOutputOnly) {
    // Initialise a population in an init function
    auto agent = FLAMEGPU->agent(AGENT_NAME);
    for (uint32_t i = 0; i < AGENT_COUNT; ++i) {
        auto instance = agent.newAgent();
        instance.setVariable<int>("id", i);
        instance.setVariable<unsigned int>("count", 0u);
        instance.setVariable<unsigned int>("sum", 0u);
    }
}
FLAMEGPU_AGENT_FUNCTION_CONDITION(EvenOnlyCondition) {
    return FLAMEGPU->getStepCounter() % 2 == 0;
}
// Fn condition to only run on even iteraitons
FLAMEGPU_AGENT_FUNCTION_CONDITION(ParentEvenOnlyCondition) {
    return FLAMEGPU->environment.getProperty<unsigned int>("parentStepCounter") % 2 == 0;
}
// Simple versionm of the output function, using just a single bin for simplicity
FLAMEGPU_AGENT_FUNCTION(out_simple, MessageNone, MessageBruteForce) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->message_out.setVariable<int>("id", id);
    return ALIVE;
}
// Agent function which iterates read in mesasges and sums the ID
FLAMEGPU_AGENT_FUNCTION(in_simple, MessageBruteForce, MessageNone) {
    const int id = FLAMEGPU->getVariable<int>("id");
    unsigned int count = 0;
    unsigned int sum = 0;
    for (auto &m : FLAMEGPU->message_in) {
        count++;
        sum += m.getVariable<int>("id");
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    FLAMEGPU->setVariable<unsigned int>("sum", sum);
    return ALIVE;
}
// Exit conditon that exits after the 2 steps have executed
FLAMEGPU_EXIT_CONDITION(ExitAfter2) {
    return FLAMEGPU->getStepCounter() >= 2 ? flamegpu::EXIT: flamegpu::CONTINUE;
}
// Step function to assert that the correct number of messages have been read, when checking that messages should not persist.
FLAMEGPU_STEP_FUNCTION(AssertEvenOutputOnly) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Get the population data
    DeviceAgentVector av = agent.getPopulationData();
    // Iterate the population, ensuring that each agent read the correct number of messages and got the correct sum of messages.
    // These values expect only a single bin is used, in the interest of simplicitly.
    const unsigned int exepctedCountEven = agent.count();
    const unsigned int expectedCountOdd = 0u;

    for (const auto& a : av) {
        if (FLAMEGPU->getStepCounter() % 2 == 0) {
            // Even iterations expect the count to match the number of agents, and sum to be non zero.
            ASSERT_EQ(a.getVariable<unsigned int>("count"), exepctedCountEven);
            ASSERT_NE(a.getVariable<unsigned int>("sum"), 0u);
        } else {
            // Odd iters expect 0 count and 0 sum
            ASSERT_EQ(a.getVariable<unsigned int>("count"), expectedCountOdd);
            ASSERT_EQ(a.getVariable<unsigned int>("sum"), 0u);
        }
    }
}
// Step function to assert that the correct number of messages have been read, when checking that messages should persist iterations.
FLAMEGPU_STEP_FUNCTION(AssertPersistent) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Get the population data
    DeviceAgentVector av = agent.getPopulationData();
    // Iterate the population, ensuring that each agent read the correct number of messages and got the correct sum of messages.
    // These values expect only a single bin is used, in the interest of simplicitly.
    const unsigned int exepctedCountEven = agent.count();

    // When messages are meant to persist, there should always be a lot of messages
    for (const auto& a : av) {
        // Even iterations expect the count to match the number of agents, and sum to be non zero.
        ASSERT_EQ(a.getVariable<unsigned int>("count"), exepctedCountEven);
        ASSERT_NE(a.getVariable<unsigned int>("sum"), 0u);
    }
}
// Test for persistence and non-persistence of messaging, by emitting messages on even iterations only
// Test that message list data does not persist iterations, by outputting on even iterations, but not outputting on odd iterations.
TEST(TestMessage_BruteForce, PersistenceOff) {
    // Construct model
    ModelDescription model("TestMessage_BruteForce");
    {
        MessageBruteForce::Description message = model.newMessage<MessageBruteForce>("msg");
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription agent = model.newAgent(AGENT_NAME);
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs
        auto af = agent.newFunction("out", out_simple);
        af.setMessageOutput("msg");
        af.setMessageOutputOptional(true);
        af.setFunctionCondition(EvenOnlyCondition);

        agent.newFunction("in", in_simple).setMessageInput("msg");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(out_simple);
    }
    {   // Layer #2
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(in_simple);
    }

    // Add an init function to generate a population
    model.addInitFunction(InitPopulationEvenOutputOnly);

    // Add a step function which validates the correct number of messages was read
    model.addStepFunction(AssertEvenOutputOnly);

    CUDASimulation cudaSimulation(model);

    // Run for 2 steps, to trigger an odd and an even step.
    cudaSimulation.SimulationConfig().steps = 2;

    EXPECT_NO_THROW(cudaSimulation.simulate());
}

// Test that message list data does not persist iterations, by outputting on even iterations, but not outputting on odd iterations.
TEST(TestMessage_BruteForce, PersistenceOn) {
    // Construct model
    ModelDescription model("TestMessage_BruteForce");
    {
        MessageBruteForce::Description message = model.newMessage<MessageBruteForce>("msg");
        message.newVariable<int>("id");
        // Mark that this should be a persistent mesasge list
        message.setPersistent(true);
    }
    {   // AgentDescription
        AgentDescription agent = model.newAgent(AGENT_NAME);
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs
        auto af = agent.newFunction("out", out_simple);
        af.setMessageOutput("msg");
        af.setMessageOutputOptional(true);
        af.setFunctionCondition(EvenOnlyCondition);

        agent.newFunction("in", in_simple).setMessageInput("msg");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(out_simple);
    }
    {   // Layer #2
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(in_simple);
    }

    // Add an init function to generate a population
    model.addInitFunction(InitPopulationEvenOutputOnly);

    // Add a step function which validates the correct number of messages was read, i.e. always reads non zero messages
    model.addStepFunction(AssertPersistent);

    CUDASimulation cudaSimulation(model);

    // Run for 2 steps, to trigger an odd and an even step.
    cudaSimulation.SimulationConfig().steps = 2;

    EXPECT_NO_THROW(cudaSimulation.simulate());
}


const char* rtc_ArrayOut_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int>("index", index);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, index * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, index * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, index * 11);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<unsigned int>("index") == my_index) {
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
            break;
        }
    }
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_BruteForce, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageBruteForce::Description message = m.newMessage<MessageBruteForce>(MESSAGE_NAME);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("index");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}

#if defined(USE_GLM)
FLAMEGPU_AGENT_FUNCTION(ArrayOut_glm, MessageNone, MessageBruteForce) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int>("index", index);
    glm::uvec3 t = glm::uvec3(index * 3, index * 7, index * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_glm, MessageBruteForce, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<unsigned int>("index") == my_index) {
            FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
            break;
        }
    }
    return ALIVE;
}
TEST(TestMessage_BruteForce, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageBruteForce::Description message = m.newMessage<MessageBruteForce>(MESSAGE_NAME);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("index");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_glm);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}
const char* rtc_ArrayOut_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int>("index", index);
    glm::uvec3 t = glm::uvec3(index * 3, index * 7, index * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<unsigned int>("index") == my_index) {
            FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
            break;
        }
    }
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_BruteForce, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageBruteForce::Description message = m.newMessage<MessageBruteForce>(MESSAGE_NAME);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("index");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func_glm);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}
#else
TEST(TestMessage_BruteForce, DISABLED_ArrayVariable_glm) { }
TEST(TestRTCMessage_BruteForce, DISABLED_ArrayVariable_glm) { }
#endif

}  // namespace test_message_brute_force
}  // namespace flamegpu
