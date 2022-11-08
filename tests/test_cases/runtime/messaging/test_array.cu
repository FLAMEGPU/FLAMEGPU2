#include <chrono>
#include <algorithm>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"
namespace flamegpu {


namespace test_message_array {
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *MESSAGE_NAME = "Message";
    const char *IN_FUNCTION_NAME = "InFunction";
    const char *OUT_FUNCTION_NAME = "OutFunction";
    const char *IN_LAYER_NAME = "InLayer";
    const char *OUT_LAYER_NAME = "OutLayer";
    const unsigned int AGENT_COUNT = 128;
    __device__ const unsigned int dAGENT_COUNT = 128;
FLAMEGPU_AGENT_FUNCTION(OutFunction, MessageNone, MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    FLAMEGPU->message_out.setIndex(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutOptionalFunction, MessageNone, MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    if (index % 2 == 0) {
        FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
        FLAMEGPU->message_out.setIndex(index);
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutOptionalNoneFunction, MessageNone, MessageArray) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutBad, MessageNone, MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    FLAMEGPU->message_out.setIndex(index == 13 ? 0 : index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunction, MessageArray, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const auto &message = FLAMEGPU->message_in.at(my_index);
    FLAMEGPU->setVariable("message_read", message.getVariable<unsigned int>("index_times_3"));
    return ALIVE;
}
TEST(TestMessage_Array, Mandatory) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Create a list of numbers
    std::array<unsigned int, AGENT_COUNT> numbers;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        numbers[i] = i;
    }
    // Shuffle the list of numbers
    const unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937_64(seed));
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", numbers[i]);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        EXPECT_EQ(index * 3, message_read);
    }
}
TEST(TestMessage_Array, Optional) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutOptionalFunction);
    fo.setMessageOutput(message);
    fo.setMessageOutputOptional(true);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Create a list of numbers
    std::array<unsigned int, AGENT_COUNT> numbers;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        numbers[i] = i;
    }
    // Shuffle the list of numbers
    const unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937_64(seed));
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", numbers[i]);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        unsigned int index = ai.getVariable<unsigned int>("index");
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        index = index % 2 == 0 ? index : 0;
        EXPECT_EQ(index * 3, message_read);
    }
}

// Test optional message output, wehre no messages are output.
TEST(TestMessage_Array, OptionalNone) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutOptionalNoneFunction);
    fo.setMessageOutput(message);
    fo.setMessageOutputOptional(true);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
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
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        // no messages should have been read.
        EXPECT_EQ(0u, message_read);
    }
}

FLAMEGPU_AGENT_FUNCTION(OutSimple, MessageNone, MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setIndex(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreTest1W, MessageArray, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in.wrap(my_index);
    auto message = filter.begin();
    unsigned int message_read = 0;
    for (int i = -1; i <= 1; ++i) {
        // Skip ourself
        if (i != 0) {
            // Wrap over boundaries
            const unsigned int their_x = (my_index + i + FLAMEGPU->message_in.size()) % FLAMEGPU->message_in.size();
            if (message->getX() == their_x)
                message_read++;
            ++message;
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreTest2W, MessageArray, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in.wrap(my_index, 2);
    auto message = filter.begin();
    unsigned int message_read = 0;
    for (int i = -2; i <= 2; ++i) {
        // Skip ourself
        if (i != 0) {
            // Wrap over boundaries
            const unsigned int their_x = (my_index + i + FLAMEGPU->message_in.size()) % FLAMEGPU->message_in.size();
            if (message->getX() == their_x)
                message_read++;
            ++message;
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return ALIVE;
}
TEST(TestMessage_Array, Moore1W) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutSimple);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, MooreTest1W);
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
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has read 8 correct messages
    for (AgentVector::Agent ai : pop) {
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        EXPECT_EQ(2u, message_read);
    }
}
TEST(TestMessage_Array, Moore2W) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutSimple);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, MooreTest2W);
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
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has read 8 correct messages
    for (AgentVector::Agent ai : pop) {
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        EXPECT_EQ(4u, message_read);
    }
}
// Exception tests
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array, DuplicateOutputException) {
#else
TEST(TestMessage_Array, DISABLED_DuplicateOutputException) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutBad);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Create a list of numbers
    std::array<unsigned int, AGENT_COUNT> numbers;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        numbers[i] = i;
    }
    // Shuffle the list of numbers
    const unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(numbers.begin(), numbers.end(), std::mt19937_64(seed));
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("message_write", i);  // numbers[i]
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    EXPECT_THROW(c.step(), exception::ArrayMessageWriteConflict);
}
TEST(TestMessage_Array, ArrayLenZeroException) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    EXPECT_THROW(message.setLength(0), exception::InvalidArgument);
}
TEST(TestMessage_Array, UnsetLength) {
    ModelDescription model(MODEL_NAME);
    model.newMessage<MessageArray>(MESSAGE_NAME);
    // message.setLength(5);  // Intentionally commented out
    EXPECT_THROW(CUDASimulation m(model), exception::InvalidMessage);
}
TEST(TestMessage_Array, reserved_name) {
    ModelDescription model(MODEL_NAME);
    MessageArray::Description message = model.newMessage<MessageArray>(MESSAGE_NAME);
    EXPECT_THROW(message.newVariable<int>("_"), exception::ReservedName);
}
FLAMEGPU_AGENT_FUNCTION(countArray, MessageArray, MessageNone) {
    unsigned int value = FLAMEGPU->message_in.at(0).getVariable<unsigned int>("value");
    FLAMEGPU->setVariable<unsigned int>("value", value);
    return ALIVE;
}
TEST(TestMessage_Array, ReadEmpty) {
// What happens if we read a message list before it has been output?
    ModelDescription model("Model");
    {   // Location message
        MessageArray::Description message = model.newMessage<MessageArray>("location");
        message.setLength(2);
        message.newVariable<int>("id");  // unused by current test
        message.newVariable<unsigned int>("value");
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<unsigned int>("value", 32323);  // Count the number of messages read
        agent.newFunction("in", countArray).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(countArray);
    }
    // Create 1 agent
    AgentVector pop_in(model.Agent("agent"), 1);
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(pop_in);
    // Execute model
    EXPECT_NO_THROW(cudaSimulation.step());
    // Check result
    AgentVector pop_out(model.Agent("agent"), 1);
    pop_out[0].setVariable<unsigned int>("value", 22221);
    cudaSimulation.getPopulationData(pop_out);
    EXPECT_EQ(pop_out.size(), 1u);
    auto ai = pop_out[0];
    EXPECT_EQ(ai.getVariable<unsigned int>("value"), 0u);  // Unset array messages should be 0
}
FLAMEGPU_AGENT_FUNCTION(InMooreWrapOutOfBoundsX, MessageArray, MessageNone) {
    for (auto a : FLAMEGPU->message_in.wrap(dAGENT_COUNT)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array, MooreWrap_InitOutOfBoundsX) {
#else
TEST(TestMessage_Array, DISABLED_MooreWrap_InitOutOfBoundsX) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreWrapOutOfBoundsX);
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
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    EXPECT_THROW(c.step(), flamegpu::exception::DeviceError);
}
FLAMEGPU_AGENT_FUNCTION(InMooreWrapBadRadius1, MessageArray, MessageNone) {
    for (auto a : FLAMEGPU->message_in.wrap(0, 0)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array, MooreWrap_BadRadius1) {
#else
TEST(TestMessage_Array, DISABLED_MooreWrap_BadRadius1) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreWrapBadRadius1);
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
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    EXPECT_THROW(c.step(), flamegpu::exception::DeviceError);
}
FLAMEGPU_AGENT_FUNCTION(InMooreWrapBadRadius2, MessageArray, MessageNone) {
    for (auto a : FLAMEGPU->message_in.wrap(0, 64)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array, MooreWrap_BadRadius2) {
#else
TEST(TestMessage_Array, DISABLED_MooreWrap_BadRadius2) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreWrapBadRadius2);
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
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    EXPECT_THROW(c.step(), flamegpu::exception::DeviceError);
}
FLAMEGPU_AGENT_FUNCTION(InMooreOutOfBoundsX, MessageArray, MessageNone) {
    for (auto a : FLAMEGPU->message_in(dAGENT_COUNT)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array, Moore_InitOutOfBoundsX) {
#else
TEST(TestMessage_Array, DISABLED_Moore_InitOutOfBoundsX) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreOutOfBoundsX);
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
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    EXPECT_THROW(c.step(), flamegpu::exception::DeviceError);
}
FLAMEGPU_AGENT_FUNCTION(InMooreBadRadius, MessageArray, MessageNone) {
    for (auto a : FLAMEGPU->message_in(0, 0)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array, Moore_BadRadius) {
#else
TEST(TestMessage_Array, DISABLED_Moore_BadRadius) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreBadRadius);
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
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    EXPECT_THROW(c.step(), flamegpu::exception::DeviceError);
}

/*
 * Test for fixed size grids with various com radii to check edge cases + expected cases.
 * 3x3x3 issue highlighted by see https://github.com/FLAMEGPU/FLAMEGPU2/issues/547
 */
FLAMEGPU_AGENT_FUNCTION(OutSimpleX, MessageNone, MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    FLAMEGPU->message_out.setVariable("index", index);
    FLAMEGPU->message_out.setIndex(x);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreWrapTestXC, MessageArray, MessageNone) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int COMRADIUS = FLAMEGPU->environment.getProperty<unsigned int>("COMRADIUS");
    // Iterate message list counting how many messages were read
    unsigned int count = 0;
    for (const auto &message : FLAMEGPU->message_in.wrap(x, COMRADIUS)) {
        // @todo - check its the correct messages?
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", count);
    return ALIVE;
}

void test_moore_wrap_comradius(
    const unsigned int GRID_WIDTH,
    const unsigned int COMRADIUS
    ) {
    // Calc the population
    const unsigned int agentCount = GRID_WIDTH;

    // Define the model
    ModelDescription model("MooreXR");

    // Use an env var for the communication radius to use, rather than a __device__ or a #define.
    EnvironmentDescription &env = model.Environment();
    env.newProperty<unsigned int>("COMRADIUS", COMRADIUS);

    // Define the message
    MessageArray::Description message = model.newMessage<MessageArray>(MESSAGE_NAME);
    message.newVariable<unsigned int>("index");
    message.setLength(GRID_WIDTH);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("index");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("message_read", UINT_MAX);
    // Define the function and layers.
    AgentFunctionDescription outputFunction = agent.newFunction("OutSimpleX", OutSimpleX);
    outputFunction.setMessageOutput(message);
    AgentFunctionDescription inputFunction = agent.newFunction("MooreWrapTestXC", MooreWrapTestXC);
    inputFunction.setMessageInput(message);
    model.newLayer().addAgentFunction(outputFunction);
    LayerDescription li = model.newLayer();
    li.addAgentFunction(inputFunction);
    // Assign the numbers in shuffled order to agents
    AgentVector population(agent, agentCount);
    for (unsigned int x = 0; x < GRID_WIDTH; x++) {
        unsigned int idx = x;
        AgentVector::Agent instance = population[idx];
        instance.setVariable<unsigned int>("index", idx);
        instance.setVariable<unsigned int>("x", x);
        instance.setVariable<unsigned int>("message_read", UINT_MAX);
    }
    // Set pop in model
    CUDASimulation simulation(model);
    simulation.setPopulationData(population);

    if ((COMRADIUS * 2) + 1 <= GRID_WIDTH) {
        simulation.step();
        simulation.getPopulationData(population);
        // Validate each agent has read correct messages

        // Calc the expected number of messages. This depoends on comm radius for wrapped moore neighbourhood
        const unsigned int expected_count = COMRADIUS * 2;

        for (AgentVector::Agent instance : population) {
            const unsigned int message_read = instance.getVariable<unsigned int>("message_read");
            ASSERT_EQ(expected_count, message_read);
        }
    } else {
        // If the comradius would lead to double message reads, a device error is thrown when SEATBELTS is enabled
        // Behaviour is otherwise undefined
#if !defined(SEATBELTS) || SEATBELTS
        EXPECT_THROW(simulation.step(), flamegpu::exception::DeviceError);
#endif
    }
}
// Test a range of environment sizes for comradius of 1, including small sizes which are an edge case, with wrapping.
TEST(TestMessage_Array, MooreWrapR1) {
    test_moore_wrap_comradius(1, 1);
    test_moore_wrap_comradius(2, 1);
    test_moore_wrap_comradius(3, 1);
    test_moore_wrap_comradius(4, 1);
}

// Test a range of environment sizes for comradius of 2, including small sizes which are an edge case, with wrapped communication.
TEST(TestMessage_Array, MooreWrapR2) {
    test_moore_wrap_comradius(1, 2);
    test_moore_wrap_comradius(2, 2);
    test_moore_wrap_comradius(3, 2);
    test_moore_wrap_comradius(4, 2);
    test_moore_wrap_comradius(5, 2);
    test_moore_wrap_comradius(6, 2);
}

FLAMEGPU_AGENT_FUNCTION(MooreTestXC, MessageArray, MessageNone) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int COMRADIUS = FLAMEGPU->environment.getProperty<unsigned int>("COMRADIUS");
    // Iterate message list counting how many messages were read
    unsigned int count = 0;
    for (const auto &message : FLAMEGPU->message_in(x, COMRADIUS)) {
        // @todo - check its the correct messages?
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", count);
    return ALIVE;
}

void test_mooore_comradius(
    const unsigned int GRID_WIDTH,
    const unsigned int COMRADIUS
    ) {
    // Calc the population
    const unsigned int agentCount = GRID_WIDTH;

    // Define the model
    ModelDescription model("MooreXR");

    // Use an env var for the communication radius to use, rather than a __device__ or a #define.
    EnvironmentDescription &env = model.Environment();
    env.newProperty<unsigned int>("COMRADIUS", COMRADIUS);

    // Define the message
    MessageArray::Description message = model.newMessage<MessageArray>(MESSAGE_NAME);
    message.newVariable<unsigned int>("index");
    message.setLength(GRID_WIDTH);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("index");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("message_read", UINT_MAX);
    // Define the function and layers.
    AgentFunctionDescription outputFunction = agent.newFunction("OutSimpleX", OutSimpleX);
    outputFunction.setMessageOutput(message);
    AgentFunctionDescription inputFunction = agent.newFunction("MooreTestXC", MooreTestXC);
    inputFunction.setMessageInput(message);
    model.newLayer().addAgentFunction(outputFunction);
    LayerDescription li = model.newLayer();
    li.addAgentFunction(inputFunction);
    // Assign the numbers in shuffled order to agents
    AgentVector population(agent, agentCount);
    for (unsigned int x = 0; x < GRID_WIDTH; x++) {
        unsigned int idx = x;
        AgentVector::Agent instance = population[idx];
        instance.setVariable<unsigned int>("index", idx);
        instance.setVariable<unsigned int>("x", x);
        instance.setVariable<unsigned int>("message_read", UINT_MAX);
    }
    // Set pop in model
    CUDASimulation simulation(model);
    simulation.setPopulationData(population);
    simulation.step();
    simulation.getPopulationData(population);
    unsigned int right_count = 0;
    // Validate each agent has read correct number of messages
    for (AgentVector::Agent instance : population) {
        const unsigned int x = instance.getVariable<unsigned int>("x");
        const unsigned int message_read = instance.getVariable<unsigned int>("message_read");

        unsigned int expected_read = 1;
        expected_read *= (std::min<int>(static_cast<int>(x + COMRADIUS), static_cast<int>(GRID_WIDTH) - 1) - std::max<int>(static_cast<int>(x) - static_cast<int>(COMRADIUS), 0) + 1);
        expected_read--;
        // ASSERT_EQ(message_read, expected_read);
        if (message_read == expected_read)
            right_count++;
    }
    ASSERT_EQ(right_count, population.size());
}
// Test a range of environment sizes for comradius of 1, including small sizes which are an edge case.
TEST(TestMessage_Array, MooreR1) {
    test_mooore_comradius(1, 1);
    test_mooore_comradius(2, 1);
    test_mooore_comradius(3, 1);
    test_mooore_comradius(4, 1);
}

// Test a range of environment sizes for comradius of 2, including small sizes which are an edge case.
TEST(TestMessage_Array, MooreR2) {
    test_mooore_comradius(1, 2);
    test_mooore_comradius(2, 2);
    test_mooore_comradius(3, 2);
    test_mooore_comradius(4, 2);
    test_mooore_comradius(5, 2);
    test_mooore_comradius(6, 2);
}

FLAMEGPU_AGENT_FUNCTION(ArrayOut, MessageNone, MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, index * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, index * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, index * 11);
    FLAMEGPU->message_out.setIndex(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn, MessageArray, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const auto &message = FLAMEGPU->message_in.at(my_index);
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
    return ALIVE;
}
TEST(TestMessage_Array, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
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
const char* rtc_ArrayOut_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, index * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, index * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, index * 11);
    FLAMEGPU->message_out.setIndex(index);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageArray, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const auto& message = FLAMEGPU->message_in.at(my_index);
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_Array, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
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

#ifdef USE_GLM
FLAMEGPU_AGENT_FUNCTION(ArrayOut_glm, MessageNone, MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    glm::uvec3 t = glm::uvec3(index * 3, index * 7, index * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setIndex(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_glm, MessageArray, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const auto& message = FLAMEGPU->message_in.at(my_index);
    FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
    return ALIVE;
}
TEST(TestMessage_Array, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
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
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    glm::uvec3 t = glm::uvec3(index * 3, index * 7, index * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setIndex(index);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageArray, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const auto& message = FLAMEGPU->message_in.at(my_index);
    FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_Array, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageArray::Description message = m.newMessage<MessageArray>(MESSAGE_NAME);
    message.setLength(AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
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
TEST(TestMessage_Array, DISABLED_ArrayVariable_glm) { }
TEST(TestRTCMessage_Array, DISABLED_ArrayVariable_glm) { }
#endif


/**
 * Tests to check for a previous bug where CUDAScatter::arrayMessageReorder would lead to a cuda error.
 * The error was triggered when:
 *   >= 2 array message lists, with different numbers of message variables
 *   The list with fewer message variables was reorderd prior to the larger message list
 *   The number of cub temp bytes required for a deviceReduce::Max was fewer than the number of bytes required by the first message list (I.e. relatively small grids, 49x49 still caused the issue)
 * 
 * See https://github.com/FLAMEGPU/FLAMEGPU2/issues/735 for more information.
*/

/*
 * Agent mesasge output function, for a message with one non-coordinate value
 */
FLAMEGPU_AGENT_FUNCTION(OutputA, MessageNone, MessageArray) {
    const unsigned int a = FLAMEGPU->getVariable<unsigned int>("a");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    FLAMEGPU->message_out.setVariable("a", a);
    FLAMEGPU->message_out.setIndex(x);
    return ALIVE;
}
/*
 * Agent mesasge output function, for a message with two non-coordinate value
 */
FLAMEGPU_AGENT_FUNCTION(OutputAB, MessageNone, MessageArray) {
    const unsigned int a = FLAMEGPU->getVariable<unsigned int>("a");
    const unsigned int b = FLAMEGPU->getVariable<unsigned int>("b");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    FLAMEGPU->message_out.setVariable("a", a);
    FLAMEGPU->message_out.setVariable("b", b);
    FLAMEGPU->message_out.setIndex(x);
    return ALIVE;
}
/**
 * Agent function to read an array message list where messages contain  two variables. 
 * This is not strictly needed, but will maintain the usefulness of this test incase message list buildIndex is moved to pre input rather than post output.
 */
FLAMEGPU_AGENT_FUNCTION(IterateAB, MessageArray, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    constexpr unsigned int COMRADIUS = 1;
    // Iterate message list counting how many messages were read.
    unsigned int count = 0;
    for (const auto &message : FLAMEGPU->message_in.wrap(x, COMRADIUS)) {
        const unsigned int m_a = message.getVariable<unsigned int>("a");
        const unsigned int m_b = message.getVariable<unsigned int>("b");
        count++;
    }
    // Don't actually do anything, the function just needs to take the message list as input.
    return ALIVE;
}
/**
 * Agent function to read an array message list where messages contain one variables. 
 * This is not strictly needed, but will maintain the usefulness of this test incase message list buildIndex is moved to pre input rather than post output.
 */
FLAMEGPU_AGENT_FUNCTION(IterateA, MessageArray, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    constexpr unsigned int COMRADIUS = 1;
    // Iterate message list counting how many messages were read.
    unsigned int count = 0;
    for (const auto &message : FLAMEGPU->message_in.wrap(x, COMRADIUS)) {
        const unsigned int m_a = message.getVariable<unsigned int>("a");
        count++;
    }
    // Don't actually do anything, the function just needs to take the message list as input.
    return ALIVE;
}

/**
 * Function which runs a model with multiple message lists containing different numbers of variables to test for a previous memory error
 */
void test_arrayMessageReorderError(
    const unsigned int GRID_WIDTH
) {
    // Calc the population
    const unsigned int agentCount = GRID_WIDTH;

    // Define the model
    ModelDescription model("ArrayMessageReorderError");

    constexpr char MESSAGE_NAME_ONEVAR[] = "A";
    constexpr char MESSAGE_NAME_TWOVAR[] = "AB";

    // Define the message list with different numbers of variables.
    MessageArray::Description messageA = model.newMessage<MessageArray>(MESSAGE_NAME_ONEVAR);
    messageA.newVariable<unsigned int>("a");
    messageA.setLength(GRID_WIDTH);
    // DEfine the message list with two non coordinate values
    MessageArray::Description messageAB = model.newMessage<MessageArray>(MESSAGE_NAME_TWOVAR);
    messageAB.newVariable<unsigned int>("a");
    messageAB.newVariable<unsigned int>("b");
    messageAB.setLength(GRID_WIDTH);

    // Define the agents
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("message_read", UINT_MAX);

    // Define the functions
    AgentFunctionDescription outputAFunction = agent.newFunction("OutputA", OutputA);
    outputAFunction.setMessageOutput(messageA);
    AgentFunctionDescription iterateABFunction = agent.newFunction("IterateAB", IterateAB);
    iterateABFunction.setMessageInput(messageAB);
    AgentFunctionDescription outputABFunction = agent.newFunction("OutputAB", OutputAB);
    outputABFunction.setMessageOutput(messageAB);
    AgentFunctionDescription iterateAFunction = agent.newFunction("IterateA", IterateA);
    iterateAFunction.setMessageInput(messageA);

    // Add the functions to layers. The mesasge list with fewer variables should be output/input prior to the larger list(s)
    model.newLayer().addAgentFunction(outputAFunction);
    model.newLayer().addAgentFunction(iterateAFunction);
    model.newLayer().addAgentFunction(outputABFunction);
    model.newLayer().addAgentFunction(iterateABFunction);

    // Initialise agent variables.
    AgentVector population(agent, agentCount);
    for (unsigned int x = 0; x < GRID_WIDTH; x++) {
        unsigned int idx = x;
        AgentVector::Agent instance = population[idx];
        instance.setVariable<unsigned int>("a", 1u);
        instance.setVariable<unsigned int>("b", 2u);
        instance.setVariable<unsigned int>("x", x);
        instance.setVariable<unsigned int>("message_read", 0u);
    }
    // Construct the simulation object
    CUDASimulation simulation(model);
    // Set the agent population
    simulation.setPopulationData(population);
    // Run a single step of the model, which should not throw any errors.
    EXPECT_NO_THROW(simulation.step());
}
/**
 * Test case which ensures tests small grids with multiple message lists of different numbers of variables run without memory erorrs.
 */ 
TEST(TestMessage_Array, arrayMessageReorderMemorySmall) {
    test_arrayMessageReorderError(4);
    test_arrayMessageReorderError(16);
    test_arrayMessageReorderError(1024);
    test_arrayMessageReorderError(2401);  // 49 * 49
}

/**
 * Test case which ensures tests larger grids with multiple message lists of different numbers of variables run without memory erorrs.
 * This did not previously error, but complements the above tests as temprorary memory from cub operating over the total number of bins can influence the amount of memory allocated.
 */ 
TEST(TestMessage_Array, arrayMessageReorderMemoryLarge) {
    test_arrayMessageReorderError(65536);  // 256 * 256
}

}  // namespace test_message_array
}  // namespace flamegpu
