#include <chrono>
#include <algorithm>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"
namespace flamegpu {


namespace test_message_array_2d {
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *MESSAGE_NAME = "Message";
    const char *IN_FUNCTION_NAME = "InFunction";
    const char *OUT_FUNCTION_NAME = "OutFunction";
    const char *IN_LAYER_NAME = "InLayer";
    const char *OUT_LAYER_NAME = "OutLayer";
    const unsigned int SQRT_AGENT_COUNT = 12;
    __device__ const unsigned int dSQRT_AGENT_COUNT = 12;
    const unsigned int AGENT_COUNT = SQRT_AGENT_COUNT * (SQRT_AGENT_COUNT + 1);
FLAMEGPU_AGENT_FUNCTION(OutFunction, MessageNone, MessageArray2D) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    const unsigned int index_x = index % dSQRT_AGENT_COUNT;
    const unsigned int index_y = index / dSQRT_AGENT_COUNT;
    FLAMEGPU->message_out.setIndex(index_x, index_y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutOptionalFunction, MessageNone, MessageArray2D) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    if (index % 2 == 0) {
        FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
        const unsigned int index_x = index % dSQRT_AGENT_COUNT;
        const unsigned int index_y = index / dSQRT_AGENT_COUNT;
        FLAMEGPU->message_out.setIndex(index_x, index_y);
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutOptionalNoneFunction, MessageNone, MessageArray2D) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutBad, MessageNone, MessageArray2D) {
    unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    index = index == 13 ? 0 : index;
    const unsigned int index_x = index % dSQRT_AGENT_COUNT;
    const unsigned int index_y = index / dSQRT_AGENT_COUNT;
    FLAMEGPU->message_out.setIndex(index_x, index_y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunction, MessageArray2D, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = my_index % dSQRT_AGENT_COUNT;
    const unsigned int index_y = my_index / dSQRT_AGENT_COUNT;
    const auto &message = FLAMEGPU->message_in.at(index_x, index_y);
    FLAMEGPU->setVariable("message_read", message.getVariable<unsigned int>("index_times_3"));
    return ALIVE;
}
TEST(TestMessage_Array2D, Mandatory) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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
TEST(TestMessage_Array2D, Optional) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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
TEST(TestMessage_Array2D, OptionalNone) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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

FLAMEGPU_AGENT_FUNCTION(OutSimple, MessageNone, MessageArray2D) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = index % dSQRT_AGENT_COUNT;
    const unsigned int index_y = index / dSQRT_AGENT_COUNT;
    FLAMEGPU->message_out.setIndex(index_x, index_y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreTest1W, MessageArray2D, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = my_index % dSQRT_AGENT_COUNT;
    const unsigned int index_y = my_index / dSQRT_AGENT_COUNT;

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in.wrap(index_x, index_y);
    auto message = filter.begin();
    unsigned int message_read = 0;
    for (int i = -1; i <= 1; ++i) {
        for (int j = -1; j <= 1; ++j) {
            // Skip ourself
            if (!(i == 0 && j == 0)) {
                // Wrap over boundaries
                const unsigned int their_x = (index_x + i + FLAMEGPU->message_in.getDimX()) % FLAMEGPU->message_in.getDimX();
                const unsigned int their_y = (index_y + j + FLAMEGPU->message_in.getDimY()) % FLAMEGPU->message_in.getDimY();
                if (message->getX() == their_x && message->getY() == their_y)
                    message_read++;
                ++message;
            }
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreTest2W, MessageArray2D, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int index_x = my_index % dSQRT_AGENT_COUNT;
    const unsigned int index_y = my_index / dSQRT_AGENT_COUNT;

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in.wrap(index_x, index_y, 2);
    auto message = filter.begin();
    unsigned int message_read = 0;
    for (int i = -2; i <= 2; ++i) {
        for (int j = -2; j <= 2; ++j) {
            // Skip ourself
            if (!(i == 0 && j == 0)) {
                // Wrap over boundaries
                const unsigned int their_x = (index_x + i + FLAMEGPU->message_in.getDimX()) % FLAMEGPU->message_in.getDimX();
                const unsigned int their_y = (index_y + j + FLAMEGPU->message_in.getDimY()) % FLAMEGPU->message_in.getDimY();
                if (message->getX() == their_x && message->getY() == their_y)
                    message_read++;
                ++message;
            }
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return ALIVE;
}
TEST(TestMessage_Array2D, Moore1W) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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
        EXPECT_EQ(8u, message_read);
    }
}
TEST(TestMessage_Array2D, Moore2W) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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
        EXPECT_EQ(24u, message_read);
    }
}

// Exception tests
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, DuplicateOutputException) {
#else
TEST(TestMessage_Array2D, DISABLED_DuplicateOutputException) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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
TEST(TestMessage_Array2D, ArrayLenZeroException) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    EXPECT_THROW(message.setDimensions(0, SQRT_AGENT_COUNT), exception::InvalidArgument);
    EXPECT_THROW(message.setDimensions({ 0, SQRT_AGENT_COUNT }), exception::InvalidArgument);
    EXPECT_THROW(message.setDimensions(SQRT_AGENT_COUNT, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setDimensions({ SQRT_AGENT_COUNT, 0 }), exception::InvalidArgument);
}
TEST(TestMessage_Array2D, UnsetDimensions) {
    ModelDescription model(MODEL_NAME);
    model.newMessage<MessageArray2D>(MESSAGE_NAME);
    // message.setDimensions(5, 5);  // Intentionally commented out
    EXPECT_THROW(CUDASimulation m(model), exception::InvalidMessage);
}
TEST(TestMessage_Array2D, reserved_name) {
    ModelDescription model(MODEL_NAME);
    MessageArray2D::Description message = model.newMessage<MessageArray2D>(MESSAGE_NAME);
    EXPECT_THROW(message.newVariable<int>("_"), exception::ReservedName);
}
FLAMEGPU_AGENT_FUNCTION(countArray2D, MessageArray2D, MessageNone) {
    unsigned int value = FLAMEGPU->message_in.at(0, 0).getVariable<unsigned int>("value");
    FLAMEGPU->setVariable<unsigned int>("value", value);
    return ALIVE;
}
TEST(TestMessage_Array2D, ReadEmpty) {
// What happens if we read a message list before it has been output?
    ModelDescription model("Model");
    {   // Location message
        MessageArray2D::Description message = model.newMessage<MessageArray2D>("location");
        message.setDimensions(2, 2);
        message.newVariable<int>("id");  // unused by current test
        message.newVariable<unsigned int>("value");  // unused by current test
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<unsigned int>("value", 32323);  // Count the number of messages read
        agent.newFunction("in", countArray2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(countArray2D);
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
FLAMEGPU_AGENT_FUNCTION(InMooreWOutOfBoundsX, MessageArray2D, MessageNone) {
    for (auto a : FLAMEGPU->message_in.wrap(dSQRT_AGENT_COUNT, 0)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, MooreW_InitOutOfBoundsX) {
#else
TEST(TestMessage_Array2D, DISABLED_MooreW_InitOutOfBoundsX) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreWOutOfBoundsX);
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
FLAMEGPU_AGENT_FUNCTION(InMooreWOutOfBoundsY, MessageArray2D, MessageNone) {
    for (auto a : FLAMEGPU->message_in.wrap(0, dSQRT_AGENT_COUNT + 1, 0)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, MooreW_InitOutOfBoundsY) {
#else
TEST(TestMessage_Array2D, DISABLED_MooreW_InitOutOfBoundsY) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreWOutOfBoundsY);
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
FLAMEGPU_AGENT_FUNCTION(InMooreWBadRadius1, MessageArray2D, MessageNone) {
    for (auto a : FLAMEGPU->message_in.wrap(0, 0, 0)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, MooreW_BadRadius1) {
#else
TEST(TestMessage_Array2D, DISABLED_MooreW_BadRadius1) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreWBadRadius1);
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
FLAMEGPU_AGENT_FUNCTION(InMooreWBadRadius2, MessageArray2D, MessageNone) {
    for (auto a : FLAMEGPU->message_in.wrap(0, 0, 6)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, MooreW_BadRadius2) {
#else
TEST(TestMessage_Array2D, DISABLED_MooreW_BadRadius2) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreWBadRadius2);
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
FLAMEGPU_AGENT_FUNCTION(InMooreOutOfBoundsX, MessageArray2D, MessageNone) {
    for (auto a : FLAMEGPU->message_in(dSQRT_AGENT_COUNT, 0)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, Moore_InitOutOfBoundsX) {
#else
TEST(TestMessage_Array2D, DISABLED_Moore_InitOutOfBoundsX) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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
FLAMEGPU_AGENT_FUNCTION(InMooreOutOfBoundsY, MessageArray2D, MessageNone) {
    for (auto a : FLAMEGPU->message_in(0, dSQRT_AGENT_COUNT + 1)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, Moore_InitOutOfBoundsY) {
#else
TEST(TestMessage_Array2D, DISABLED_Moore_InitOutOfBoundsY) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, InMooreOutOfBoundsY);
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
FLAMEGPU_AGENT_FUNCTION(InMooreBadRadius, MessageArray2D, MessageNone) {
    for (auto a : FLAMEGPU->message_in(0, 0, 0)) {
        FLAMEGPU->setVariable<unsigned int>("index", a.getVariable<unsigned int>("index_times_3"));
    }
    return ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(TestMessage_Array2D, Moore_BadRadius) {
#else
TEST(TestMessage_Array2D, DISABLED_Moore_BadRadius) {
#endif
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
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
FLAMEGPU_AGENT_FUNCTION(OutSimpleXY, MessageNone, MessageArray2D) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    FLAMEGPU->message_out.setVariable("index", index);
    FLAMEGPU->message_out.setVariable<unsigned int>("v", x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v2", 0, x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v2", 1, y * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v2", 2, y * 11);
    FLAMEGPU->message_out.setIndex(x, y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreWTestXYC, MessageArray2D, MessageNone) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    const unsigned int COMRADIUS = FLAMEGPU->environment.getProperty<unsigned int>("COMRADIUS");
    // Iterate message list counting how many messages were read.
    unsigned int count = 0;
    bool value_success = true;
    for (const auto &msg : FLAMEGPU->message_in.wrap(x, y, COMRADIUS)) {
        // @todo - check its the correct messages?
        count++;
        if (msg.getVariable<unsigned int, 3>("v2", 0) != msg.getX() * 3 ||
            msg.getVariable<unsigned int, 3>("v2", 1) != msg.getY() * 7 ||
            msg.getVariable<unsigned int, 3>("v2", 2) != msg.getY() * 11 ||
            msg.getVariable<unsigned int>("v") != msg.getX() * 3) {
            value_success = false;
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", count);
    FLAMEGPU->setVariable<unsigned int>("value_success", static_cast<unsigned int>(value_success));
    return ALIVE;
}

void test_moore_wrap_comradius(
    const unsigned int GRID_WIDTH,
    const unsigned int GRID_HEIGHT,
    const unsigned int COMRADIUS
    ) {
    // Calc the population
    const unsigned int agentCount = GRID_WIDTH * GRID_HEIGHT;

    // Define the model
    ModelDescription model("MooreXRC");

    // Use an env var for the communication radius to use, rather than a __device__ or a #define.
    EnvironmentDescription &env = model.Environment();
    env.newProperty<unsigned int>("COMRADIUS", COMRADIUS);

    // Define the message
    MessageArray2D::Description message = model.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.newVariable<unsigned int>("index");
    message.newVariable<unsigned int>("v");
    message.newVariable<unsigned int, 3>("v2");
    message.setDimensions(GRID_WIDTH, GRID_HEIGHT);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("index");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("y");
    agent.newVariable<unsigned int>("message_read", UINT_MAX);
    agent.newVariable<unsigned int>("value_success", 0);
    // Define the function and layers.
    AgentFunctionDescription outputFunction = agent.newFunction("OutSimpleXY", OutSimpleXY);
    outputFunction.setMessageOutput(message);
    AgentFunctionDescription inputFunction = agent.newFunction("MooreWTestXYC", MooreWTestXYC);
    inputFunction.setMessageInput(message);
    model.newLayer().addAgentFunction(outputFunction);
    LayerDescription li = model.newLayer();
    li.addAgentFunction(inputFunction);
    // Assign the numbers in shuffled order to agents
    AgentVector population(agent, agentCount);
    for (unsigned int x = 0; x < GRID_WIDTH; x++) {
        for (unsigned int y = 0; y < GRID_HEIGHT; y++) {
            unsigned int idx = (x * GRID_HEIGHT) + y;
            AgentVector::Agent instance = population[idx];
            instance.setVariable<unsigned int>("index", idx);
            instance.setVariable<unsigned int>("x", x);
            instance.setVariable<unsigned int>("y", y);
            instance.setVariable<unsigned int>("message_read", UINT_MAX);
        }
    }
    // Set pop in model
    CUDASimulation simulation(model);
    simulation.setPopulationData(population);

    if ((COMRADIUS * 2) + 1 <= GRID_WIDTH &&
        (COMRADIUS * 2) + 1 <= GRID_HEIGHT) {
        simulation.step();
        simulation.getPopulationData(population);
        // Validate each agent has read correct messages

        // Calc the expected number of messages. This depoends on comm radius for wrapped moore neighbourhood
        const unsigned int expected_count = static_cast<unsigned int>(pow((COMRADIUS * 2) + 1, 2)) - 1;

        for (AgentVector::Agent instance : population) {
            const unsigned int message_read = instance.getVariable<unsigned int>("message_read");
            ASSERT_EQ(expected_count, message_read);
            const unsigned int value_success = instance.getVariable<unsigned int>("value_success");
            ASSERT_EQ(value_success, 1u);
        }
    } else {
        // If the comradius would lead to double message reads, a device error is thrown when SEATBELTS is enabled
        // Behaviour is otherwise undefined
#if !defined(SEATBELTS) || SEATBELTS
        EXPECT_THROW(simulation.step(), flamegpu::exception::DeviceError);
#endif
    }
}
// Test a range of environment sizes for comradius of 1, including small sizes which are an edge case, with wrapping
TEST(TestMessage_Array2D, MooreWrapR1) {
    test_moore_wrap_comradius(1, 1, 1);
    test_moore_wrap_comradius(2, 2, 1);
    test_moore_wrap_comradius(3, 3, 1);
    test_moore_wrap_comradius(4, 4, 1);
}
// Test a range of environment sizes for comradius of 1, including small sizes which are an edge case, with wrapping and non-uniform dimensions
TEST(TestMessage_Array2D, MooreWrapR1NonUniform) {
    test_moore_wrap_comradius(2, 1, 1);
    test_moore_wrap_comradius(3, 1, 1);
    test_moore_wrap_comradius(4, 1, 1);
}

// Test a range of environment sizes for comradius of 2, including small sizes which are an edge case, with wrapping
TEST(TestMessage_Array2D, MooreWrapR2) {
    test_moore_wrap_comradius(1, 1, 2);
    test_moore_wrap_comradius(2, 2, 2);
    test_moore_wrap_comradius(3, 3, 2);
    test_moore_wrap_comradius(4, 4, 2);
    test_moore_wrap_comradius(5, 5, 2);
    test_moore_wrap_comradius(6, 6, 2);
}
// Test a range of environment sizes for comradius of 2, including small sizes which are an edge case, with wrapping and non-uniform dimensions
TEST(TestMessage_Array2D, MooreWrapR2NonUniform) {
    test_moore_wrap_comradius(2, 1, 2);
    test_moore_wrap_comradius(3, 1, 2);
    test_moore_wrap_comradius(4, 1, 2);
    test_moore_wrap_comradius(5, 1, 2);
    test_moore_wrap_comradius(6, 1, 2);
}

FLAMEGPU_AGENT_FUNCTION(MooreTestXYC, MessageArray2D, MessageNone) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    const unsigned int COMRADIUS = FLAMEGPU->environment.getProperty<unsigned int>("COMRADIUS");
    // Iterate message list counting how many messages were read.
    unsigned int count = 0;
    bool value_success = true;
    for (const auto &msg : FLAMEGPU->message_in(x, y, COMRADIUS)) {
        // @todo - check its the correct messages?
        count++;
        if (msg.getVariable<unsigned int, 3>("v2", 0) != msg.getX() * 3 ||
            msg.getVariable<unsigned int, 3>("v2", 1) != msg.getY() * 7 ||
            msg.getVariable<unsigned int, 3>("v2", 2) != msg.getY() * 11 ||
            msg.getVariable<unsigned int>("v") != msg.getX() * 3) {
            value_success = false;
        }
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", count);
    FLAMEGPU->setVariable<unsigned int>("value_success", static_cast<unsigned int>(value_success));
    return ALIVE;
}

void test_mooore_comradius(
    const unsigned int GRID_WIDTH,
    const unsigned int GRID_HEIGHT,
    const unsigned int COMRADIUS
    ) {
    // Calc the population
    const unsigned int agentCount = GRID_WIDTH * GRID_HEIGHT;

    // Define the model
    ModelDescription model("MooreXRC");

    // Use an env var for the communication radius to use, rather than a __device__ or a #define.
    EnvironmentDescription &env = model.Environment();
    env.newProperty<unsigned int>("COMRADIUS", COMRADIUS);

    // Define the message
    MessageArray2D::Description message = model.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.newVariable<unsigned int>("index");
    message.newVariable<unsigned int>("v");
    message.newVariable<unsigned int, 3>("v2");
    message.setDimensions(GRID_WIDTH, GRID_HEIGHT);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("index");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("y");
    agent.newVariable<unsigned int>("message_read", UINT_MAX);
    agent.newVariable<unsigned int>("value_success", 0);
    // Define the function and layers.
    AgentFunctionDescription outputFunction = agent.newFunction("OutSimpleXY", OutSimpleXY);
    outputFunction.setMessageOutput(message);
    AgentFunctionDescription inputFunction = agent.newFunction("MooreTestXYC", MooreTestXYC);
    inputFunction.setMessageInput(message);
    model.newLayer().addAgentFunction(outputFunction);
    LayerDescription li = model.newLayer();
    li.addAgentFunction(inputFunction);
    // Assign the numbers in shuffled order to agents
    AgentVector population(agent, agentCount);
    for (unsigned int x = 0; x < GRID_WIDTH; x++) {
        for (unsigned int y = 0; y < GRID_HEIGHT; y++) {
            unsigned int idx = (x * GRID_HEIGHT) + y;
            AgentVector::Agent instance = population[idx];
            instance.setVariable<unsigned int>("index", idx);
            instance.setVariable<unsigned int>("x", x);
            instance.setVariable<unsigned int>("y", y);
            instance.setVariable<unsigned int>("message_read", UINT_MAX);
        }
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
        const unsigned int y = instance.getVariable<unsigned int>("y");
        const unsigned int message_read = instance.getVariable<unsigned int>("message_read");

        unsigned int expected_read = 1;
        expected_read *= (std::min<int>(static_cast<int>(x + COMRADIUS), static_cast<int>(GRID_WIDTH) - 1) - std::max<int>(static_cast<int>(x) - static_cast<int>(COMRADIUS), 0) + 1);
        expected_read *= (std::min<int>(static_cast<int>(y + COMRADIUS), static_cast<int>(GRID_HEIGHT) - 1) - std::max<int>(static_cast<int>(y) - static_cast<int>(COMRADIUS), 0) + 1);
        expected_read--;
        // ASSERT_EQ(message_read, expected_read);
        if (message_read == expected_read)
            right_count++;
        const unsigned int value_success = instance.getVariable<unsigned int>("value_success");
        ASSERT_EQ(value_success, 1u);
    }
    ASSERT_EQ(right_count, population.size());
}
// Test a range of environment sizes for comradius of 1, including small sizes which are an edge case.
TEST(TestMessage_Array2D, MooreR1) {
    test_mooore_comradius(1, 1, 1);
    test_mooore_comradius(2, 2, 1);
    test_mooore_comradius(3, 3, 1);
    test_mooore_comradius(4, 4, 1);
}
// Test a range of environment sizes for comradius of 1, with non-uniform dimensions
TEST(TestMessage_Array2D, MooreR1NonUnifiorm) {
    test_mooore_comradius(2, 1, 1);
    test_mooore_comradius(3, 1, 1);
    test_mooore_comradius(4, 1, 1);
}

// Test a range of environment sizes for comradius of 2, including small sizes which are an edge case.
TEST(TestMessage_Array2D, MooreR2) {
    test_mooore_comradius(1, 1, 2);
    test_mooore_comradius(2, 2, 2);
    test_mooore_comradius(3, 3, 2);
    test_mooore_comradius(4, 4, 2);
    test_mooore_comradius(5, 5, 2);
    test_mooore_comradius(6, 6, 2);
}
// Test a range of environment sizes for comradius of 2, with non uniform dimensions
TEST(TestMessage_Array2D, MooreR2NonUniform) {
    test_mooore_comradius(2, 1, 2);
    test_mooore_comradius(3, 1, 2);
    test_mooore_comradius(4, 1, 2);
    test_mooore_comradius(5, 1, 2);
    test_mooore_comradius(6, 1, 2);
}

FLAMEGPU_AGENT_FUNCTION(ArrayOut, MessageNone, MessageArray2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, y * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, y * 11);
    FLAMEGPU->message_out.setVariable<unsigned int>("v2", x * 3);
    FLAMEGPU->message_out.setIndex(x, y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const auto& message = FLAMEGPU->message_in.at(x, y);
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
    return ALIVE;
}
TEST(TestMessage_Array2D, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("v2");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
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
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", {i, j});
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}
const char* rtc_ArrayOut_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, y * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, y * 11);
    FLAMEGPU->message_out.setIndex(x, y);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const auto& message = FLAMEGPU->message_in.at(x, y);
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
    FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_Array2D, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
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
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}

#if defined(USE_GLM)
FLAMEGPU_AGENT_FUNCTION(ArrayOut_glm, MessageNone, MessageArray2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    glm::uvec3 t = glm::uvec3(x * 3, y * 7, y * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setIndex(x, y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_glm, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const auto& message = FLAMEGPU->message_in.at(x, y);
    FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
    return ALIVE;
}
TEST(TestMessage_Array2D, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
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
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}
const char* rtc_ArrayOut_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut_glm, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    glm::uvec3 t = glm::uvec3(x * 3, y * 7, y * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setIndex(x, y);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const auto& message = FLAMEGPU->message_in.at(x, y);
    FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_Array2D, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("v2");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
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
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}
#else
TEST(TestMessage_Array2D, DISABLED_ArrayVariable_glm) { }
TEST(TestRTCMessage_Array2D, DISABLED_ArrayVariable_glm) { }
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
FLAMEGPU_AGENT_FUNCTION(OutputA, MessageNone, MessageArray2D) {
    const unsigned int a = FLAMEGPU->getVariable<unsigned int>("a");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    FLAMEGPU->message_out.setVariable("a", a);
    FLAMEGPU->message_out.setIndex(x, y);
    return ALIVE;
}
/*
 * Agent mesasge output function, for a message with two non-coordinate value
 */
FLAMEGPU_AGENT_FUNCTION(OutputAB, MessageNone, MessageArray2D) {
    const unsigned int a = FLAMEGPU->getVariable<unsigned int>("a");
    const unsigned int b = FLAMEGPU->getVariable<unsigned int>("b");
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    FLAMEGPU->message_out.setVariable("a", a);
    FLAMEGPU->message_out.setVariable("b", b);
    FLAMEGPU->message_out.setIndex(x, y);
    return ALIVE;
}
/**
 * Agent function to read an array message list where messages contain  two variables. 
 * This is not strictly needed, but will maintain the usefulness of this test incase message list buildIndex is moved to pre input rather than post output.
 */
FLAMEGPU_AGENT_FUNCTION(IterateAB, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    constexpr unsigned int COMRADIUS = 1;
    // Iterate message list counting how many messages were read.
    unsigned int count = 0;
    for (const auto &message : FLAMEGPU->message_in.wrap(x, y, COMRADIUS)) {
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
FLAMEGPU_AGENT_FUNCTION(IterateA, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    const unsigned int y = FLAMEGPU->getVariable<unsigned int>("y");
    constexpr unsigned int COMRADIUS = 1;
    // Iterate message list counting how many messages were read.
    unsigned int count = 0;
    for (const auto &message : FLAMEGPU->message_in.wrap(x, y, COMRADIUS)) {
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
    const unsigned int GRID_WIDTH,
    const unsigned int GRID_HEIGHT
) {
    // Calc the population
    const unsigned int agentCount = GRID_WIDTH * GRID_HEIGHT;

    // Define the model
    ModelDescription model("ArrayMessageReorderError");

    constexpr char MESSAGE_NAME_ONEVAR[] = "A";
    constexpr char MESSAGE_NAME_TWOVAR[] = "AB";

    // Define the message list with different numbers of variables.
    MessageArray2D::Description messageA = model.newMessage<MessageArray2D>(MESSAGE_NAME_ONEVAR);
    messageA.newVariable<unsigned int>("a");
    messageA.setDimensions(GRID_WIDTH, GRID_HEIGHT);
    // DEfine the message list with two non coordinate values
    MessageArray2D::Description messageAB = model.newMessage<MessageArray2D>(MESSAGE_NAME_TWOVAR);
    messageAB.newVariable<unsigned int>("a");
    messageAB.newVariable<unsigned int>("b");
    messageAB.setDimensions(GRID_WIDTH, GRID_HEIGHT);

    // Define the agents
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("y");
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
        for (unsigned int y = 0; y < GRID_HEIGHT; y++) {
            unsigned int idx = (x * GRID_HEIGHT) + y;
            AgentVector::Agent instance = population[idx];
            instance.setVariable<unsigned int>("a", 1u);
            instance.setVariable<unsigned int>("b", 2u);
            instance.setVariable<unsigned int>("x", x);
            instance.setVariable<unsigned int>("y", y);
            instance.setVariable<unsigned int>("message_read", 0u);
        }
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
TEST(TestMessage_Array2D, arrayMessageReorderMemorySmall) {
    test_arrayMessageReorderError(4, 4);
    test_arrayMessageReorderError(32, 32);
    test_arrayMessageReorderError(49, 49);  // This was the known case, but smaller tests should also trigger this.
}

/**
 * Test case which ensures tests larger grids with multiple message lists of different numbers of variables run without memory erorrs.
 * This did not previously error, but complements the above tests as temprorary memory from cub operating over the total number of bins can influence the amount of memory allocated.
 */ 
TEST(TestMessage_Array2D, arrayMessageReorderMemoryLarge) {
    // 256x256 was known to require 1023 bytes of cub temporary memory on an SM86 device.
    test_arrayMessageReorderError(256, 256);
}

FLAMEGPU_AGENT_FUNCTION(ArrayIn_vn_r1, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const int relative_x_order[] = { -1, 0, 0, 1 };
    const int relative_y_order[] = { 0, -1, 1, 0 };
    const int len = 4;
    int i = 0;
    bool success = true;
    bool value_success = true;
    for (auto &msg : FLAMEGPU->message_in.vn(x, y)) {
        if (msg.getVariable<unsigned int, 3>("v", 0) != msg.getX() * 3 ||
            msg.getVariable<unsigned int, 3>("v", 1) != msg.getY() * 7 ||
            msg.getVariable<unsigned int, 3>("v", 2) != msg.getY() * 11 ||
            msg.getVariable<unsigned int>("v2") != msg.getX() * 3) {
            value_success = false;
        }
        // Skip relative coords which would be out of bounds
        while (!(static_cast<int>(x) + relative_x_order[i] >= 0 && static_cast<int>(x) + relative_x_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimX()) &&
            static_cast<int>(y) + relative_y_order[i] >= 0 && static_cast<int>(y) + relative_y_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimY()))) {
            ++i;
            if (i >= len)
                break;
        }
        if (i < len) {
            if (msg.getOffsetX() != relative_x_order[i] || msg.getOffsetY() != relative_y_order[i])
                success = false;
        }
        ++i;
    }
    if (i < len) {
        // Skip trailing relative coords which would be out of bounds
        while (!(static_cast<int>(x) + relative_x_order[i] >= 0 && static_cast<int>(x) + relative_x_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimX()) &&
            static_cast<int>(y) + relative_y_order[i] >= 0 && static_cast<int>(y) + relative_y_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimY()))) {
            ++i;
            if (i >= len)
                break;
        }
    }
    if (i != len)
        success = false;

    FLAMEGPU->setVariable<unsigned int>("success", static_cast<unsigned int>(success));
    FLAMEGPU->setVariable<unsigned int>("value_success", static_cast<unsigned int>(value_success));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_vn_r2, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const int relative_x_order[] = { -2, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2 };
    const int relative_y_order[] = { 0, -1, 0, 1, -2, -1, 1, 2, -1, 0, 1, 0 };
    const int len = 12;
    int i = 0;
    bool success = true;
    bool value_success = true;
    for (auto& msg : FLAMEGPU->message_in.vn(x, y, 2)) {
        if (msg.getVariable<unsigned int, 3>("v", 0) != msg.getX() * 3 ||
            msg.getVariable<unsigned int, 3>("v", 1) != msg.getY() * 7 ||
            msg.getVariable<unsigned int, 3>("v", 2) != msg.getY() * 11 ||
            msg.getVariable<unsigned int>("v2") != msg.getX() * 3) {
            value_success = false;
        }
        // Skip relative coords which would be out of bounds
        while (!(static_cast<int>(x) + relative_x_order[i] >= 0 && static_cast<int>(x) + relative_x_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimX()) &&
            static_cast<int>(y) + relative_y_order[i] >= 0 && static_cast<int>(y) + relative_y_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimY()))) {
            ++i;
            if (i >= len)
                break;
        }
        if (i < len) {
            if (msg.getOffsetX() != relative_x_order[i] || msg.getOffsetY() != relative_y_order[i])
                success = false;
        }
        ++i;
    }
    if (i < len) {
        // Skip trailing relative coords which would be out of bounds
        while (!(static_cast<int>(x) + relative_x_order[i] >= 0 && static_cast<int>(x) + relative_x_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimX()) &&
            static_cast<int>(y) + relative_y_order[i] >= 0 && static_cast<int>(y) + relative_y_order[i] < static_cast<int>(FLAMEGPU->message_in.getDimY()))) {
            ++i;
            if (i >= len)
                break;
        }
    }
    if (i != len)
        success = false;

    FLAMEGPU->setVariable<unsigned int>("success", static_cast<unsigned int>(success));
    FLAMEGPU->setVariable<unsigned int>("value_success", static_cast<unsigned int>(value_success));
    return ALIVE;
}
TEST(TestMessage_Array2D, VonNeumann_r1) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("v2");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int>("success", 0);
    a.newVariable<unsigned int>("value_success", 0);
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_vn_r1);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", {i, j});
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (const AgentVector::Agent &ai : pop) {
        unsigned int v = ai.getVariable<unsigned int>("success");
        EXPECT_EQ(v, 1u);
        unsigned int v2 = ai.getVariable<unsigned int>("value_success");
        EXPECT_EQ(v2, 1u);
    }
}
TEST(TestMessage_Array2D, VonNeumann_r2) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("v2");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int>("success", 0);
    a.newVariable<unsigned int>("value_success", 0);
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_vn_r2);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (const AgentVector::Agent& ai : pop) {
        unsigned int v = ai.getVariable<unsigned int>("success");
        EXPECT_EQ(v, 1u);
        unsigned int v2 = ai.getVariable<unsigned int>("value_success");
        EXPECT_EQ(v2, 1u);
    }
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_vn_wrap_r1, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const int relative_x_order[] = { -1, 0, 0, 1 };
    const int relative_y_order[] = { 0, -1, 1, 0 };
    const int len = 4;
    int i = 0;
    bool success = true;
    bool value_success = true;
    for (auto &msg : FLAMEGPU->message_in.vn_wrap(x, y)) {
        if (msg.getVariable<unsigned int, 3>("v", 0) != msg.getX() * 3 ||
            msg.getVariable<unsigned int, 3>("v", 1) != msg.getY() * 7 ||
            msg.getVariable<unsigned int, 3>("v", 2) != msg.getY() * 11 ||
            msg.getVariable<unsigned int>("v2") != msg.getX() * 3) {
            value_success = false;
        }
        if (i < len) {
            if (msg.getOffsetX() != relative_x_order[i] || msg.getOffsetY() != relative_y_order[i])
                success = false;
        }
        ++i;
    }
    if (i != len)
        success = false;

    FLAMEGPU->setVariable<unsigned int>("success", static_cast<unsigned int>(success));
    FLAMEGPU->setVariable<unsigned int>("value_success", static_cast<unsigned int>(value_success));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_vn_wrap_r2, MessageArray2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    const int relative_x_order[] = { -2, -1, -1, -1, 0, 0, 0, 0, 1, 1, 1, 2 };
    const int relative_y_order[] = { 0, -1, 0, 1, -2, -1, 1, 2, -1, 0, 1, 0 };
    const int len = 12;
    int i = 0;
    bool success = true;
    bool value_success = true;
    for (auto& msg : FLAMEGPU->message_in.vn_wrap(x, y, 2)) {
        if (msg.getVariable<unsigned int, 3>("v", 0) != msg.getX() * 3 ||
            msg.getVariable<unsigned int, 3>("v", 1) != msg.getY() * 7 ||
            msg.getVariable<unsigned int, 3>("v", 2) != msg.getY() * 11 ||
            msg.getVariable<unsigned int>("v2") != msg.getX() * 3) {
            value_success = false;
        }
        if (i < len) {
            if (msg.getOffsetX() != relative_x_order[i] || msg.getOffsetY() != relative_y_order[i])
                success = false;
        }
        ++i;
    }
    if (i != len)
        success = false;

    FLAMEGPU->setVariable<unsigned int>("success", static_cast<unsigned int>(success));
    FLAMEGPU->setVariable<unsigned int>("value_success", static_cast<unsigned int>(value_success));
    return ALIVE;
}
TEST(TestMessage_Array2D, VonNeumann_wrap_r1) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("v2");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int>("success", 0);
    a.newVariable<unsigned int>("value_success", 0);
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_vn_wrap_r1);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", {i, j});
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (const AgentVector::Agent &ai : pop) {
        unsigned int v = ai.getVariable<unsigned int>("success");
        EXPECT_EQ(v, 1u);
        unsigned int v2 = ai.getVariable<unsigned int>("value_success");
        EXPECT_EQ(v2, 1u);
    }
}
TEST(TestMessage_Array2D, VonNeumann_wrap_r2) {
    ModelDescription m(MODEL_NAME);
    MessageArray2D::Description message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    message.newVariable<unsigned int>("v2");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int>("success", 0);
    a.newVariable<unsigned int>("value_success", 0);
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_vn_wrap_r2);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (const AgentVector::Agent& ai : pop) {
        unsigned int v = ai.getVariable<unsigned int>("success");
        EXPECT_EQ(v, 1u);
        unsigned int v2 = ai.getVariable<unsigned int>("value_success");
        EXPECT_EQ(v2, 1u);
    }
}
}  // namespace test_message_array_2d
}  // namespace flamegpu
