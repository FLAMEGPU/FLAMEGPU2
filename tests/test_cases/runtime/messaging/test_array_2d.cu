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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Create a list of numbers
    std::array<unsigned int, AGENT_COUNT> numbers;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        numbers[i] = i;
    }
    // Shuffle the list of numbers
    const unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));
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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutOptionalFunction);
    fo.setMessageOutput(message);
    fo.setMessageOutputOptional(true);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Create a list of numbers
    std::array<unsigned int, AGENT_COUNT> numbers;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        numbers[i] = i;
    }
    // Shuffle the list of numbers
    const unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));
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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutOptionalNoneFunction);
    fo.setMessageOutput(message);
    fo.setMessageOutputOptional(true);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);
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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutSimple);
    fo.setMessageOutput(message);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, MooreTest1W);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutSimple);
    fo.setMessageOutput(message);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, MooreTest2W);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutBad);
    fo.setMessageOutput(message);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Create a list of numbers
    std::array<unsigned int, AGENT_COUNT> numbers;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        numbers[i] = i;
    }
    // Shuffle the list of numbers
    const unsigned seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count());
    std::shuffle(numbers.begin(), numbers.end(), std::default_random_engine(seed));
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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
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
    MessageArray2D::Description &message = model.newMessage<MessageArray2D>(MESSAGE_NAME);
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
        MessageArray2D::Description &message = model.newMessage<MessageArray2D>("location");
        message.setDimensions(2, 2);
        message.newVariable<int>("id");  // unused by current test
        message.newVariable<unsigned int>("value");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<unsigned int>("value", 32323);  // Count the number of messages read
        agent.newFunction("in", countArray2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, InMooreWOutOfBoundsX);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, InMooreWOutOfBoundsY);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, InMooreWBadRadius1);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, InMooreWBadRadius2);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, InMooreOutOfBoundsX);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, InMooreOutOfBoundsY);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT + 1);
    message.newVariable<unsigned int>("index_times_3");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, InMooreBadRadius);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    for (const auto &message : FLAMEGPU->message_in.wrap(x, y, COMRADIUS)) {
        // @todo - check its the correct messages?
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", count);
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
    MessageArray2D::Description &message = model.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.newVariable<unsigned int>("index");
    message.setDimensions(GRID_WIDTH, GRID_HEIGHT);
    AgentDescription &agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("index");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("y");
    agent.newVariable<unsigned int>("message_read", UINT_MAX);
    // Define the function and layers.
    AgentFunctionDescription &outputFunction = agent.newFunction("OutSimpleXY", OutSimpleXY);
    outputFunction.setMessageOutput(message);
    AgentFunctionDescription &inputFunction = agent.newFunction("MooreWTestXYC", MooreWTestXYC);
    inputFunction.setMessageInput(message);
    model.newLayer().addAgentFunction(outputFunction);
    LayerDescription &li = model.newLayer();
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
    for (const auto &message : FLAMEGPU->message_in(x, y, COMRADIUS)) {
        // @todo - check its the correct messages?
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("message_read", count);
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
    MessageArray2D::Description &message = model.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.newVariable<unsigned int>("index");
    message.setDimensions(GRID_WIDTH, GRID_HEIGHT);
    AgentDescription &agent = model.newAgent(AGENT_NAME);
    agent.newVariable<unsigned int>("index");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("y");
    agent.newVariable<unsigned int>("message_read", UINT_MAX);
    // Define the function and layers.
    AgentFunctionDescription &outputFunction = agent.newFunction("OutSimpleXY", OutSimpleXY);
    outputFunction.setMessageOutput(message);
    AgentFunctionDescription &inputFunction = agent.newFunction("MooreTestXYC", MooreTestXYC);
    inputFunction.setMessageInput(message);
    model.newLayer().addAgentFunction(outputFunction);
    LayerDescription &li = model.newLayer();
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
    MessageArray2D::Description &message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription& fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription& fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_glm);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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
    MessageArray2D::Description& message = m.newMessage<MessageArray2D>(MESSAGE_NAME);
    message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription& fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription& fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func_glm);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
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

}  // namespace test_message_array_2d
}  // namespace flamegpu
