#include <chrono>
#include <algorithm>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_message_array {
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *MESSAGE_NAME = "Message";
    const char *IN_FUNCTION_NAME = "InFunction";
    const char *OUT_FUNCTION_NAME = "OutFunction";
    const char *IN_LAYER_NAME = "InLayer";
    const char *OUT_LAYER_NAME = "OutLayer";
    const unsigned int AGENT_COUNT = 128;
FLAMEGPU_AGENT_FUNCTION(OutFunction, MsgNone, MsgArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    FLAMEGPU->message_out.setIndex(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutOptionalFunction, MsgNone, MsgArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    if (index % 2 == 0) {
        FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
        FLAMEGPU->message_out.setIndex(index);
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OutBad, MsgNone, MsgArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("message_write");
    FLAMEGPU->message_out.setVariable<unsigned int>("index_times_3", index * 3);
    FLAMEGPU->message_out.setIndex(index == 13 ? 0 : index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(InFunction, MsgArray, MsgNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    const auto &message = FLAMEGPU->message_in.at(my_index);
    FLAMEGPU->setVariable("message_read", message.getVariable<unsigned int>("index_times_3"));
    return ALIVE;
}
TEST(TestMessage_Array, Mandatory) {
    ModelDescription m(MODEL_NAME);
    MsgArray::Description &msg = m.newMessage<MsgArray>(MESSAGE_NAME);
    msg.setLength(AGENT_COUNT);
    msg.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutFunction);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(msg);
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
    AgentPopulation pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", numbers[i]);
    }
    // Set pop in model
    CUDAAgentModel c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int index = ai.getVariable<unsigned int>("index");
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        EXPECT_EQ(index * 3, message_read);
    }
}
TEST(TestMessage_Array, Optional) {
    ModelDescription m(MODEL_NAME);
    MsgArray::Description &msg = m.newMessage<MsgArray>(MESSAGE_NAME);
    msg.setLength(AGENT_COUNT);
    msg.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutOptionalFunction);
    fo.setMessageOutput(msg);
    fo.setMessageOutputOptional(true);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(msg);
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
    AgentPopulation pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", numbers[i]);
    }
    // Set pop in model
    CUDAAgentModel c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        unsigned int index = ai.getVariable<unsigned int>("index");
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        index = index % 2 == 0 ? index : 0;
        EXPECT_EQ(index * 3, message_read);
    }
}

FLAMEGPU_AGENT_FUNCTION(OutSimple, MsgNone, MsgArray) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setIndex(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreTest1, MsgArray, MsgNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in(my_index);
    auto msg = filter.begin();
    unsigned int message_read = 0;
    for (int i = -1; i <= 1; ++i) {
        // Skip ourself
        if (i != 0) {
            // Wrap over boundaries
            const unsigned int their_x = (my_index + i + FLAMEGPU->message_in.size()) % FLAMEGPU->message_in.size();
            if (msg->getX() == their_x)
                message_read++;
            ++msg;
        }
    }
    if (msg == filter.end())
        message_read++;
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MooreTest2, MsgArray, MsgNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");

    // Iterate and check it aligns
    auto filter = FLAMEGPU->message_in(my_index, 2);
    auto msg = filter.begin();
    unsigned int message_read = 0;
    for (int i = -2; i <= 2; ++i) {
        // Skip ourself
        if (i != 0) {
            // Wrap over boundaries
            const unsigned int their_x = (my_index + i + FLAMEGPU->message_in.size()) % FLAMEGPU->message_in.size();
            if (msg->getX() == their_x)
                message_read++;
            ++msg;
        }
    }
    if (msg == filter.end())
        message_read++;
    FLAMEGPU->setVariable<unsigned int>("message_read", message_read);
    return ALIVE;
}
TEST(TestMessage_Array, Moore1) {
    ModelDescription m(MODEL_NAME);
    MsgArray::Description &msg = m.newMessage<MsgArray>(MESSAGE_NAME);
    msg.setLength(AGENT_COUNT);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutSimple);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, MooreTest1);
    fi.setMessageInput(msg);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentPopulation pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
    }
    // Set pop in model
    CUDAAgentModel c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has read 8 correct messages
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        EXPECT_EQ(3u, message_read);
    }
}
TEST(TestMessage_Array, Moore2) {
    ModelDescription m(MODEL_NAME);
    MsgArray::Description &msg = m.newMessage<MsgArray>(MESSAGE_NAME);
    msg.setLength(AGENT_COUNT);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutSimple);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, MooreTest2);
    fi.setMessageInput(msg);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentPopulation pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
    }
    // Set pop in model
    CUDAAgentModel c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has read 8 correct messages
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        const unsigned int message_read = ai.getVariable<unsigned int>("message_read");
        EXPECT_EQ(5u, message_read);
    }
}
// Exception tests
TEST(TestMessage_Array, DuplicateOutputException) {
    ModelDescription m(MODEL_NAME);
    MsgArray::Description &msg = m.newMessage<MsgArray>(MESSAGE_NAME);
    msg.setLength(AGENT_COUNT);
    msg.newVariable<unsigned int>("index_times_3");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int>("message_read", UINT_MAX);
    a.newVariable<unsigned int>("message_write");
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OutBad);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, InFunction);
    fi.setMessageInput(msg);
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
    AgentPopulation pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        ai.setVariable<unsigned int>("index", i);
        ai.setVariable<unsigned int>("message_read", UINT_MAX);
        ai.setVariable<unsigned int>("message_write", numbers[i]);
    }
    // Set pop in model
    CUDAAgentModel c(m);
    c.setPopulationData(pop);
    EXPECT_THROW(c.step(), ArrayMessageWriteConflict);
}
TEST(TestMessage_Array, ArrayLenZeroException) {
    ModelDescription m(MODEL_NAME);
    MsgArray::Description &msg = m.newMessage<MsgArray>(MESSAGE_NAME);
    EXPECT_THROW(msg.setLength(0), InvalidArgument);
}
TEST(TestMessage_Array, UnsetLength) {
    ModelDescription model(MODEL_NAME);
    MsgArray::Description &message = model.newMessage<MsgArray>(MESSAGE_NAME);
    // message.setLength(5);  // Intentionally commented out
    EXPECT_THROW(CUDAAgentModel m(model), InvalidMessage);
}

}  // namespace test_message_array
