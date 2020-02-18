#include <random>
#include <ctime>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_message_AppendTruncate {
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *MESSAGE_NAME = "Message";
    const char *IN_FUNCTION_NAME = "In_AppendTruncate";
    const char *OUT_FUNCTION_NAME = "Out_AppendTruncate";
    const char *IN_LAYER_NAME = "InLayer";
    const char *OUT_LAYER_NAME = "OutLayer";
    const char *OUT_LAYER2_NAME = "OutLayer2";
    const unsigned int AGENT_COUNT = 1024;

    FLAMEGPU_AGENT_FUNCTION(Out_AppendTruncate, MsgNone, MsgBruteForce) {
        FLAMEGPU->message_out.setVariable("x", 0);  // Value doesn't matter we're only tracking count
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(In_AppendTruncate, MsgBruteForce, MsgNone) {
        int count = 0;
        for (auto &message : FLAMEGPU->message_in) {
            count++;
        }
        FLAMEGPU->setVariable<unsigned int>("count", count);
        return ALIVE;
    }


    /**
    * Test whether a group of agents can output unique messages which can then all be read back by the same (all) agents
    */
    TEST(TestMessage_AppendTruncate, Truncate) {
        ModelDescription m(MODEL_NAME);
        MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
        msg.newVariable<int>("x");
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count");
        AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, Out_AppendTruncate);
        fo.setMessageOutput(msg);
        AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate);
        fi.setMessageInput(msg);

        AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getNextInstance();
            ai.setVariable<unsigned int>("count", 0);
        }
        LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
        lo.addAgentFunction(fo);
        LayerDescription &li = m.newLayer(IN_LAYER_NAME);
        li.addAgentFunction(fi);
        CUDAAgentModel c(m);
        c.setPopulationData(pop);
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), AGENT_COUNT);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), AGENT_COUNT);
        }
    }
    TEST(TestMessage_AppendTruncate, Append) {
        ModelDescription m(MODEL_NAME);
        MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
        msg.newVariable<int>("x");
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count");
        AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, Out_AppendTruncate);
        fo.setMessageOutput(msg);
        AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate);
        fi.setMessageInput(msg);

        AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getNextInstance();
            ai.setVariable<unsigned int>("count", 0);
        }
        LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
        lo.addAgentFunction(fo);
        LayerDescription &lo2 = m.newLayer(OUT_LAYER2_NAME);
        lo2.addAgentFunction(fo);
        LayerDescription &li = m.newLayer(IN_LAYER_NAME);
        li.addAgentFunction(fi);
        CUDAAgentModel c(m);
        c.setPopulationData(pop);
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), 2 * AGENT_COUNT);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), 2 * AGENT_COUNT);
        }
    }
    TEST(TestMessage_AppendTruncate, Append_AcrossTop) {
        ModelDescription m(MODEL_NAME);
        MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
        msg.newVariable<int>("x");
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count");
        AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, Out_AppendTruncate);
        fo.setMessageOutput(msg);
        AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate);
        fi.setMessageInput(msg);

        AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getNextInstance();
            ai.setVariable<unsigned int>("count", 0);
        }
        LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
        lo.addAgentFunction(fo);
        LayerDescription &li = m.newLayer(IN_LAYER_NAME);
        li.addAgentFunction(fi);
        LayerDescription &lo2 = m.newLayer(OUT_LAYER2_NAME);
        lo2.addAgentFunction(fo);
        CUDAAgentModel c(m);
        c.setPopulationData(pop);
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), AGENT_COUNT);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), 2 * AGENT_COUNT);
        }
    }
}