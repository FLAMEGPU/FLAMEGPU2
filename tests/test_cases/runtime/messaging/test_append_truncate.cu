#include <random>
#include <ctime>

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace test_message_AppendTruncate {
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const char *MESSAGE_NAME = "Message";
    const char *IN_FUNCTION_NAME = "In_AppendTruncate";
    const char *OUT_FUNCTION_NAME = "Out_AppendTruncate";
    const char *OUT_FUNCTION_NAME2 = "Out_AppendTruncate2";
    const char *IN_LAYER_NAME = "InLayer";
    const char *OUT_LAYER_NAME = "OutLayer";
    const char *OUT_LAYER2_NAME = "OutLayer2";
    const unsigned int AGENT_COUNT = 1024;

    FLAMEGPU_AGENT_FUNCTION(Out_AppendTruncate, MsgNone, MsgBruteForce) {
        FLAMEGPU->message_out.setVariable("x", 0);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(Out_AppendTruncate2, MsgNone, MsgBruteForce) {
        FLAMEGPU->message_out.setVariable("x", 1);
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
    FLAMEGPU_AGENT_FUNCTION(In_AppendTruncate2, MsgBruteForce, MsgNone) {
        int count0 = 0;
        int count1 = 0;
        for (auto &message : FLAMEGPU->message_in) {
            if (message.getVariable<int>("x") == 0) {
                count0++;
            } else if (message.getVariable<int>("x") == 1) {
                count1++;
            }
        }
        FLAMEGPU->setVariable<unsigned int>("count0", count0);
        FLAMEGPU->setVariable<unsigned int>("count1", count1);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(OptionalOut_AppendTruncate, MsgNone, MsgBruteForce) {
        if (FLAMEGPU->getVariable<unsigned int>("do_out") > 0) {
            FLAMEGPU->message_out.setVariable("x", 0);
            FLAMEGPU->setVariable<unsigned int>("do_out", 0);
        } else {
            FLAMEGPU->setVariable<unsigned int>("do_out", 1);
        }
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(OptionalOut_AppendTruncate2, MsgNone, MsgBruteForce) {
        if (FLAMEGPU->getVariable<unsigned int>("do_out") > 0) {
            FLAMEGPU->message_out.setVariable("x", 1);
        }
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
        CUDASimulation c(m);
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
    TEST(TestMessage_AppendTruncate, Append_KeepData) {
        ModelDescription m(MODEL_NAME);
        MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
        msg.newVariable<int>("x");
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count0");
        a.newVariable<unsigned int>("count1");
        AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, Out_AppendTruncate);
        fo.setMessageOutput(msg);
        AgentFunctionDescription &fo2 = a.newFunction(OUT_FUNCTION_NAME2, Out_AppendTruncate2);
        fo2.setMessageOutput(msg);
        AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate2);
        fi.setMessageInput(msg);

        AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getNextInstance();
            ai.setVariable<unsigned int>("count0", 0);
            ai.setVariable<unsigned int>("count1", 0);
        }
        LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
        lo.addAgentFunction(fo);
        LayerDescription &lo2 = m.newLayer(OUT_LAYER2_NAME);
        lo2.addAgentFunction(fo2);
        LayerDescription &li = m.newLayer(IN_LAYER_NAME);
        li.addAgentFunction(fi);
        CUDASimulation c(m);
        c.setPopulationData(pop);
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), AGENT_COUNT);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), AGENT_COUNT);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), AGENT_COUNT);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), AGENT_COUNT);
        }
    }
    TEST(TestMessage_AppendTruncate, OptionalTruncate) {
        ModelDescription m(MODEL_NAME);
        MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
        msg.newVariable<int>("x");
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count");
        a.newVariable<unsigned int>("do_out");
        AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OptionalOut_AppendTruncate);
        fo.setMessageOutputOptional(true);
        fo.setMessageOutput(msg);
        AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate);
        fi.setMessageInput(msg);
        std::default_random_engine rng;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        unsigned int result_count = 0;
        AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getNextInstance();
            if (dist(rng) < 0.7) {  // 70% chance of outputting
                ai.setVariable<unsigned int>("do_out", 1);
                result_count++;
            } else {
                ai.setVariable<unsigned int>("do_out", 0);
            }
            ai.setVariable<unsigned int>("count", 0);
        }
        LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
        lo.addAgentFunction(fo);
        LayerDescription &li = m.newLayer(IN_LAYER_NAME);
        li.addAgentFunction(fi);
        CUDASimulation c(m);
        c.setPopulationData(pop);
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), result_count);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), AGENT_COUNT - result_count);
        }
    }
    TEST(TestMessage_AppendTruncate, OptionalAppend_KeepData) {
        ModelDescription m(MODEL_NAME);
        MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
        msg.newVariable<int>("x");
        AgentDescription &a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count0");
        a.newVariable<unsigned int>("count1");
        a.newVariable<unsigned int>("do_out");
        AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, OptionalOut_AppendTruncate);
        fo.setMessageOutputOptional(true);
        fo.setMessageOutput(msg);
        AgentFunctionDescription &fo2 = a.newFunction(OUT_FUNCTION_NAME2, OptionalOut_AppendTruncate2);
        fo2.setMessageOutputOptional(true);
        fo2.setMessageOutput(msg);
        AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate2);
        fi.setMessageInput(msg);
        std::default_random_engine rng;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        unsigned int result_count = 0;
        AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getNextInstance();
            if (dist(rng) < 0.7) {  // 70% chance of outputting
                ai.setVariable<unsigned int>("do_out", 1);
                result_count++;
            } else {
                ai.setVariable<unsigned int>("do_out", 0);
            }
            ai.setVariable<unsigned int>("count0", 0);
            ai.setVariable<unsigned int>("count1", 0);
        }
        LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
        lo.addAgentFunction(fo);
        LayerDescription &lo2 = m.newLayer(OUT_LAYER2_NAME);
        lo2.addAgentFunction(fo2);
        LayerDescription &li = m.newLayer(IN_LAYER_NAME);
        li.addAgentFunction(fi);
        CUDASimulation c(m);
        c.setPopulationData(pop);
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), result_count);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), AGENT_COUNT - result_count);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
            AgentInstance ai = pop.getInstanceAt(i);
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), AGENT_COUNT - result_count);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), result_count);
        }
    }
}  // namespace test_message_AppendTruncate
