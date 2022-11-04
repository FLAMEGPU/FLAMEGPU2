#include <random>
#include <ctime>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"
namespace flamegpu {


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

    FLAMEGPU_AGENT_FUNCTION(Out_AppendTruncate, MessageNone, MessageBruteForce) {
        FLAMEGPU->message_out.setVariable("x", 0);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(Out_AppendTruncate2, MessageNone, MessageBruteForce) {
        FLAMEGPU->message_out.setVariable("x", 1);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(In_AppendTruncate, MessageBruteForce, MessageNone) {
        int count = 0;
        for (auto &message : FLAMEGPU->message_in) {
            count++;
        }
        FLAMEGPU->setVariable<unsigned int>("count", count);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(In_AppendTruncate2, MessageBruteForce, MessageNone) {
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
    FLAMEGPU_AGENT_FUNCTION(OptionalOut_AppendTruncate, MessageNone, MessageBruteForce) {
        if (FLAMEGPU->getVariable<unsigned int>("do_out") > 0) {
            FLAMEGPU->message_out.setVariable("x", 0);
            FLAMEGPU->setVariable<unsigned int>("do_out", 0);
        } else {
            FLAMEGPU->setVariable<unsigned int>("do_out", 1);
        }
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(OptionalOut_AppendTruncate2, MessageNone, MessageBruteForce) {
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
        MessageBruteForce::Description &message = m.newMessage(MESSAGE_NAME);
        message.newVariable<int>("x");
        AgentDescription a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count");
        AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, Out_AppendTruncate);
        fo.setMessageOutput(message);
        AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate);
        fi.setMessageInput(message);

        AgentVector pop(a, (unsigned int)AGENT_COUNT);
        for (AgentVector::Agent ai : pop) {
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
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), AGENT_COUNT);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), AGENT_COUNT);
        }
    }
    TEST(TestMessage_AppendTruncate, Append_KeepData) {
        ModelDescription m(MODEL_NAME);
        MessageBruteForce::Description &message = m.newMessage(MESSAGE_NAME);
        message.newVariable<int>("x");
        AgentDescription a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count0");
        a.newVariable<unsigned int>("count1");
        AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, Out_AppendTruncate);
        fo.setMessageOutput(message);
        AgentFunctionDescription fo2 = a.newFunction(OUT_FUNCTION_NAME2, Out_AppendTruncate2);
        fo2.setMessageOutput(message);
        AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate2);
        fi.setMessageInput(message);

        AgentVector pop(a, (unsigned int)AGENT_COUNT);
        for (AgentVector::Agent ai : pop) {
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
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), AGENT_COUNT);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), AGENT_COUNT);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), AGENT_COUNT);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), AGENT_COUNT);
        }
    }
    TEST(TestMessage_AppendTruncate, OptionalTruncate) {
        ModelDescription m(MODEL_NAME);
        MessageBruteForce::Description &message = m.newMessage(MESSAGE_NAME);
        message.newVariable<int>("x");
        AgentDescription a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count");
        a.newVariable<unsigned int>("do_out");
        AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OptionalOut_AppendTruncate);
        fo.setMessageOutputOptional(true);
        fo.setMessageOutput(message);
        AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate);
        fi.setMessageInput(message);
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        unsigned int result_count = 0;
        AgentVector pop(a, (unsigned int)AGENT_COUNT);
        for (AgentVector::Agent ai : pop) {
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
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), result_count);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count"), AGENT_COUNT - result_count);
        }
    }
    TEST(TestMessage_AppendTruncate, OptionalAppend_KeepData) {
        ModelDescription m(MODEL_NAME);
        MessageBruteForce::Description &message = m.newMessage(MESSAGE_NAME);
        message.newVariable<int>("x");
        AgentDescription a = m.newAgent(AGENT_NAME);
        a.newVariable<unsigned int>("count0");
        a.newVariable<unsigned int>("count1");
        a.newVariable<unsigned int>("do_out");
        AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, OptionalOut_AppendTruncate);
        fo.setMessageOutputOptional(true);
        fo.setMessageOutput(message);
        AgentFunctionDescription fo2 = a.newFunction(OUT_FUNCTION_NAME2, OptionalOut_AppendTruncate2);
        fo2.setMessageOutputOptional(true);
        fo2.setMessageOutput(message);
        AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, In_AppendTruncate2);
        fi.setMessageInput(message);
        std::mt19937_64 rng;
        std::uniform_real_distribution<double> dist(0.0, 1.0);
        unsigned int result_count = 0;
        AgentVector pop(a, (unsigned int)AGENT_COUNT);
        for (AgentVector::Agent ai : pop) {
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
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), result_count);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), AGENT_COUNT - result_count);
        }
        c.step();
        c.getPopulationData(pop);
        // Validate each agent has same result
        for (AgentVector::Agent ai : pop) {
            ASSERT_EQ(ai.getVariable<unsigned int>("count0"), AGENT_COUNT - result_count);
            ASSERT_EQ(ai.getVariable<unsigned int>("count1"), result_count);
        }
    }
    FLAMEGPU_AGENT_FUNCTION(Out_1, MessageNone, MessageBruteForce) {
        FLAMEGPU->message_out.setVariable("x", 1);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(Out_2, MessageNone, MessageBruteForce) {
        FLAMEGPU->message_out.setVariable("x", 2);
        return ALIVE;
    }
    FLAMEGPU_AGENT_FUNCTION(In, MessageBruteForce, MessageNone) {
        unsigned int _0 = 0;
        unsigned int _1 = 0;
        unsigned int _2 = 0;
        for (auto& message : FLAMEGPU->message_in) {
            const int x = message.getVariable<int>("x");
            if (x == 1) {
                _1++;
            } else if (x == 2) {
                _2++;
            } else {
                _0++;
            }
        }
        FLAMEGPU->setVariable<unsigned int>("0", _0);
        FLAMEGPU->setVariable<unsigned int>("1", _1);
        FLAMEGPU->setVariable<unsigned int>("2", _2);
        return ALIVE;
    }
    TEST(TestMessage_AppendTruncate, Append_KeepData_Resize) {
        // This is to a cover a bug (#725), where message list resizing before a message list append, would lose existing message data
        // Resizing is performed automatically, but by having the 2nd agent population to output messages
        // twice the size of the initial agent population, automatic resize should be forced
        ModelDescription m(MODEL_NAME);
        MessageBruteForce::Description& message = m.newMessage(MESSAGE_NAME);
        message.newVariable<int>("x");
        AgentDescription a = m.newAgent("a");
        a.newFunction("Out_1", Out_1).setMessageOutput(message);
        AgentDescription b = m.newAgent("b");
        b.newFunction("Out_2", Out_2).setMessageOutput(message);
        AgentDescription c = m.newAgent("c");
        c.newFunction("In", In).setMessageInput(message);
        c.newVariable<unsigned int>("0", 0);
        c.newVariable<unsigned int>("1", 0);
        c.newVariable<unsigned int>("2", 0);

        LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
        lo.addAgentFunction(Out_1);
        LayerDescription& lo2 = m.newLayer(OUT_LAYER2_NAME);
        lo2.addAgentFunction(Out_2);
        LayerDescription& li = m.newLayer(IN_LAYER_NAME);
        li.addAgentFunction(In);
        AgentVector pop_a(a, AGENT_COUNT);
        AgentVector pop_b(b, AGENT_COUNT * 2);
        AgentVector pop_c(c, 1);
        CUDASimulation sim(m);
        sim.setPopulationData(pop_a);
        sim.setPopulationData(pop_b);
        sim.setPopulationData(pop_c);
        sim.step();
        sim.getPopulationData(pop_c);
        // Validate
        ASSERT_EQ(pop_c[0].getVariable<unsigned int>("0"), 0u);
        ASSERT_EQ(pop_c[0].getVariable<unsigned int>("1"), AGENT_COUNT);
        ASSERT_EQ(pop_c[0].getVariable<unsigned int>("2"), AGENT_COUNT * 2);
    }
}  // namespace test_message_AppendTruncate
}  // namespace flamegpu
