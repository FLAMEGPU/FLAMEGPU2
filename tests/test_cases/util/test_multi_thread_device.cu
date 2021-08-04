#include <thread>
#include <vector>
#include <memory>

#include "flamegpu/flamegpu.h"
#include "gtest/gtest.h"
#include "flamegpu/util/detail/compute_capability.cuh"

namespace flamegpu {


namespace test_multi_thread_device {
const char *MODEL_NAME = "Model";
const char *AGENT_NAME = "Agent1";
const char *MESSAGE_NAME = "Message1";
const char *FUNCTION_NAME1 = "Fn1";
const char *FUNCTION_NAME2 = "Fn2";

void runSim(CUDASimulation &sim, bool &exception_thrown) {
    exception_thrown = false;
    try {
        sim.simulate();
    } catch(std::exception &e) {
        exception_thrown = true;
        printf("%s\n", e.what());
    }
}

FLAMEGPU_AGENT_FUNCTION(SlowFn, MessageNone, MessageNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (int i = 0; i < FLAMEGPU->environment.getProperty<int>("ten thousand"); ++i) {
        x += FLAMEGPU->environment.getProperty<int>("zero");
    }
    FLAMEGPU->setVariable<int>("x", x + 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(FastFn, MessageNone, MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + 1);
    return ALIVE;
}
TEST(MultiThreadDeviceTest, SameModelSeperateThread_Agent) {
    const unsigned int POP_SIZE = 10000;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn);
    a.newFunction(FUNCTION_NAME2, FastFn);
    m.newLayer().addAgentFunction(SlowFn);
    m.newLayer().addAgentFunction(FastFn);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);

    AgentVector pop_in1(a, POP_SIZE);
    AgentVector pop_in2(a, POP_SIZE);
    AgentVector pop_in3(a, POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop_in1[i].setVariable<int>("x", 1);
        pop_in2[i].setVariable<int>("x", 2);
        pop_in3[i].setVariable<int>("x", 3);
    }

    CUDASimulation sim1(m);
    sim1.SimulationConfig().steps = 10;
    sim1.setPopulationData(pop_in1);
    CUDASimulation sim2(m);
    sim2.SimulationConfig().steps = 10;
    sim2.setPopulationData(pop_in2);
    CUDASimulation sim3(m);
    sim3.SimulationConfig().steps = 10;
    sim3.setPopulationData(pop_in3);

    bool has_error_1 = false, has_error_2 = false, has_error_3 = false;
    std::thread thread1(runSim, std::ref(sim1), std::ref(has_error_1));
    std::thread thread2(runSim, std::ref(sim2), std::ref(has_error_2));
    std::thread thread3(runSim, std::ref(sim3), std::ref(has_error_3));
    // synchronize threads:
    thread1.join();
    thread2.join();
    thread3.join();
    // Check exceptions
    ASSERT_FALSE(has_error_1);
    ASSERT_FALSE(has_error_2);
    ASSERT_FALSE(has_error_3);
    // Check results
    AgentVector pop1(a, POP_SIZE);
    AgentVector pop2(a, POP_SIZE);
    AgentVector pop3(a, POP_SIZE);
    sim1.getPopulationData(pop1);
    sim2.getPopulationData(pop2);
    sim3.getPopulationData(pop3);
    // Use expect, rather than assert for first 3 agents.
    for (unsigned int i = 0; i < 3; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 10 steps with 2x +1 each.
        EXPECT_EQ(x1, 21);
        EXPECT_EQ(x2, 22);
        EXPECT_EQ(x3, 23);
    }
    for (unsigned int i = 3; i < POP_SIZE; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 10 steps with 2x +1 each.
        ASSERT_EQ(x1, 21);
        ASSERT_EQ(x2, 22);
        ASSERT_EQ(x3, 23);
    }
}

FLAMEGPU_AGENT_FUNCTION(SlowFnMessage, MessageBruteForce, MessageNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    int y = 0;
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (auto &m : FLAMEGPU->message_in) {
        y += m.getVariable<int>("x");
    }
    FLAMEGPU->setVariable<int>("x", x + y);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(FastFnMessage, MessageNone, MessageBruteForce) {
    const int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->message_out.setVariable<int>("x", 1);
    return ALIVE;
}
TEST(MultiThreadDeviceTest, SameModelSeperateThread_Message) {
    const unsigned int POP_SIZE = 10000;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    MessageBruteForce::Description &message = m.newMessage(MESSAGE_NAME);
    message.newVariable<int>("x");
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, FastFnMessage).setMessageOutput(message);
    a.newFunction(FUNCTION_NAME2, SlowFnMessage).setMessageInput(message);
    m.newLayer().addAgentFunction(FastFnMessage);
    m.newLayer().addAgentFunction(SlowFnMessage);

    AgentVector pop_in1(a, POP_SIZE);
    AgentVector pop_in2(a, POP_SIZE);
    AgentVector pop_in3(a, POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop_in1[i].setVariable<int>("x", 1);
        pop_in2[i].setVariable<int>("x", 2);
        pop_in3[i].setVariable<int>("x", 3);
    }

    CUDASimulation sim1(m);
    sim1.SimulationConfig().steps = 10;
    sim1.setPopulationData(pop_in1);
    CUDASimulation sim2(m);
    sim2.SimulationConfig().steps = 10;
    sim2.setPopulationData(pop_in2);
    CUDASimulation sim3(m);
    sim3.SimulationConfig().steps = 10;
    sim3.setPopulationData(pop_in3);

    bool has_error_1 = false, has_error_2 = false, has_error_3 = false;
    std::thread thread1(runSim, std::ref(sim1), std::ref(has_error_1));
    std::thread thread2(runSim, std::ref(sim2), std::ref(has_error_2));
    std::thread thread3(runSim, std::ref(sim3), std::ref(has_error_3));
    // synchronize threads:
    thread1.join();
    thread2.join();
    thread3.join();
    // Check exceptions
    ASSERT_FALSE(has_error_1);
    ASSERT_FALSE(has_error_2);
    ASSERT_FALSE(has_error_3);
    // Check results
    AgentVector pop1(a, POP_SIZE);
    AgentVector pop2(a, POP_SIZE);
    AgentVector pop3(a, POP_SIZE);
    sim1.getPopulationData(pop1);
    sim2.getPopulationData(pop2);
    sim3.getPopulationData(pop3);
    // Use expect, rather than assert for first 3 agents.
    for (unsigned int i = 0; i < 3; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 10 steps with 1x +10000 each.
        EXPECT_EQ(x1, 100001);
        EXPECT_EQ(x2, 100002);
        EXPECT_EQ(x3, 100003);
    }
    for (unsigned int i = 3; i < POP_SIZE; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 10 steps with 1x +10000 each.
        ASSERT_EQ(x1, 100001);
        ASSERT_EQ(x2, 100002);
        ASSERT_EQ(x3, 100003);
    }
}

FLAMEGPU_AGENT_FUNCTION(SlowFn2, MessageNone, MessageNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (int i = 0; i < FLAMEGPU->environment.getProperty<int>("ten thousand"); ++i) {
        x += FLAMEGPU->environment.getProperty<int>("zero");
    }
    FLAMEGPU->setVariable<int>("x", x + FLAMEGPU->environment.getProperty<int>("one"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(FastFn2, MessageNone, MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + FLAMEGPU->environment.getProperty<int>("three"));
    return ALIVE;
}
TEST(MultiThreadDeviceTest, SameModelSeperateThread_Environment) {
    const unsigned int POP_SIZE = 10000;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers environment variable get
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn2);
    a.newFunction(FUNCTION_NAME2, FastFn2);
    m.newLayer().addAgentFunction(SlowFn2);
    m.newLayer().addAgentFunction(FastFn2);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);
    m.Environment().newProperty<int>("one", 1);
    m.Environment().newProperty<int>("three", 3);

    AgentVector pop_in1(a, POP_SIZE);
    AgentVector pop_in2(a, POP_SIZE);
    AgentVector pop_in3(a, POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop_in1[i].setVariable<int>("x", 1);
        pop_in2[i].setVariable<int>("x", 2);
        pop_in3[i].setVariable<int>("x", 3);
    }

    CUDASimulation sim1(m);
    sim1.SimulationConfig().steps = 10;
    sim1.setPopulationData(pop_in1);
    m.Environment().setProperty<int>("one", 10);
    m.Environment().setProperty<int>("three", 30);
    CUDASimulation sim2(m);
    sim2.SimulationConfig().steps = 10;
    sim2.setPopulationData(pop_in2);
    m.Environment().setProperty<int>("one", 100);
    m.Environment().setProperty<int>("three", 300);
    CUDASimulation sim3(m);
    sim3.SimulationConfig().steps = 10;
    sim3.setPopulationData(pop_in3);

    bool has_error_1 = false, has_error_2 = false, has_error_3 = false;
    std::thread thread1(runSim, std::ref(sim1), std::ref(has_error_1));
    std::thread thread2(runSim, std::ref(sim2), std::ref(has_error_2));
    std::thread thread3(runSim, std::ref(sim3), std::ref(has_error_3));
    // synchronize threads:
    thread1.join();
    thread2.join();
    thread3.join();
    // Check exceptions
    ASSERT_FALSE(has_error_1);
    ASSERT_FALSE(has_error_2);
    ASSERT_FALSE(has_error_3);
    // Check results
    AgentVector pop1(a, POP_SIZE);
    AgentVector pop2(a, POP_SIZE);
    AgentVector pop3(a, POP_SIZE);
    sim1.getPopulationData(pop1);
    sim2.getPopulationData(pop2);
    sim3.getPopulationData(pop3);
    // Use expect, rather than assert for first 3 agents.
    for (unsigned int i = 0; i < 3; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 10 steps with 1x +4 each.
        EXPECT_EQ(x1, 41);
        EXPECT_EQ(x2, 402);
        EXPECT_EQ(x3, 4003);
    }
    for (unsigned int i = 3; i < POP_SIZE; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 10 steps with 1x +4 each.
        ASSERT_EQ(x1, 41);
        ASSERT_EQ(x2, 402);
        ASSERT_EQ(x3, 4003);
    }
}
FLAMEGPU_AGENT_FUNCTION(SlowFn3, MessageNone, MessageNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (int i = 0; i < FLAMEGPU->environment.getProperty<int>("ten thousand"); ++i) {
        x += FLAMEGPU->environment.getProperty<int>("zero");
    }
    FLAMEGPU->setVariable<int>("x", x + 1);
    FLAMEGPU->agent_out.setVariable<int>("x", 0);
    return ALIVE;
}
TEST(MultiThreadDeviceTest, SameModelSeperateThread_AgentOutput) {
    const unsigned int POP_SIZE = 1000;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent output
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn3).setAgentOutput(a);
    a.newFunction(FUNCTION_NAME2, FastFn);
    m.newLayer().addAgentFunction(SlowFn3);
    m.newLayer().addAgentFunction(FastFn);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);

    AgentVector pop_in1(a, POP_SIZE);
    AgentVector pop_in2(a, POP_SIZE);
    AgentVector pop_in3(a, POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop_in1[i].setVariable<int>("x", 1);
        pop_in2[i].setVariable<int>("x", 2);
        pop_in3[i].setVariable<int>("x", 3);
    }

    CUDASimulation sim1(m);
    sim1.SimulationConfig().steps = 5;
    sim1.setPopulationData(pop_in1);
    CUDASimulation sim2(m);
    sim2.SimulationConfig().steps = 5;
    sim2.setPopulationData(pop_in2);
    CUDASimulation sim3(m);
    sim3.SimulationConfig().steps = 5;
    sim3.setPopulationData(pop_in3);

    bool has_error_1 = false, has_error_2 = false, has_error_3 = false;
    std::thread thread1(runSim, std::ref(sim1), std::ref(has_error_1));
    std::thread thread2(runSim, std::ref(sim2), std::ref(has_error_2));
    std::thread thread3(runSim, std::ref(sim3), std::ref(has_error_3));
    // synchronize threads:
    thread1.join();
    thread2.join();
    thread3.join();
    // Check exceptions
    ASSERT_FALSE(has_error_1);
    ASSERT_FALSE(has_error_2);
    ASSERT_FALSE(has_error_3);
    // Check results
    // 1000 * 2^5
    AgentVector pop1(a);
    AgentVector pop2(a);
    AgentVector pop3(a);
    sim1.getPopulationData(pop1);
    sim2.getPopulationData(pop2);
    sim3.getPopulationData(pop3);
    EXPECT_EQ(pop1.size(), 32 * POP_SIZE);
    EXPECT_EQ(pop2.size(), 32 * POP_SIZE);
    EXPECT_EQ(pop3.size(), 32 * POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 5 steps with 2x +1 each.
        ASSERT_EQ(x1, 11);
        ASSERT_EQ(x2, 12);
        ASSERT_EQ(x3, 13);
    }
    // Check all the new agents
    for (int step = 0; step < 5; ++step) {
        const unsigned int initial_pop = POP_SIZE * static_cast<unsigned int>(pow(2, step));
        const unsigned int next_pop = POP_SIZE * static_cast<unsigned int>(pow(2, step+1));
        for (unsigned int i = initial_pop; i < next_pop; ++i) {
            int x1 = pop1[i].getVariable<int>("x");
            int x2 = pop2[i].getVariable<int>("x");
            int x3 = pop3[i].getVariable<int>("x");
            ASSERT_EQ(x1, 9 - (step * 2));
            ASSERT_EQ(x2, 9 - (step * 2));
            ASSERT_EQ(x3, 9 - (step * 2));
        }
    }
}
FLAMEGPU_AGENT_FUNCTION_CONDITION(AllowEvenOnly) {
    return FLAMEGPU->getVariable<int>("x")%2 == 0;
}
TEST(MultiThreadDeviceTest, SameModelSeperateThread_AgentFunctionCondition) {
    const unsigned int POP_SIZE = 10000;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get within agent function conditions
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn).setFunctionCondition(AllowEvenOnly);
    a.newFunction(FUNCTION_NAME2, FastFn);
    m.newLayer().addAgentFunction(SlowFn);
    m.newLayer().addAgentFunction(FastFn);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);

    AgentVector pop_in1(a, POP_SIZE);
    AgentVector pop_in2(a, POP_SIZE);
    AgentVector pop_in3(a, POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop_in1[i].setVariable<int>("x", 1);
        pop_in2[i].setVariable<int>("x", 2);
        pop_in3[i].setVariable<int>("x", 3);
    }

    CUDASimulation sim1(m);
    sim1.SimulationConfig().steps = 10;
    sim1.setPopulationData(pop_in1);
    CUDASimulation sim2(m);
    sim2.SimulationConfig().steps = 10;
    sim2.setPopulationData(pop_in2);
    CUDASimulation sim3(m);
    sim3.SimulationConfig().steps = 10;
    sim3.setPopulationData(pop_in3);

    bool has_error_1 = false, has_error_2 = false, has_error_3 = false;
    std::thread thread1(runSim, std::ref(sim1), std::ref(has_error_1));
    std::thread thread2(runSim, std::ref(sim2), std::ref(has_error_2));
    std::thread thread3(runSim, std::ref(sim3), std::ref(has_error_3));
    // synchronize threads:
    thread1.join();
    thread2.join();
    thread3.join();
    // Check exceptions
    ASSERT_FALSE(has_error_1);
    ASSERT_FALSE(has_error_2);
    ASSERT_FALSE(has_error_3);
    // Check results
    AgentVector pop1(a, POP_SIZE);
    AgentVector pop2(a, POP_SIZE);
    AgentVector pop3(a, POP_SIZE);
    sim1.getPopulationData(pop1);
    sim2.getPopulationData(pop2);
    sim3.getPopulationData(pop3);
    // Use expect, rather than assert for first 3 agents.
    for (unsigned int i = 0; i < 3; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 1, +1, +2*9
        // 2, +2*10
        // 3, +1, +2*9
        EXPECT_EQ(x1, 20);
        EXPECT_EQ(x2, 22);
        EXPECT_EQ(x3, 22);
    }
    for (unsigned int i = 3; i < POP_SIZE; ++i) {
        int x1 = pop1[i].getVariable<int>("x");
        int x2 = pop2[i].getVariable<int>("x");
        int x3 = pop3[i].getVariable<int>("x");
        // 5 steps with 1x +1, 1x +2 each.
        ASSERT_EQ(x1, 20);
        ASSERT_EQ(x2, 22);
        ASSERT_EQ(x3, 22);
    }
}
void initRunSim(std::shared_ptr<CUDASimulation> sim, const AgentDescription &a, int offset, int device, unsigned int POP_SIZE, int &exception_thrown, int steps = 10) {
    exception_thrown = 0;
    try {
        AgentVector pop_in(a, POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop_in[i].setVariable<int>("x", offset);
        }
        sim->SimulationConfig().steps = steps;
        sim->CUDAConfig().device_id = device;
        sim->applyConfig();
        sim->setPopulationData(pop_in);
        sim->simulate();
    } catch(std::exception &e) {
        exception_thrown = 1;
        printf("%s\n", e.what());
    }
}
TEST(MultiThreadDeviceTest, SameModelMultiDevice_Agent) {
    const unsigned int POP_SIZE = 10000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 10;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn);
    a.newFunction(FUNCTION_NAME2, FastFn);
    m.newLayer().addAgentFunction(SlowFn);
    m.newLayer().addAgentFunction(FastFn);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);

    int devices = 0;
    if (cudaSuccess != cudaGetDeviceCount(&devices) || devices <= 0) {
        // Skip the test, if no CUDA or GPUs.
        return;
    }
    std::vector<std::shared_ptr<CUDASimulation>> sims;
    sims.reserve(devices * 3);
    std::vector<int> results;
    results.reserve(devices * 3);
    std::vector<std::thread> threads;
    int offset = 0;
    // For each device
    for (int device = 0; device < devices; ++device) {
        // If built with a suitable compute capability
        if (util::detail::compute_capability::checkComputeCapability(device)) {
            for (int i = 0; i < SIMS_PER_DEVICE; ++i) {
                // Set sim Running
                sims.emplace(sims.end(), std::make_shared<CUDASimulation>(m));
                results.push_back(false);
                auto thread = std::thread(initRunSim, std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]), STEPS);
                threads.push_back(std::move(thread));
                // Use this version for debugging without threads
                // initRunSim(std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]));
                offset++;
            }
        }
    }
    // Wait for all to finish
    for (auto &th : threads) {
        th.join();
    }
    AgentVector pop(a, POP_SIZE);
    // Check results
    for (unsigned int i = 0; i < results.size(); ++i) {
        // Check exceptions
        ASSERT_FALSE(results[i]);
        // Get agent data
        ASSERT_EQ(cudaSetDevice(sims[i]->CUDAConfig().device_id), cudaSuccess);
        sims[i]->getPopulationData(pop);
        for (unsigned int j = 0; j < POP_SIZE; ++j) {
            int x = pop[j].getVariable<int>("x");
            ASSERT_EQ(x, static_cast<int>(2 * STEPS + i));
        }
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}
TEST(MultiThreadDeviceTest, SameModelMultiDevice_Message) {
    const unsigned int POP_SIZE = 10000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 10;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    MessageBruteForce::Description &message = m.newMessage(MESSAGE_NAME);
    message.newVariable<int>("x");
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, FastFnMessage).setMessageOutput(message);
    a.newFunction(FUNCTION_NAME2, SlowFnMessage).setMessageInput(message);
    m.newLayer().addAgentFunction(FastFnMessage);
    m.newLayer().addAgentFunction(SlowFnMessage);

    int devices = 0;
    if (cudaSuccess != cudaGetDeviceCount(&devices) || devices <= 0) {
        // Skip the test, if no CUDA or GPUs.
        return;
    }
    std::vector<std::shared_ptr<CUDASimulation>> sims;
    sims.reserve(devices * 3);
    std::vector<int> results;
    results.reserve(devices * 3);
    std::vector<std::thread> threads;
    int offset = 0;
    // For each device
    for (int device = 0; device < devices; ++device) {
        // If built with a suitable compute capability
        if (util::detail::compute_capability::checkComputeCapability(device)) {
            for (int i = 0; i < SIMS_PER_DEVICE; ++i) {
                // Set sim Running
                sims.emplace(sims.end(), std::make_shared<CUDASimulation>(m));
                results.push_back(false);
                auto thread = std::thread(initRunSim, std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]), STEPS);
                threads.push_back(std::move(thread));
                // Use this version for debugging without threads
                // initRunSim(std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]));
                offset++;
            }
        }
    }
    // Wait for all to finish
    for (auto &th : threads) {
        th.join();
    }
    AgentVector pop(a, POP_SIZE);
    // Check results
    for (unsigned int i = 0; i < results.size(); ++i) {
        // Check exceptions
        ASSERT_FALSE(results[i]);
        // Get agent data
        ASSERT_EQ(cudaSetDevice(sims[i]->CUDAConfig().device_id), cudaSuccess);
        sims[i]->getPopulationData(pop);
        for (unsigned int j = 0; j < POP_SIZE; ++j) {
            int x = pop[j].getVariable<int>("x");
            ASSERT_EQ(x, static_cast<int>(POP_SIZE * STEPS + i));
        }
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}
TEST(MultiThreadDeviceTest, SameModelMultiDevice_Environment) {
    const unsigned int POP_SIZE = 10000;
    const int SIMS_PER_DEVICE = 10;
    const int STEPS = 10;
    const int MAX_DEVICES = 4;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn2);
    a.newFunction(FUNCTION_NAME2, FastFn2);
    m.newLayer().addAgentFunction(SlowFn2);
    m.newLayer().addAgentFunction(FastFn2);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);
    m.Environment().newProperty<int>("one", 1);
    m.Environment().newProperty<int>("three", 3);

    int devices = 0;
    if (cudaSuccess != cudaGetDeviceCount(&devices) || devices <= 0) {
        // Skip the test, if no CUDA or GPUs.
        return;
    }
    devices = devices > MAX_DEVICES ? MAX_DEVICES : devices;
    // BEGIN: Attempt to pre init contexts
    for (int device = 0; device < devices; ++device) {
        ASSERT_EQ(cudaSetDevice(device), cudaSuccess);
        ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
    // END: Attempt to pre init contexts
    std::vector<std::shared_ptr<CUDASimulation>> sims;
    sims.reserve(devices * SIMS_PER_DEVICE);
    std::vector<int> results;
    results.reserve(devices * SIMS_PER_DEVICE);
    std::vector<std::thread> threads;
    int offset = 0;
    // For each device
    for (int device = 0; device < devices; ++device) {
        // If built with a suitable compute capability
        if (util::detail::compute_capability::checkComputeCapability(device)) {
            for (int i = 0; i < SIMS_PER_DEVICE; ++i) {
                // Set sim Running
                m.Environment().setProperty<int>("one", 1 * (offset + 1));
                m.Environment().setProperty<int>("three", 3 * (offset + 1));
                sims.emplace(sims.end(), std::make_shared<CUDASimulation>(m));
                results.push_back(false);
                auto thread = std::thread(initRunSim, std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]), STEPS);
                threads.push_back(std::move(thread));
                // Use this version for debugging without threads
                // initRunSim(std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]));
                offset++;
            }
        }
    }
    // Wait for all to finish
    for (auto &th : threads) {
        th.join();
    }
    AgentVector pop(a, POP_SIZE);
    // Check results
    for (unsigned int i = 0; i < results.size(); ++i) {
        // Check exceptions
        ASSERT_FALSE(results[i]);
        // Get agent data
        ASSERT_EQ(cudaSetDevice(sims[i]->CUDAConfig().device_id), cudaSuccess);
        sims[i]->getPopulationData(pop);
        int bad = 0;
        int x = static_cast<int>(4 * STEPS * (i + 1) + i);
        for (unsigned int j = 0; j < POP_SIZE; ++j) {
            x = pop[j].getVariable<int>("x");
            if (x != static_cast<int>(4 * STEPS * (i + 1) + i)) {
                bad++;
            }
        }
        if (bad > 0) {
            printf("Device: %d, Thread: %d: %d failures.   Expected: %d, Received: %d\n", sims[i]->CUDAConfig().device_id, i%SIMS_PER_DEVICE, bad, static_cast<int>(4 * STEPS * (i + 1) + i), x);
        }
    }
    for (unsigned int i = 0; i < results.size(); ++i) {
        // Check exceptions
        ASSERT_FALSE(results[i]);
        // Get agent data
        ASSERT_EQ(cudaSetDevice(sims[i]->CUDAConfig().device_id), cudaSuccess);
        sims[i]->getPopulationData(pop);
        for (unsigned int j = 0; j < POP_SIZE; ++j) {
            int x = pop[j].getVariable<int>("x");
            ASSERT_EQ(x, static_cast<int>(4 * STEPS * (i + 1) + i));
        }
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}
TEST(MultiThreadDeviceTest, SameModelMultiDevice_AgentOutput) {
    const unsigned int POP_SIZE = 1000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 5;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn3).setAgentOutput(a);
    a.newFunction(FUNCTION_NAME2, FastFn);
    m.newLayer().addAgentFunction(SlowFn3);
    m.newLayer().addAgentFunction(FastFn);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);

    int devices = 0;
    if (cudaSuccess != cudaGetDeviceCount(&devices) || devices <= 0) {
        // Skip the test, if no CUDA or GPUs.
        return;
    }
    std::vector<std::shared_ptr<CUDASimulation>> sims;
    sims.reserve(devices * 3);
    std::vector<int> results;
    results.reserve(devices * 3);
    std::vector<std::thread> threads;
    int offset = 0;
    // For each device
    for (int device = 0; device < devices; ++device) {
        // If built with a suitable compute capability
        if (util::detail::compute_capability::checkComputeCapability(device)) {
            for (int i = 0; i < SIMS_PER_DEVICE; ++i) {
                // Set sim Running
                sims.emplace(sims.end(), std::make_shared<CUDASimulation>(m));
                results.push_back(false);
                auto thread = std::thread(initRunSim, std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]), STEPS);
                threads.push_back(std::move(thread));
                // Use this version for debugging without threads
                // initRunSim(std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]));
                offset++;
            }
        }
    }
    // Wait for all to finish
    for (auto &th : threads) {
        th.join();
    }
    AgentVector pop(a, POP_SIZE);
    // Check results
    for (unsigned int i = 0; i < results.size(); ++i) {
        // Check exceptions
        ASSERT_FALSE(results[i]);
        // Get agent data
        ASSERT_EQ(cudaSetDevice(sims[i]->CUDAConfig().device_id), cudaSuccess);
        sims[i]->getPopulationData(pop);
        for (unsigned int j = 0; j < POP_SIZE; ++j) {
            int x = pop[j].getVariable<int>("x");
            // 5 steps with 2x +1 each.
            ASSERT_EQ(x, static_cast<int>(10 + i));
        }
        // Check all the new agents
        for (int step = 0; step < STEPS; ++step) {
            const unsigned int initial_pop = POP_SIZE * static_cast<unsigned int>(pow(2, step));
            const unsigned int next_pop = POP_SIZE * static_cast<unsigned int>(pow(2, step+1));
            for (unsigned int j = initial_pop; j < next_pop; ++j) {
                int x = pop[j].getVariable<int>("x");
                ASSERT_EQ(x, 9 - (step * 2));
            }
        }
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}
TEST(MultiThreadDeviceTest, SameModelMultiDevice_AgentFunctionCondition) {
    const unsigned int POP_SIZE = 10000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 10;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    a.newFunction(FUNCTION_NAME1, SlowFn).setFunctionCondition(AllowEvenOnly);
    a.newFunction(FUNCTION_NAME2, FastFn);
    m.newLayer().addAgentFunction(SlowFn);
    m.newLayer().addAgentFunction(FastFn);

    m.Environment().newProperty<int>("ten thousand", 10000);
    m.Environment().newProperty<int>("zero", 0);

    int devices = 0;
    if (cudaSuccess != cudaGetDeviceCount(&devices) || devices <= 0) {
        // Skip the test, if no CUDA or GPUs.
        return;
    }
    std::vector<std::shared_ptr<CUDASimulation>> sims;
    sims.reserve(devices * 3);
    std::vector<int> results;
    results.reserve(devices * 3);
    std::vector<std::thread> threads;
    int offset = 0;
    // For each device
    for (int device = 0; device < devices; ++device) {
        // If built with a suitable compute capability
        if (util::detail::compute_capability::checkComputeCapability(device)) {
            for (int i = 0; i < SIMS_PER_DEVICE; ++i) {
                // Set sim Running
                sims.emplace(sims.end(), std::make_shared<CUDASimulation>(m));
                results.push_back(false);
                auto thread = std::thread(initRunSim, std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]), STEPS);
                threads.push_back(std::move(thread));
                // Use this version for debugging without threads
                // initRunSim(std::ref(sims[offset]), std::ref(a), offset, device, POP_SIZE, std::ref(results[offset]));
                offset++;
            }
        }
    }
    // Wait for all to finish
    for (auto &th : threads) {
        th.join();
    }
    AgentVector pop(a, POP_SIZE);
    // Check results
    for (unsigned int i = 0; i < results.size(); ++i) {
        // Check exceptions
        ASSERT_FALSE(results[i]);
        // Get agent data
        cudaSetDevice(sims[i]->CUDAConfig().device_id);
        sims[i]->getPopulationData(pop);
        for (unsigned int j = 0; j < POP_SIZE; ++j) {
            int x = pop[j].getVariable<int>("x");
            // 0, +2*10
            // 1, +1, +2*9
            // 2, +2*10
            // 3, +1, +2*9
            if (i%2 == 0) {
                ASSERT_EQ(x, static_cast<int>(2 * STEPS + i));
            } else {
                ASSERT_EQ(x, static_cast<int>(2 * STEPS - 1 + i));
            }
        }
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}
}  // namespace test_multi_thread_device
}  // namespace flamegpu
