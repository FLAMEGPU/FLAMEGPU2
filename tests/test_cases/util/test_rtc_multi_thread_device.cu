#include <thread>
#include <vector>
#include <memory>

#include "flamegpu/flame_api.h"
#include "gtest/gtest.h"
#include "flamegpu/util/compute_capability.cuh"

namespace test_rtc_multi_thread_device {
const char *MODEL_NAME = "Model";
const char *AGENT_NAME = "Agent1";
const char *MESSAGE_NAME = "Message1";
const char *FUNCTION_NAME1 = "Fn1";
const char *FUNCTION_NAME2 = "Fn2";
const char* rtc_SlowFn = R"###(
FLAMEGPU_AGENT_FUNCTION(SlowFn, MsgNone, MsgNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (int i = 0; i < FLAMEGPU->environment.getProperty<int>("ten thousand"); ++i) {
        x += FLAMEGPU->environment.getProperty<int>("zero");
    }
    FLAMEGPU->setVariable<int>("x", x + 1);
    return ALIVE;
}
)###";
const char* rtc_FastFn = R"###(
FLAMEGPU_AGENT_FUNCTION(FastFn, MsgNone, MsgNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + 1);
    return ALIVE;
}
)###";
const char* rtc_SlowFnMsg = R"###(
FLAMEGPU_AGENT_FUNCTION(SlowFnMsg, MsgBruteForce, MsgNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    int y = 0;
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (auto &m : FLAMEGPU->message_in) {
        y += m.getVariable<int>("x");
    }
    FLAMEGPU->setVariable<int>("x", x + y);
    return ALIVE;
}
)###";
const char* rtc_FastFnMsg = R"###(
FLAMEGPU_AGENT_FUNCTION(FastFnMsg, MsgNone, MsgBruteForce) {
    const int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->message_out.setVariable<int>("x", 1);
    return ALIVE;
}
)###";
const char* rtc_SlowFn2 = R"###(
FLAMEGPU_AGENT_FUNCTION(SlowFn2, MsgNone, MsgNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (int i = 0; i < FLAMEGPU->environment.getProperty<int>("ten thousand"); ++i) {
        x += FLAMEGPU->environment.getProperty<int>("zero");
    }
    FLAMEGPU->setVariable<int>("x", x + FLAMEGPU->environment.getProperty<int>("one"));
    return ALIVE;
}
)###";
const char* rtc_FastFn2 = R"###(
FLAMEGPU_AGENT_FUNCTION(FastFn2, MsgNone, MsgNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + FLAMEGPU->environment.getProperty<int>("three"));
    return ALIVE;
}
)###";
const char* rtc_SlowFn3 = R"###(
FLAMEGPU_AGENT_FUNCTION(SlowFn3, MsgNone, MsgNone) {
    int x = FLAMEGPU->getVariable<int>("x");
    // Do nothing, just waste time. Get values from environment to prevent optimisation
    for (int i = 0; i < FLAMEGPU->environment.getProperty<int>("ten thousand"); ++i) {
        x += FLAMEGPU->environment.getProperty<int>("zero");
    }
    FLAMEGPU->setVariable<int>("x", x + 1);
    FLAMEGPU->agent_out.setVariable<int>("x", 0);
    return ALIVE;
}
)###";
const char* rtc_AllowEvenOnly = R"###(
FLAMEGPU_AGENT_FUNCTION_CONDITION(AllowEvenOnly) {
    return FLAMEGPU->getVariable<int>("x")%2 == 0;
}
)###";
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
TEST(RTCMultiThreadDeviceTest, SameModelMultiDevice_Message) {
    const unsigned int POP_SIZE = 10000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 10;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    MsgBruteForce::Description &msg = m.newMessage(MESSAGE_NAME);
    msg.newVariable<int>("x");
    a.newVariable<int>("x", 0);
    auto &fn1 = a.newRTCFunction(FUNCTION_NAME1, rtc_FastFnMsg);
    auto &fn2 = a.newRTCFunction(FUNCTION_NAME2, rtc_SlowFnMsg);
    fn1.setMessageOutput(msg);
    fn2.setMessageInput(msg);
    m.newLayer().addAgentFunction(fn1);
    m.newLayer().addAgentFunction(fn2);

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
        if (util::compute_capability::checkComputeCapability(device)) {
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
TEST(RTCMultiThreadDeviceTest, SameModelMultiDevice_Environment) {
    const unsigned int POP_SIZE = 10000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 10;
    const int MAX_DEVICES = 4;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    auto &fn1 = a.newRTCFunction(FUNCTION_NAME1, rtc_SlowFn2);
    auto &fn2 = a.newRTCFunction(FUNCTION_NAME2, rtc_FastFn2);
    m.newLayer().addAgentFunction(fn1);
    m.newLayer().addAgentFunction(fn2);

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
        if (util::compute_capability::checkComputeCapability(device)) {
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
        for (AgentVector::Agent ai : pop) {
            x = ai.getVariable<int>("x");
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
        for (AgentVector::Agent ai : pop) {
            int x = ai.getVariable<int>("x");
            ASSERT_EQ(x, static_cast<int>(4 * STEPS * (i + 1) + i));
        }
    }
    ASSERT_EQ(cudaSetDevice(0), cudaSuccess);
}
TEST(RTCMultiThreadDeviceTest, SameModelMultiDevice_AgentOutput) {
    const unsigned int POP_SIZE = 1000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 5;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    auto &fn1 = a.newRTCFunction(FUNCTION_NAME1, rtc_SlowFn3);
    auto &fn2 = a.newRTCFunction(FUNCTION_NAME2, rtc_FastFn);
    fn1.setAgentOutput(a);
    m.newLayer().addAgentFunction(fn1);
    m.newLayer().addAgentFunction(fn2);

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
        if (util::compute_capability::checkComputeCapability(device)) {
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
TEST(RTCMultiThreadDeviceTest, SameModelMultiDevice_AgentFunctionCondition) {
    const unsigned int POP_SIZE = 10000;
    const int SIMS_PER_DEVICE = 3;
    const int STEPS = 10;
    // The purpose of this test is to try and catch whether CURVE will hit collisions if two identical models execute at the same time.
    // Success of this test does not mean there isn't a problem
    // This test covers agent variable set/get

    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<int>("x", 0);
    auto &fn1 = a.newRTCFunction(FUNCTION_NAME1, rtc_SlowFn);
    auto &fn2 = a.newRTCFunction(FUNCTION_NAME2, rtc_FastFn);
    fn1.setRTCFunctionCondition(rtc_AllowEvenOnly);
    m.newLayer().addAgentFunction(fn1);
    m.newLayer().addAgentFunction(fn2);

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
        if (util::compute_capability::checkComputeCapability(device)) {
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
        for (AgentVector::Agent ai : pop) {
            int x = ai.getVariable<int>("x");
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
}  // namespace test_rtc_multi_thread_device
