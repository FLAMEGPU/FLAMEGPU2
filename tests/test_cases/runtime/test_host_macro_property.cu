/**
 * Tests of class: HostMacroProperty
 * ReadTest: Test that HostMacroProperty can read data (written via agent fn's)
 * WriteTest: Test that HostMacroProperty can write data (read via agent fn's)
 * ZeroTest: Test that HostMacroProperty can zero data (read via agent fn's)
 */

#include <array>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace {
const unsigned int TEST_DIMS[4] = {2, 3, 4, 5};
FLAMEGPU_STEP_FUNCTION(HostRead) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    unsigned int a = 0;
    for (unsigned int i = 0; i < TEST_DIMS[0]; ++i) {
        for (unsigned int j = 0; j < TEST_DIMS[1]; ++j) {
            for (unsigned int k = 0; k < TEST_DIMS[2]; ++k) {
                for (unsigned int w = 0; w < TEST_DIMS[3]; ++w) {
                    ASSERT_EQ(t[i][j][k][w], a++);
                }
            }
        }
    }
}
FLAMEGPU_AGENT_FUNCTION(AgentWrite, MessageNone, MessageNone) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    const unsigned int i = FLAMEGPU->getVariable<unsigned int>("i");
    const unsigned int j = FLAMEGPU->getVariable<unsigned int>("j");
    const unsigned int k = FLAMEGPU->getVariable<unsigned int>("k");
    const unsigned int w = FLAMEGPU->getVariable<unsigned int>("w");
    t[i][j][k][w].exchange(FLAMEGPU->getVariable<unsigned int>("a"));
    return flamegpu::ALIVE;
}
TEST(HostMacroPropertyTest, ReadTest) {
    // Fill MacroProperty with DeviceAPI
    // Test values match expected value with HostAPI and test in-place
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("i");
    agent.newVariable<unsigned int>("j");
    agent.newVariable<unsigned int>("k");
    agent.newVariable<unsigned int>("w");
    agent.newVariable<unsigned int>("a");
    agent.newFunction("agentwrite", AgentWrite);
    model.newLayer().addAgentFunction(AgentWrite);
    model.newLayer().addHostFunction(HostRead);
    const unsigned int total_agents = TEST_DIMS[0] * TEST_DIMS[1] * TEST_DIMS[2] * TEST_DIMS[3];
    AgentVector population(agent, total_agents);
    unsigned int a = 0;
    for (unsigned int i = 0; i < TEST_DIMS[0]; ++i) {
        for (unsigned int j = 0; j < TEST_DIMS[1]; ++j) {
            for (unsigned int k = 0; k < TEST_DIMS[2]; ++k) {
                for (unsigned int w = 0; w < TEST_DIMS[3]; ++w) {
                    auto p = population[a];
                    p.setVariable<unsigned int>("a", a++);
                    p.setVariable<unsigned int>("i", i);
                    p.setVariable<unsigned int>("j", j);
                    p.setVariable<unsigned int>("k", k);
                    p.setVariable<unsigned int>("w", w);
                }
            }
        }
    }
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
}
FLAMEGPU_STEP_FUNCTION(HostWrite) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    unsigned int a = 0;
    for (unsigned int i = 0; i < TEST_DIMS[0]; ++i) {
        for (unsigned int j = 0; j < TEST_DIMS[1]; ++j) {
            for (unsigned int k = 0; k < TEST_DIMS[2]; ++k) {
                for (unsigned int w = 0; w < TEST_DIMS[3]; ++w) {
                    t[i][j][k][w] = 1+a++;
                }
            }
        }
    }
}
FLAMEGPU_AGENT_FUNCTION(AgentRead, MessageNone, MessageNone) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    const unsigned int i = FLAMEGPU->getVariable<unsigned int>("i");
    const unsigned int j = FLAMEGPU->getVariable<unsigned int>("j");
    const unsigned int k = FLAMEGPU->getVariable<unsigned int>("k");
    const unsigned int w = FLAMEGPU->getVariable<unsigned int>("w");
    if (2 + t[i][j][k][w] == FLAMEGPU->getVariable<unsigned int>("a")+1) {
        FLAMEGPU->setVariable<unsigned int>("a", 12);
    } else {
        FLAMEGPU->setVariable<unsigned int>("a", 0);
    }
    return flamegpu::ALIVE;
}
TEST(HostMacroPropertyTest, WriteTest) {
    // Fill MacroProperty with HostAPI
    // Test values match expected value with DeviceAPI
    // Write results back to agent variable, and check at the end
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    model.Environment().newMacroProperty<unsigned int>("plusequal");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("i");
    agent.newVariable<unsigned int>("j");
    agent.newVariable<unsigned int>("k");
    agent.newVariable<unsigned int>("w");
    agent.newVariable<unsigned int>("a");
    agent.newFunction("agentread", AgentRead);
    model.newLayer().addHostFunction(HostWrite);
    model.newLayer().addAgentFunction(AgentRead);
    const unsigned int total_agents = TEST_DIMS[0] * TEST_DIMS[1] * TEST_DIMS[2] * TEST_DIMS[3];
    AgentVector population(agent, total_agents);
    unsigned int a = 0;
    for (unsigned int i = 0; i < TEST_DIMS[0]; ++i) {
        for (unsigned int j = 0; j < TEST_DIMS[1]; ++j) {
            for (unsigned int k = 0; k < TEST_DIMS[2]; ++k) {
                for (unsigned int w = 0; w < TEST_DIMS[3]; ++w) {
                    auto p = population[a];
                    p.setVariable<unsigned int>("a", 2+a++);
                    p.setVariable<unsigned int>("i", i);
                    p.setVariable<unsigned int>("j", j);
                    p.setVariable<unsigned int>("k", k);
                    p.setVariable<unsigned int>("w", w);
                }
            }
        }
    }
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    cudaSimulation.getPopulationData(population);
    // Check results
    unsigned int correct = 0;
    for (auto p : population) {
        correct += p.getVariable<unsigned int>("a") == 12 ? 1 : 0;
    }
    ASSERT_EQ(correct, total_agents);
}
FLAMEGPU_STEP_FUNCTION(HostZero) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    t.zero();
}
FLAMEGPU_AGENT_FUNCTION(AgentReadZero, MessageNone, MessageNone) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    const unsigned int i = FLAMEGPU->getVariable<unsigned int>("i");
    const unsigned int j = FLAMEGPU->getVariable<unsigned int>("j");
    const unsigned int k = FLAMEGPU->getVariable<unsigned int>("k");
    const unsigned int w = FLAMEGPU->getVariable<unsigned int>("w");
    if (t[i][j][k][w] == 0) {
        FLAMEGPU->setVariable<unsigned int>("a", 1);
    } else {
        FLAMEGPU->setVariable<unsigned int>("a", 0);
    }
    return flamegpu::ALIVE;
}
TEST(HostMacroPropertyTest, ZeroTest) {
    // Fill MacroProperty with HostAPI
    // Test values match expected value with DeviceAPI
    // Write results back to agent variable, and check at the end
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("i");
    agent.newVariable<unsigned int>("j");
    agent.newVariable<unsigned int>("k");
    agent.newVariable<unsigned int>("w");
    agent.newVariable<unsigned int>("a");
    agent.newFunction("agentwrite", AgentWrite);
    agent.newFunction("agentread", AgentReadZero);
    model.newLayer().addAgentFunction(AgentWrite);
    model.newLayer().addHostFunction(HostZero);
    model.newLayer().addAgentFunction(AgentReadZero);
    const unsigned int total_agents = TEST_DIMS[0] * TEST_DIMS[1] * TEST_DIMS[2] * TEST_DIMS[3];
    AgentVector population(agent, total_agents);
    unsigned int a = 0;
    for (unsigned int i = 0; i < TEST_DIMS[0]; ++i) {
        for (unsigned int j = 0; j < TEST_DIMS[1]; ++j) {
            for (unsigned int k = 0; k < TEST_DIMS[2]; ++k) {
                for (unsigned int w = 0; w < TEST_DIMS[3]; ++w) {
                    auto p = population[a++];
                    p.setVariable<unsigned int>("i", i);
                    p.setVariable<unsigned int>("j", j);
                    p.setVariable<unsigned int>("k", k);
                    p.setVariable<unsigned int>("w", w);
                }
            }
        }
    }
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    cudaSimulation.getPopulationData(population);
    // Check results
    unsigned int correct = 0;
    for (auto p : population) {
        correct += p.getVariable<unsigned int>("a") == 1 ? 1 : 0;
    }
    ASSERT_EQ(correct, total_agents);
}
FLAMEGPU_STEP_FUNCTION(HostArithmeticInit) {
    FLAMEGPU->environment.getMacroProperty<int>("int") = 10;
    FLAMEGPU->environment.getMacroProperty<unsigned int>("uint") = 10u;
    FLAMEGPU->environment.getMacroProperty<int8_t>("int8") = 10;
    FLAMEGPU->environment.getMacroProperty<uint8_t>("uint8") = 10u;
    FLAMEGPU->environment.getMacroProperty<int64_t>("int64") = 10;
    FLAMEGPU->environment.getMacroProperty<uint64_t>("uint64") = 10u;
    FLAMEGPU->environment.getMacroProperty<flamegpu::id_t>("id") = 10u;
    FLAMEGPU->environment.getMacroProperty<float>("float") = 10.0f;
    FLAMEGPU->environment.getMacroProperty<double>("double") = 10.0;
}
FLAMEGPU_STEP_FUNCTION(HostArithmetic) {
    {
        auto t = FLAMEGPU->environment.getMacroProperty<int>("int");
        EXPECT_EQ(t++, 10);
        EXPECT_EQ(++t, 12);
        EXPECT_EQ(t--, 12);
        EXPECT_EQ(--t, 10);
        EXPECT_EQ(t / 5, 10 / 5);
        EXPECT_EQ(t / 3, 10 / 3);
        EXPECT_EQ(t / 3.0, 10 / 3.0);
        EXPECT_EQ(t + 5, 10 + 5);
        EXPECT_EQ(t + 3.0, 10 + 3.0);
        EXPECT_EQ(t - 3, 10 - 3);
        EXPECT_EQ(t - 3.0, 10 - 3.0);
        EXPECT_EQ(t * 5, 10 * 5);
        EXPECT_EQ(t * 3.0, 10 * 3.0);
        EXPECT_EQ(t % 5, 10 % 5);
        EXPECT_EQ(t % 3, 10 % 3);
        EXPECT_EQ(t, 10);
        t += 10;
        EXPECT_EQ(t, 20);
        t -= 10;
        EXPECT_EQ(t, 10);
        t *= 2;
        EXPECT_EQ(t, 20);
        t /= 2;
        EXPECT_EQ(t, 10);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<unsigned int>("uint");
        EXPECT_EQ(t++, 10u);
        EXPECT_EQ(++t, 12u);
        EXPECT_EQ(t--, 12u);
        EXPECT_EQ(--t, 10u);
        EXPECT_EQ(t / 5, 10u / 5);
        EXPECT_EQ(t / 3, 10u / 3);
        EXPECT_EQ(t / 3.0, 10u / 3.0);
        EXPECT_EQ(t + 5, 10u + 5);
        EXPECT_EQ(t + 3.0, 10u + 3.0);
        EXPECT_EQ(t - 3, 10u - 3);
        EXPECT_EQ(t - 3.0, 10u - 3.0);
        EXPECT_EQ(t * 5, 10u * 5);
        EXPECT_EQ(t * 3.0, 10u * 3.0);
        EXPECT_EQ(t % 5, 10u % 5);
        EXPECT_EQ(t % 3, 10u % 3);
        EXPECT_EQ(t, 10u);
        t += 10;
        EXPECT_EQ(t, 20u);
        t -= 10;
        EXPECT_EQ(t, 10u);
        t *= 2;
        EXPECT_EQ(t, 20u);
        t /= 2;
        EXPECT_EQ(t, 10u);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<int8_t>("int8");
        EXPECT_EQ(t++, 10);
        EXPECT_EQ(++t, 12);
        EXPECT_EQ(t--, 12);
        EXPECT_EQ(--t, 10);
        EXPECT_EQ(t / 5, 10 / 5);
        EXPECT_EQ(t / 3, 10 / 3);
        EXPECT_EQ(t / 3.0, 10 / 3.0);
        EXPECT_EQ(t + 5, 10 + 5);
        EXPECT_EQ(t + 3.0, 10 + 3.0);
        EXPECT_EQ(t - 3, 10 - 3);
        EXPECT_EQ(t - 3.0, 10 - 3.0);
        EXPECT_EQ(t * 5, 10 * 5);
        EXPECT_EQ(t * 3.0, 10 * 3.0);
        EXPECT_EQ(t % 5, 10 % 5);
        EXPECT_EQ(t % 3, 10 % 3);
        EXPECT_EQ(t, 10);
        t += static_cast<int8_t>(10);
        EXPECT_EQ(t, 20);
        t -= static_cast<int8_t>(10);
        EXPECT_EQ(t, 10);
        t *= static_cast<int8_t>(2);
        EXPECT_EQ(t, 20);
        t /= static_cast<int8_t>(2);
        EXPECT_EQ(t, 10);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<uint8_t>("uint8");
        EXPECT_EQ(t++, 10u);
        EXPECT_EQ(++t, 12u);
        EXPECT_EQ(t--, 12u);
        EXPECT_EQ(--t, 10u);
        EXPECT_EQ(t / 5, static_cast<uint8_t>(10u) / 5);
        EXPECT_EQ(t / 3, static_cast<uint8_t>(10u) / 3);
        EXPECT_EQ(t / 3.0, static_cast<uint8_t>(10u) / 3.0);
        EXPECT_EQ(t + 5, static_cast<uint8_t>(10u) + 5);
        EXPECT_EQ(t + 3.0, static_cast<uint8_t>(10u) + 3.0);
        EXPECT_EQ(t - 3, static_cast<uint8_t>(10u) - 3);
        EXPECT_EQ(t - 3.0, static_cast<uint8_t>(10u) - 3.0);
        EXPECT_EQ(t * 5, static_cast<uint8_t>(10u) * 5);
        EXPECT_EQ(t * 3.0, static_cast<uint8_t>(10u) * 3.0);
        EXPECT_EQ(t % 5, static_cast<uint8_t>(10u) % 5);
        EXPECT_EQ(t % 3, static_cast<uint8_t>(10u) % 3);
        EXPECT_EQ(t, 10u);
        t += static_cast<uint8_t>(10u);
        EXPECT_EQ(t, 20u);
        t -= static_cast<uint8_t>(10u);
        EXPECT_EQ(t, 10u);
        t *= static_cast<uint8_t>(2u);
        EXPECT_EQ(t, 20u);
        t /= static_cast<uint8_t>(2u);
        EXPECT_EQ(t, 10u);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<int64_t>("int64");
        EXPECT_EQ(t++, 10);
        EXPECT_EQ(++t, 12);
        EXPECT_EQ(t--, 12);
        EXPECT_EQ(--t, 10);
        EXPECT_EQ(t / 5, 10 / 5);
        EXPECT_EQ(t / 3, 10 / 3);
        EXPECT_EQ(t / 3.0, 10 / 3.0);
        EXPECT_EQ(t + 5, 10 + 5);
        EXPECT_EQ(t + 3.0, 10 + 3.0);
        EXPECT_EQ(t - 3, 10 - 3);
        EXPECT_EQ(t - 3.0, 10 - 3.0);
        EXPECT_EQ(t * 5, 10 * 5);
        EXPECT_EQ(t * 3.0, 10 * 3.0);
        EXPECT_EQ(t % 5, 10 % 5);
        EXPECT_EQ(t % 3, 10 % 3);
        EXPECT_EQ(t, 10);
        t += 10;
        EXPECT_EQ(t, 20);
        t -= 10;
        EXPECT_EQ(t, 10);
        t *= 2;
        EXPECT_EQ(t, 20);
        t /= 2;
        EXPECT_EQ(t, 10);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<uint64_t>("uint64");
        EXPECT_EQ(t++, 10u);
        EXPECT_EQ(++t, 12u);
        EXPECT_EQ(t--, 12u);
        EXPECT_EQ(--t, 10u);
        EXPECT_EQ(t / 5, 10u / 5);
        EXPECT_EQ(t / 3, 10u / 3);
        EXPECT_EQ(t / 3.0, 10u / 3.0);
        EXPECT_EQ(t + 5, 10u + 5);
        EXPECT_EQ(t + 3.0, 10u + 3.0);
        EXPECT_EQ(t - 3, 10u - 3);
        EXPECT_EQ(t - 3.0, 10u - 3.0);
        EXPECT_EQ(t * 5, 10u * 5);
        EXPECT_EQ(t * 3.0, 10u * 3.0);
        EXPECT_EQ(t % 5, 10u % 5);
        EXPECT_EQ(t % 3, 10u % 3);
        EXPECT_EQ(t, 10u);
        t += 10;
        EXPECT_EQ(t, 20u);
        t -= 10;
        EXPECT_EQ(t, 10u);
        t *= 2;
        EXPECT_EQ(t, 20u);
        t /= 2;
        EXPECT_EQ(t, 10u);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<flamegpu::id_t>("id");
        EXPECT_EQ(t++, 10u);
        EXPECT_EQ(++t, 12u);
        EXPECT_EQ(t--, 12u);
        EXPECT_EQ(--t, 10u);
        EXPECT_EQ(t / 5, 10u / 5);
        EXPECT_EQ(t / 3, 10u / 3);
        EXPECT_EQ(t / 3.0, 10u / 3.0);
        EXPECT_EQ(t + 5, 10u + 5);
        EXPECT_EQ(t + 3.0, 10u + 3.0);
        EXPECT_EQ(t - 3, 10u - 3);
        EXPECT_EQ(t - 3.0, 10u - 3.0);
        EXPECT_EQ(t * 5, 10u * 5);
        EXPECT_EQ(t * 3.0, 10u * 3.0);
        EXPECT_EQ(t % 5, 10u % 5);
        EXPECT_EQ(t % 3, 10u % 3);
        EXPECT_EQ(t, 10u);
        t += 10;
        EXPECT_EQ(t, 20u);
        t -= 10;
        EXPECT_EQ(t, 10u);
        t *= 2;
        EXPECT_EQ(t, 20u);
        t /= 2;
        EXPECT_EQ(t, 10u);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<float>("float");
        EXPECT_EQ(t++, 10);
        EXPECT_EQ(++t, 12);
        EXPECT_EQ(t--, 12);
        EXPECT_EQ(--t, 10);
        EXPECT_EQ(t / 5, 10.0f / 5);
        EXPECT_EQ(t / 3, 10.0f / 3);
        EXPECT_EQ(t + 5, 10.0f + 5);
        EXPECT_EQ(t - 3, 10.0f - 3);
        EXPECT_EQ(t * 5, 10.0f * 5);
        // EXPECT_EQ(t % 5, 10.0f % 5);  // remainder op does not support floating point types
        // EXPECT_EQ(t % 3, 10.0f % 3);  // remainder op does not support floating point types
        EXPECT_EQ(t, 10);
        t += 10;
        EXPECT_EQ(t, 20);
        t -= 10;
        EXPECT_EQ(t, 10);
        t *= 2;
        EXPECT_EQ(t, 20);
        t /= 2;
        EXPECT_EQ(t, 10);
    }
    {
        auto t = FLAMEGPU->environment.getMacroProperty<double>("double");
        EXPECT_EQ(t++, 10);
        EXPECT_EQ(++t, 12);
        EXPECT_EQ(t--, 12);
        EXPECT_EQ(--t, 10);
        EXPECT_EQ(t / 5, 10.0 / 5);
        EXPECT_EQ(t / 3, 10.0 / 3);
        EXPECT_EQ(t + 5, 10.0 + 5);
        EXPECT_EQ(t - 3, 10.0 - 3);
        EXPECT_EQ(t * 5, 10.0 * 5);
        // EXPECT_EQ(t % 5, 10.0 % 5);  // remainder op does not support floating point types
        // EXPECT_EQ(t % 3, 10.0 % 3);  // remainder op does not support floating point types
        EXPECT_EQ(t, 10);
        t += 10;
        EXPECT_EQ(t, 20);
        t -= 10;
        EXPECT_EQ(t, 10);
        t *= 2;
        EXPECT_EQ(t, 20);
        t /= 2;
        EXPECT_EQ(t, 10);
    }
}
TEST(HostMacroPropertyTest, ArithmeticTest) {
    // Create single macro property for each type
    // Fill MacroProperties with HostAPI to a known value
    // Use all airthmetic ops, and test values match expected value with DeviceAPI
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<int>("int");
    model.Environment().newMacroProperty<unsigned int>("uint");
    model.Environment().newMacroProperty<int8_t>("int8");
    model.Environment().newMacroProperty<uint8_t>("uint8");
    model.Environment().newMacroProperty<int64_t>("int64");
    model.Environment().newMacroProperty<uint64_t>("uint64");
    model.Environment().newMacroProperty<flamegpu::id_t>("id");
    model.Environment().newMacroProperty<float>("float");
    model.Environment().newMacroProperty<double>("double");
    // Setup agent fn
    model.newAgent("agent");
    model.newLayer().addHostFunction(HostArithmeticInit);
    model.newLayer().addHostFunction(HostArithmetic);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    ASSERT_NO_THROW(cudaSimulation.simulate());
}

/* These tests, test functionality which is not exposed unless LayerDescription allows agent fn and host fn in the same layer
#if !defined(SEATBELTS) || SEATBELTS
TEST(HostMacroPropertyTest, ReadSameLayerAsAgentWrite) {
#else
TEST(HostMacroPropertyTest, DISABLED_ReadSameLayerAsAgentWrite) {
#endif
    // Ensure an exception is thrown if an agent fn writes to the same macro property as a host function reads (or writes) from in the same layer
    // This is currently safe, as we do not execute host and agent functions concurrently, but may change in future

    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("i");
    agent.newVariable<unsigned int>("j");
    agent.newVariable<unsigned int>("k");
    agent.newVariable<unsigned int>("w");
    agent.newVariable<unsigned int>("a");
    agent.newFunction("agentwrite", AgentWrite);
    auto &l = model.newLayer();
    l.addAgentFunction(AgentWrite);
    l.addHostFunction(HostRead);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_THROW(cudaSimulation.simulate(), flamegpu::exception::InvalidOperation);
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(HostMacroPropertyTest, WriteSameLayerAsAgentRead) {
#else
TEST(HostMacroPropertyTest, DISABLED_WriteSameLayerAsAgentRead) {
#endif
    // Ensure an exception is thrown if an agent fn reads from the same macro property as a host function writes to in the same layer
    // This is currently safe, as we do not execute host and agent functions concurrently, but may change in future
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    model.Environment().newMacroProperty<unsigned int>("plusequal");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("i");
    agent.newVariable<unsigned int>("j");
    agent.newVariable<unsigned int>("k");
    agent.newVariable<unsigned int>("w");
    agent.newVariable<unsigned int>("a");
    agent.newFunction("agentread", AgentRead);
    auto &l = model.newLayer();
    l.addHostFunction(HostWrite);
    l.addAgentFunction(AgentRead);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_THROW(cudaSimulation.simulate(), flamegpu::exception::InvalidOperation);
}
*/
}  // namespace
}  // namespace flamegpu
