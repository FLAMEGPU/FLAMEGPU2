#ifndef TESTS_TEST_CASES_RUNTIME_TEST_AGENT_RANDOM_H_
#define TESTS_TEST_CASES_RUNTIME_TEST_AGENT_RANDOM_H_

#include <string>
#include <tuple>
#include <vector>

#include "flamegpu/flamegpu.h"
#include "helpers/common.h"

#include "gtest/gtest.h"

namespace flamegpu {

namespace test_agent_random {
FLAMEGPU_AGENT_FUNCTION(random1_func, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<float>("a", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("b", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("c", FLAMEGPU->random.uniform<float>());

    return ALIVE;
}
/**
 * @brief      To verify the correctness of agent random
 *
 * This test checks whether AgentRandom and RandomManager behave correctly
*/
TEST(AgentRandomTest, AgentRandomCheck) {
    GTEST_COUT << "Testing AgentRandom and DeviceRandomArray Name .." << std::endl;

    const unsigned int AGENT_COUNT = 5;

    ModelDescription model("random_model");
    AgentDescription &agent = model.newAgent("agent");

    agent.newVariable<float>("a");
    agent.newVariable<float>("b");
    agent.newVariable<float>("c");

    AgentFunctionDescription &af = agent.newFunction("random1", random1_func);

    AgentVector init_population(agent, AGENT_COUNT);
    AgentVector population(agent, AGENT_COUNT);
    for (AgentVector::Agent instance : init_population) {
        instance.setVariable<float>("a", 0);
        instance.setVariable<float>("b", 0);
        instance.setVariable<float>("c", 0);
    }


    LayerDescription &layer = model.newLayer("layer");
    layer.addAgentFunction(af);

    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    const char *args_1[5] = { "process.exe", "-r", "0", "-s", "1" };
    const char *args_2[5] = { "process.exe", "-r", "1", "-s", "1" };
    std::string _t_unused = std::string();
    std::vector<std::tuple<float, float, float>> results1, results2;
    {
        /**
        * Test Model 1
        * Do agents generate different random numbers
        * Does random number change each time it's called
        */
        // Seed random
        cudaSimulation.initialise(5, args_1);
        cudaSimulation.setPopulationData(init_population);

        cudaSimulation.simulate();

        cudaSimulation.getPopulationData(population);

        float a1 = -1, b1 = -1, c1 = -1, a2 = -1, b2 = -1, c2 = -1;
        for (unsigned int i = 0; i < population.size(); i++) {
            AgentVector::Agent instance = population[i];
            if (i != 0) {
                a2 = a1;
                b2 = b1;
                c2 = c1;
            }
            a1 = instance.getVariable<float>("a");
            b1 = instance.getVariable<float>("b");
            c1 = instance.getVariable<float>("c");
            results1.push_back(std::make_tuple(a1, b1, c1));
            if (i != 0) {
                // Different agents get different random numbers
                EXPECT_TRUE(a1 != a2);
                EXPECT_TRUE(b1 != b2);
                EXPECT_TRUE(c1 != c2);
            }
            // Multiple calls get multiple random numbers
            EXPECT_TRUE(a1 != b1);
            EXPECT_TRUE(b1 != c1);
            EXPECT_TRUE(a1 != c1);
        }
        EXPECT_EQ(results1.size(), AGENT_COUNT);
    }
    {
        /**
         * Test Model 2
         * Different seed produces different random numbers
         */
        // Seed random
        cudaSimulation.initialise(5, args_2);
        cudaSimulation.setPopulationData(init_population);

        cudaSimulation.simulate();

        cudaSimulation.getPopulationData(population);

        for (unsigned int i = 0; i < population.size(); i++) {
            AgentVector::Agent instance = population[i];
            results2.push_back(std::make_tuple(
                instance.getVariable<float>("a"),
                instance.getVariable<float>("b"),
                instance.getVariable<float>("c")));
        }
        EXPECT_TRUE(results2.size() == AGENT_COUNT);

        for (unsigned int i = 0; i < results1.size(); ++i) {
            // Different seed produces different results
            EXPECT_NE(results1[i], results2[i]);
        }
    }
    {
        /**
        * Test Model 3
        * Different seed produces different random numbers
        */
        results2.clear();
        // Seed random
        cudaSimulation.initialise(5, args_1);
        cudaSimulation.setPopulationData(init_population);

        cudaSimulation.simulate();

        cudaSimulation.getPopulationData(population);

        for (unsigned int i = 0; i < population.size(); i++) {
            AgentVector::Agent instance = population[i];
            results2.push_back(std::make_tuple(
                instance.getVariable<float>("a"),
                instance.getVariable<float>("b"),
                instance.getVariable<float>("c")));
        }
        EXPECT_EQ(results2.size(), AGENT_COUNT);

        for (unsigned int i = 0; i < results1.size(); ++i) {
            // Same seed produces same results
            EXPECT_EQ(results1[i], results2[i]);
        }
    }
}

FLAMEGPU_AGENT_FUNCTION(random2_func, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<float>("uniform_float", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<double>("uniform_double", FLAMEGPU->random.uniform<double>());

    FLAMEGPU->setVariable<float>("normal_float", FLAMEGPU->random.normal<float>());
    FLAMEGPU->setVariable<double>("normal_double", FLAMEGPU->random.normal<double>());

    FLAMEGPU->setVariable<float>("logNormal_float", FLAMEGPU->random.logNormal<float>(0, 1));
    FLAMEGPU->setVariable<double>("logNormal_double", FLAMEGPU->random.logNormal<double>(0, 1));

    // char
    FLAMEGPU->setVariable<char>("uniform_char", FLAMEGPU->random.uniform<char>(CHAR_MIN, CHAR_MAX));
    FLAMEGPU->setVariable<unsigned char>("uniform_u_char", FLAMEGPU->random.uniform<unsigned char>(0, UCHAR_MAX));
    // short
    FLAMEGPU->setVariable<int16_t>("uniform_short", FLAMEGPU->random.uniform<int16_t>(INT16_MIN, INT16_MAX));
    FLAMEGPU->setVariable<uint16_t>("uniform_u_short", FLAMEGPU->random.uniform<uint16_t>(0, UINT16_MAX));
    // int
    FLAMEGPU->setVariable<int32_t>("uniform_int", FLAMEGPU->random.uniform<int32_t>(INT32_MIN, INT32_MAX));
    FLAMEGPU->setVariable<uint32_t>("uniform_u_int", FLAMEGPU->random.uniform<uint32_t>(0, UINT32_MAX));
    // long long
    FLAMEGPU->setVariable<int64_t>("uniform_longlong", FLAMEGPU->random.uniform<int64_t>(INT64_MIN, INT64_MAX));
    FLAMEGPU->setVariable<uint64_t>("uniform_u_longlong", FLAMEGPU->random.uniform<uint64_t>(0, UINT64_MAX));

    return ALIVE;
}

TEST(AgentRandomTest, AgentRandomFunctionsNoExcept) {
    GTEST_COUT << "Testing AgentRandom functions all work" << std::endl;

    const unsigned int AGENT_COUNT = 5;


    ModelDescription model("random_model");
    AgentDescription &agent = model.newAgent("agent");

    agent.newVariable<float>("uniform_float");
    agent.newVariable<double>("uniform_double");

    agent.newVariable<float>("normal_float");
    agent.newVariable<double>("normal_double");

    agent.newVariable<float>("logNormal_float");
    agent.newVariable<double>("logNormal_double");

    // char
    agent.newVariable<char>("uniform_char");
    agent.newVariable<unsigned char>("uniform_u_char");
    // short
    agent.newVariable<int16_t>("uniform_short");
    agent.newVariable<uint16_t>("uniform_u_short");
    // int
    agent.newVariable<int32_t>("uniform_int");
    agent.newVariable<uint32_t>("uniform_u_int");
    // long long
    agent.newVariable<int64_t>("uniform_longlong");
    agent.newVariable<uint64_t>("uniform_u_longlong");

    // do_random.setFunction(&random1);
    AgentFunctionDescription &do_random = agent.newFunction("random2", random2_func);

    AgentVector population(agent, AGENT_COUNT);

    LayerDescription &layer = model.newLayer("layer");
    layer.addAgentFunction(do_random);

    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    // Success if we get this far without an exception being thrown.
}

TEST(AgentRandomTest, AgentRandomArrayResizeNoExcept) {
    GTEST_COUT << "Testing d_random scales up / down without breaking" << std::endl;

    std::vector<int> AGENT_COUNTS = {1024, 512 * 1024, 1024, 1024 * 1024};

    // TODO(Rob): Can't yet control agent population up/down

    // Success if we get this far without an exception being thrown.
}
}  // namespace test_agent_random
}  // namespace flamegpu
#endif  // TESTS_TEST_CASES_RUNTIME_TEST_AGENT_RANDOM_H_
