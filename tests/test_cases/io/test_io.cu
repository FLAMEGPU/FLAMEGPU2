#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"


namespace test_io {
bool validate_has_run = false;
const char *XML_FILE_NAME = "test.xml";
const char *JSON_FILE_NAME = "test.json";
FLAMEGPU_STEP_FUNCTION(VALIDATE_ENV) {
    EXPECT_EQ(FLAMEGPU->environment.getProperty<float>("float"), 12.0f);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<float>("float"), 12.0f);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<double>("double"), 13.0);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<int64_t>("int64_t"), 14);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<uint64_t>("uint64_t"), 15u);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<int32_t>("int32_t"), 16);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<uint32_t>("uint32_t"), 17u);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<int16_t>("int16_t"), 18);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<uint16_t>("uint16_t"), 19u);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<int8_t>("int8_t"), 20);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<uint8_t>("uint8_t"), 21u);
    const bool float_array_eq = FLAMEGPU->environment.getProperty<float, 3>("float_a") == std::array<float, 3>{ 12.0f, 0.0f, 1.0f };
    EXPECT_TRUE(float_array_eq);
    const bool double_array_eq = FLAMEGPU->environment.getProperty<double, 3>("double_a") == std::array<double, 3>{ 13.0, 0.0, 1.0 };
    EXPECT_TRUE(double_array_eq);
    const bool int64_t_array_eq = FLAMEGPU->environment.getProperty<int64_t, 3>("int64_t_a") == std::array<int64_t, 3>{ 14, 0, 1 };
    EXPECT_TRUE(int64_t_array_eq);
    const bool uint64_t_array_eq = FLAMEGPU->environment.getProperty<uint64_t, 3>("uint64_t_a") == std::array<uint64_t, 3>{ 15u, 0u, 1u };
    EXPECT_TRUE(uint64_t_array_eq);
    const bool int32_t_array_eq = FLAMEGPU->environment.getProperty<int32_t, 3>("int32_t_a") == std::array<int32_t, 3>{ 16, 0, 1 };
    EXPECT_TRUE(int32_t_array_eq);
    const bool uint32_t_array_eq = FLAMEGPU->environment.getProperty<uint32_t, 3>("uint32_t_a") == std::array<uint32_t, 3> { 17u, 0u, 1u };
    EXPECT_TRUE(uint32_t_array_eq);
    const bool int16_t_array_eq = FLAMEGPU->environment.getProperty<int16_t, 3>("int16_t_a") == std::array<int16_t, 3>{ 18, 0, 1 };
    EXPECT_TRUE(int16_t_array_eq);
    const bool uint16_t_array_eq = FLAMEGPU->environment.getProperty<uint16_t, 3>("uint16_t_a") == std::array<uint16_t, 3>{ 19u, 0u, 1u };
    EXPECT_TRUE(uint16_t_array_eq);
    const bool int8_t_array_eq = FLAMEGPU->environment.getProperty<int8_t, 3>("int8_t_a") == std::array<int8_t, 3>{ 20, 0, 1 };
    EXPECT_TRUE(int8_t_array_eq);
    const bool uint8_t_array_eq = FLAMEGPU->environment.getProperty<uint8_t, 3>("uint8_t_a") == std::array<uint8_t, 3>{ 21u, 0u, 1u };
    EXPECT_TRUE(uint8_t_array_eq);
    validate_has_run = true;
}
FLAMEGPU_STEP_FUNCTION(RESET_ENV) {
    FLAMEGPU->environment.setProperty<float>("float", {});
    FLAMEGPU->environment.setProperty<double>("double", {});
    FLAMEGPU->environment.setProperty<int64_t>("int64_t", {});
    FLAMEGPU->environment.setProperty<uint64_t>("uint64_t", {});
    FLAMEGPU->environment.setProperty<int32_t>("int32_t", {});
    FLAMEGPU->environment.setProperty<uint32_t>("uint32_t", {});
    FLAMEGPU->environment.setProperty<int16_t>("int16_t", {});
    FLAMEGPU->environment.setProperty<uint16_t>("uint16_t", {});
    FLAMEGPU->environment.setProperty<int8_t>("int8_t", {});
    FLAMEGPU->environment.setProperty<uint8_t>("uint8_t", {});
    FLAMEGPU->environment.setProperty<float, 3>("float_a", {});
    FLAMEGPU->environment.setProperty<double, 3>("double_a", {});
    FLAMEGPU->environment.setProperty<int64_t, 3>("int64_t_a", {});
    FLAMEGPU->environment.setProperty<uint64_t, 3>("uint64_t_a", {});
    FLAMEGPU->environment.setProperty<int32_t, 3>("int32_t_a", {});
    FLAMEGPU->environment.setProperty<uint32_t, 3>("uint32_t_a", {});
    FLAMEGPU->environment.setProperty<int16_t, 3>("int16_t_a", {});
    FLAMEGPU->environment.setProperty<uint16_t, 3>("uint16_t_a", {});
    FLAMEGPU->environment.setProperty<int8_t, 3>("int8_t_a", {});
    FLAMEGPU->environment.setProperty<uint8_t, 3>("uint8_t_a", {});
}

class MiniSim {
 public:
    void run(const std::string &test_file_name) {
        ModelDescription model("test_model");
        AgentDescription &a = model.newAgent("a");
        {
            a.newVariable<float>("float");
            a.newVariable<double>("double");
            a.newVariable<int64_t>("int64_t");
            a.newVariable<uint64_t>("uint64_t");
            a.newVariable<int32_t>("int32_t");
            a.newVariable<uint32_t>("uint32_t");
            a.newVariable<int16_t>("int16_t");
            a.newVariable<uint16_t>("uint16_t");
            a.newVariable<int8_t>("int8_t");
            a.newVariable<uint8_t>("uint8_t");
        }
        AgentDescription &b = model.newAgent("b");
        b.newState("1");
        b.newState("2");
        {
            b.newVariable<float, 3>("float");
            b.newVariable<double, 3>("double");
            b.newVariable<int64_t, 3>("int64_t");
            b.newVariable<uint64_t, 3>("uint64_t");
            b.newVariable<int32_t, 3>("int32_t");
            b.newVariable<uint32_t, 3>("uint32_t");
            b.newVariable<int16_t, 3>("int16_t");
            b.newVariable<uint16_t, 3>("uint16_t");
            b.newVariable<int8_t, 3>("int8_t");
            b.newVariable<uint8_t, 3>("uint8_t");
        }
        {
            EnvironmentDescription &e = model.Environment();
            e.newProperty<float>("float", 12.0f);
            e.newProperty<double>("double", 13.0);
            e.newProperty<int64_t>("int64_t", 14);
            e.newProperty<uint64_t>("uint64_t", 15u);
            e.newProperty<int32_t>("int32_t", 16);
            e.newProperty<uint32_t>("uint32_t", 17u);
            e.newProperty<int16_t>("int16_t", 18);
            e.newProperty<uint16_t>("uint16_t", 19u);
            e.newProperty<int8_t>("int8_t", 20);
            e.newProperty<uint8_t>("uint8_t", 21u);
            e.newProperty<float, 3>("float_a", { 12.0f, 0.0f, 1.0f });
            e.newProperty<double, 3>("double_a", { 13.0, 0.0, 1.0 });
            e.newProperty<int64_t, 3>("int64_t_a", { 14, 0, 1 });
            e.newProperty<uint64_t, 3>("uint64_t_a", { 15u, 0u, 1u });
            e.newProperty<int32_t, 3>("int32_t_a", { 16, 0, 1 });
            e.newProperty<uint32_t, 3>("uint32_t_a", { 17u, 0u, 1u });
            e.newProperty<int16_t, 3>("int16_t_a", { 18, 0, 1});
            e.newProperty<uint16_t, 3>("uint16_t_a", { 19u, 0u, 1u });
            e.newProperty<int8_t, 3>("int8_t_a", { 20, 0, 1 });
            e.newProperty<uint8_t, 3>("uint8_t_a", {21u, 0u, 1u});
        }
        AgentPopulation pop_a_out(a, 5);
        for (unsigned int i = 0; i < 5; ++i) {
            auto agent = pop_a_out.getNextInstance();
            agent.setVariable<float>("float", static_cast<float>(1.0f + i));
            agent.setVariable<double>("double", static_cast<double>(2.0 + i));
            agent.setVariable<int64_t>("int64_t", 3 + i);
            agent.setVariable<uint64_t>("uint64_t", 4u + i);
            agent.setVariable<int32_t>("int32_t", 5 + i);
            agent.setVariable<uint32_t>("uint32_t", 6u + i);
            agent.setVariable<int16_t>("int16_t", static_cast<int16_t>(7 + i));
            agent.setVariable<uint16_t>("uint16_t", static_cast<uint16_t>(8u + i));
            agent.setVariable<int8_t>("int8_t", static_cast<int8_t>(9 + i));
            agent.setVariable<uint8_t>("uint8_t", static_cast<uint8_t>(10u + i));
        }
        AgentPopulation pop_b_out(b, 5);
        for (unsigned int i = 0; i < 5; ++i) {
            auto agent = pop_b_out.getNextInstance("2");  // Create them in the not initial state
            agent.setVariable<float, 3>("float", { 1.0f, static_cast<float>(i), 1.0f });
            agent.setVariable<double, 3>("double", { 2.0, static_cast<double>(i), 1.0 });
            agent.setVariable<int64_t, 3>("int64_t", { 3, static_cast<int64_t>(i), 1 });
            agent.setVariable<uint64_t, 3>("uint64_t", { 4u, static_cast<uint64_t>(i), 1u });
            agent.setVariable<int32_t, 3>("int32_t", { 5, static_cast<int32_t>(i), 1 });
            agent.setVariable<uint32_t, 3>("uint32_t", { 6u, static_cast<uint32_t>(i), 1u });
            agent.setVariable<int16_t, 3>("int16_t", { static_cast<int16_t>(7), static_cast<int16_t>(i), static_cast<int16_t>(1) });
            agent.setVariable<uint16_t, 3>("uint16_t", { static_cast<uint16_t>(8), static_cast<uint16_t>(i), static_cast<uint16_t>(1) });
            agent.setVariable<int8_t, 3>("int8_t", { static_cast<int8_t>(9), static_cast<int8_t>(i), static_cast<int8_t>(1) });
            agent.setVariable<uint8_t, 3>("uint8_t", { static_cast<uint8_t>(10), static_cast<uint8_t>(i), static_cast<uint8_t>(1) });
        }
        model.newLayer().addHostFunction(VALIDATE_ENV);
        model.newLayer().addHostFunction(RESET_ENV);
        {  // Run export
            CUDASimulation am(model);
            am.setPopulationData(pop_a_out);
            am.setPopulationData(pop_b_out);
            // Set config files for export too
            am.SimulationConfig().input_file = "invalid";
            am.SimulationConfig().random_seed = 654321;
            am.SimulationConfig().steps = 123;
            am.SimulationConfig().timing = true;
            am.SimulationConfig().verbose = false;
            am.CUDAConfig().device_id = 0;
            am.exportData(test_file_name);
        }
        {  // Run Import
            CUDASimulation am(model);
            // Ensure config doesn;t match
            am.SimulationConfig().random_seed = 0;
            am.SimulationConfig().steps = 0;
            am.SimulationConfig().timing = false;
            am.SimulationConfig().verbose = true;
            am.CUDAConfig().device_id = 1000;
            // Perform import
            am.SimulationConfig().input_file = test_file_name;
            EXPECT_NO_THROW(am.applyConfig());  // If loading device id from config file didn't work, this would throw InvalidCUDAdevice
            // Validate config matches
            EXPECT_EQ(am.getSimulationConfig().random_seed, 654321u);
            EXPECT_EQ(am.getSimulationConfig().steps, 123u);
            EXPECT_EQ(am.getSimulationConfig().timing, true);
            EXPECT_EQ(am.getSimulationConfig().verbose, false);
            EXPECT_EQ(am.getSimulationConfig().input_file, test_file_name);
            EXPECT_EQ(am.getCUDAConfig().device_id, 0);
            AgentPopulation pop_a_in(a, 5);
            AgentPopulation pop_b_in(b, 5);
            am.getPopulationData(pop_a_in);
            am.getPopulationData(pop_b_in);
            // Valid agent none array vars
            ASSERT_EQ(pop_a_in.getCurrentListSize(), pop_a_out.getCurrentListSize());
            for (unsigned int i = 0; i < pop_a_in.getCurrentListSize(); ++i) {
                const auto agent_in = pop_a_in.getInstanceAt(i);
                const auto agent_out = pop_a_out.getInstanceAt(i);
                EXPECT_EQ(agent_in.getVariable<float>("float"), agent_out.getVariable<float>("float"));
                EXPECT_EQ(agent_in.getVariable<double>("double"), agent_out.getVariable<double>("double"));
                EXPECT_EQ(agent_in.getVariable<int64_t>("int64_t"), agent_out.getVariable<int64_t>("int64_t"));
                EXPECT_EQ(agent_in.getVariable<uint64_t>("uint64_t"), agent_out.getVariable<uint64_t>("uint64_t"));
                EXPECT_EQ(agent_in.getVariable<int32_t>("int32_t"), agent_out.getVariable<int32_t>("int32_t"));
                EXPECT_EQ(agent_in.getVariable<uint32_t>("uint32_t"), agent_out.getVariable<uint32_t>("uint32_t"));
                EXPECT_EQ(agent_in.getVariable<int16_t>("int16_t"), agent_out.getVariable<int16_t>("int16_t"));
                EXPECT_EQ(agent_in.getVariable<uint16_t>("uint16_t"), agent_out.getVariable<uint16_t>("uint16_t"));
                EXPECT_EQ(agent_in.getVariable<int8_t>("int8_t"), agent_out.getVariable<int8_t>("int8_t"));
                EXPECT_EQ(agent_in.getVariable<uint8_t>("uint8_t"), agent_out.getVariable<uint8_t>("uint8_t"));
            }
            // Valid agent array vars
            ASSERT_EQ(pop_b_in.getCurrentListSize("2"), pop_b_out.getCurrentListSize("2"));
            for (unsigned int i = 0; i < pop_b_in.getCurrentListSize("2"); ++i) {
                const auto agent_in = pop_b_in.getInstanceAt(i, "2");
                const auto agent_out = pop_b_out.getInstanceAt(i, "2");
                const bool float_array = agent_in.getVariable<float, 3>("float") == agent_out.getVariable<float, 3>("float");
                const bool double_array = agent_in.getVariable<double, 3>("double") == agent_out.getVariable<double, 3>("double");
                const bool int64_t_array = agent_in.getVariable<int64_t, 3>("int64_t") == agent_out.getVariable<int64_t, 3>("int64_t");
                const bool uint64_t_array = agent_in.getVariable<uint64_t, 3>("uint64_t") == agent_out.getVariable<uint64_t, 3>("uint64_t");
                const bool int32_t_array = agent_in.getVariable<int32_t, 3>("int32_t") == agent_out.getVariable<int32_t, 3>("int32_t");
                const bool uint32_t_array = agent_in.getVariable<uint32_t, 3>("uint32_t") == agent_out.getVariable<uint32_t, 3>("uint32_t");
                const bool int16_t_t_array = agent_in.getVariable<int16_t, 3>("int16_t") == agent_out.getVariable<int16_t, 3>("int16_t");
                const bool uint16_t_t_array = agent_in.getVariable<uint16_t, 3>("uint16_t") == agent_out.getVariable<uint16_t, 3>("uint16_t");
                const bool int8_t_t_array = agent_in.getVariable<int8_t, 3>("int8_t") == agent_out.getVariable<int8_t, 3>("int8_t");
                const bool uint8_t_array = agent_in.getVariable<uint8_t, 3>("uint8_t") == agent_out.getVariable<uint8_t, 3>("uint8_t");
                EXPECT_TRUE(float_array);
                EXPECT_TRUE(double_array);
                EXPECT_TRUE(int64_t_array);
                EXPECT_TRUE(uint64_t_array);
                EXPECT_TRUE(int32_t_array);
                EXPECT_TRUE(uint32_t_array);
                EXPECT_TRUE(int16_t_t_array);
                EXPECT_TRUE(uint16_t_t_array);
                EXPECT_TRUE(int8_t_t_array);
                EXPECT_TRUE(uint8_t_array);
            }
        }
        {  // Validate env_vars
            // Load model
            CUDASimulation am(model);
            // Step once, this checks and clears env vars
            validate_has_run = false;
            am.step();
            ASSERT_TRUE(validate_has_run);
            // Reload env vars from file
            am.SimulationConfig().input_file = test_file_name;
            am.applyConfig();
            // Step again, check they have been loaded
            validate_has_run = false;
            am.step();
            ASSERT_TRUE(validate_has_run);
        }
        // Cleanup
        ASSERT_EQ(::remove(test_file_name.c_str()), 0);
    }
};

/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class IOTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
    }

    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};

TEST_F(IOTest, XML_WriteRead) {
    ms->run(XML_FILE_NAME);
}
TEST_F(IOTest, JSON_WriteRead) {
    ms->run(JSON_FILE_NAME);
}
}  // namespace test_io
