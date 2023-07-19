#include <iostream>
#include <fstream>

#include "gtest/gtest.h"

#include "flamegpu/flamegpu.h"

namespace flamegpu {

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
    // Limits
    EXPECT_TRUE(std::isnan(FLAMEGPU->environment.getProperty<float>("float_qnan")));
    EXPECT_TRUE(std::isnan(FLAMEGPU->environment.getProperty<float>("float_snan")));
    EXPECT_EQ(FLAMEGPU->environment.getProperty<float>("float_inf"), std::numeric_limits<float>::infinity());
    EXPECT_EQ(FLAMEGPU->environment.getProperty<float>("float_inf_neg"), -std::numeric_limits<float>::infinity());
    EXPECT_TRUE(std::isnan(FLAMEGPU->environment.getProperty<double>("double_qnan")));
    EXPECT_TRUE(std::isnan(FLAMEGPU->environment.getProperty<double>("double_snan")));
    EXPECT_EQ(FLAMEGPU->environment.getProperty<double>("double_inf"), std::numeric_limits<double>::infinity());
    EXPECT_EQ(FLAMEGPU->environment.getProperty<double>("double_inf_neg"), -std::numeric_limits<double>::infinity());
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
    FLAMEGPU->environment.setProperty<float>("float_qnan", {});
    FLAMEGPU->environment.setProperty<float>("float_snan", {});
    FLAMEGPU->environment.setProperty<float>("float_inf", {});
    FLAMEGPU->environment.setProperty<float>("float_inf_neg", {});
    FLAMEGPU->environment.setProperty<double>("double_qnan", {});
    FLAMEGPU->environment.setProperty<double>("double_snan", {});
    FLAMEGPU->environment.setProperty<double>("double_inf", {});
    FLAMEGPU->environment.setProperty<double>("double_inf_neg", {});
}

class MiniSim {
 public:
    void run(const std::string &test_file_name) {
        // Assertions for limits
        ASSERT_TRUE(std::numeric_limits<float>::has_quiet_NaN);
        ASSERT_TRUE(std::numeric_limits<float>::has_signaling_NaN);
        ASSERT_TRUE(std::numeric_limits<double>::has_quiet_NaN);
        ASSERT_TRUE(std::numeric_limits<double>::has_signaling_NaN);
        // Model description
        ModelDescription model("test_model");
        AgentDescription a = model.newAgent("a");
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
            a.newVariable<float>("float_qnan");
            a.newVariable<float>("float_snan");
            a.newVariable<float>("float_inf");
            a.newVariable<float>("float_inf_neg");
            a.newVariable<double>("double_qnan");
            a.newVariable<double>("double_snan");
            a.newVariable<double>("double_inf");
            a.newVariable<double>("double_inf_neg");
        }
        AgentDescription b = model.newAgent("b");
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
            EnvironmentDescription e = model.Environment();
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
            // Limit values
            e.newProperty<float>("float_qnan", std::numeric_limits<float>::quiet_NaN());
            e.newProperty<float>("float_snan", std::numeric_limits<float>::signaling_NaN());
            e.newProperty<float>("float_inf", std::numeric_limits<float>::infinity());
            e.newProperty<float>("float_inf_neg", -std::numeric_limits<float>::infinity());
            e.newProperty<double>("double_qnan", std::numeric_limits<double>::quiet_NaN());
            e.newProperty<double>("double_snan", std::numeric_limits<double>::quiet_NaN());
            e.newProperty<double>("double_inf", std::numeric_limits<double>::infinity());
            e.newProperty<double>("double_inf_neg", -std::numeric_limits<double>::infinity());
        }
        AgentVector pop_a_out(a, 5);
        for (unsigned int i = 0; i < 5; ++i) {
            auto agent = pop_a_out[i];
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
            // Limit values
            agent.setVariable<float>("float_qnan", std::numeric_limits<float>::quiet_NaN());
            agent.setVariable<float>("float_snan", std::numeric_limits<float>::signaling_NaN());
            agent.setVariable<float>("float_inf", std::numeric_limits<float>::infinity());
            agent.setVariable<float>("float_inf_neg", -std::numeric_limits<float>::infinity());
            agent.setVariable<double>("double_qnan", std::numeric_limits<double>::quiet_NaN());
            agent.setVariable<double>("double_snan", std::numeric_limits<double>::signaling_NaN());
            agent.setVariable<double>("double_inf", std::numeric_limits<double>::infinity());
            agent.setVariable<double>("double_inf_neg", -std::numeric_limits<double>::infinity());
        }
        AgentVector pop_b_out(b, 5);
        for (unsigned int i = 0; i < 5; ++i) {
            auto agent = pop_b_out[i];
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
            am.setPopulationData(pop_b_out, "2");  // Create them in the not initial state
            // Set config files for export too
            am.SimulationConfig().input_file = "invalid";
            am.SimulationConfig().step_log_file = "step";
            am.SimulationConfig().exit_log_file = "exit";
            am.SimulationConfig().common_log_file = "common";
            am.SimulationConfig().truncate_log_files = false;
            am.SimulationConfig().random_seed = 654321;
            am.SimulationConfig().steps = 123;
            am.SimulationConfig().verbosity = Verbosity::Quiet;
            am.SimulationConfig().timing = true;
#ifdef FLAMEGPU_VISUALISATION
            am.SimulationConfig().console_mode = true;
#endif
            am.CUDAConfig().device_id = 0;
            am.CUDAConfig().inLayerConcurrency = false;
            am.exportData(test_file_name);
        }
        {   // Run Import
            CUDASimulation am(model);
            // Ensure config doesnt match
            am.SimulationConfig().step_log_file = "";
            am.SimulationConfig().exit_log_file = "";
            am.SimulationConfig().common_log_file = "";
            am.SimulationConfig().truncate_log_files = true;
            am.SimulationConfig().random_seed = 0;
            am.SimulationConfig().steps = 0;
            am.SimulationConfig().verbosity = Verbosity::Verbose;
            am.SimulationConfig().timing = false;
#ifdef FLAMEGPU_VISUALISATION
            am.SimulationConfig().console_mode = false;
#endif
            am.CUDAConfig().device_id = 1000;
            am.CUDAConfig().inLayerConcurrency = true;
            // Perform import
            am.SimulationConfig().input_file = test_file_name;
            EXPECT_NO_THROW(am.applyConfig());  // If loading device id from config file didn't work, this would throw exception::InvalidCUDAdevice
            // Validate config matches
            EXPECT_EQ(am.getSimulationConfig().input_file, test_file_name);
            EXPECT_EQ(am.getSimulationConfig().step_log_file, "step");
            EXPECT_EQ(am.getSimulationConfig().exit_log_file, "exit");
            EXPECT_EQ(am.getSimulationConfig().common_log_file, "common");
            EXPECT_EQ(am.getSimulationConfig().truncate_log_files, false);
            EXPECT_EQ(am.getSimulationConfig().random_seed, 654321u);
            EXPECT_EQ(am.getSimulationConfig().steps, 123u);
            EXPECT_EQ(am.getSimulationConfig().verbosity, Verbosity::Quiet);
            EXPECT_EQ(am.getSimulationConfig().timing, true);
#ifdef FLAMEGPU_VISUALISATION
            EXPECT_EQ(am.getSimulationConfig().console_mode, true);
#endif
            EXPECT_EQ(am.getCUDAConfig().device_id, 0);
            EXPECT_EQ(am.getCUDAConfig().inLayerConcurrency, false);
            // Check population data
            AgentVector pop_a_in(a, 5);
            AgentVector pop_b_in(b, 5);
            am.getPopulationData(pop_a_in);
            am.getPopulationData(pop_b_in, "2");
            // Valid agent none array vars
            ASSERT_EQ(pop_a_in.size(), pop_a_out.size());
            for (unsigned int i = 0; i < pop_a_in.size(); ++i) {
                const auto agent_in = pop_a_in[i];
                const auto agent_out = pop_a_out[i];
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
                // Limit values
                EXPECT_TRUE(std::isnan(agent_in.getVariable<float>("float_qnan")));
                EXPECT_TRUE(std::isnan(agent_in.getVariable<float>("float_snan")));
                EXPECT_EQ(agent_in.getVariable<float>("float_inf"), std::numeric_limits<float>::infinity());
                EXPECT_EQ(agent_in.getVariable<float>("float_inf_neg"), -std::numeric_limits<float>::infinity());
                EXPECT_TRUE(std::isnan(agent_in.getVariable<double>("double_qnan")));
                EXPECT_TRUE(std::isnan(agent_in.getVariable<double>("double_snan")));
                EXPECT_EQ(agent_in.getVariable<double>("double_inf"), std::numeric_limits<double>::infinity());
                EXPECT_EQ(agent_in.getVariable<double>("double_inf_neg"), -std::numeric_limits<double>::infinity());
            }
            // Valid agent array vars
            ASSERT_EQ(pop_b_in.size(), pop_b_out.size());
            for (unsigned int i = 0; i < pop_b_in.size(); ++i) {
                const auto agent_in = pop_b_in[i];
                const auto agent_out = pop_b_out[i];
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
        {   // Run Import, but using run args
            CUDASimulation am(model);
            // Ensure config doesnt match

            am.SimulationConfig().step_log_file = "";
            am.SimulationConfig().exit_log_file = "";
            am.SimulationConfig().common_log_file = "";
            am.SimulationConfig().truncate_log_files = true;
            am.SimulationConfig().random_seed = 0;
            am.SimulationConfig().steps = 0;
            am.SimulationConfig().verbosity = Verbosity::Verbose;
            am.SimulationConfig().timing = false;
#ifdef FLAMEGPU_VISUALISATION
            am.SimulationConfig().console_mode = false;
#endif
            am.CUDAConfig().device_id = 1000;
            am.CUDAConfig().inLayerConcurrency = true;
            // Perform import
            const char *argv[3] = { "prog.exe", "--in", test_file_name.c_str()};
            EXPECT_NO_THROW(am.initialise(sizeof(argv) / sizeof(char*), argv));
            // Validate config matches
            EXPECT_EQ(am.getSimulationConfig().input_file, test_file_name);
            EXPECT_EQ(am.getSimulationConfig().step_log_file, "step");
            EXPECT_EQ(am.getSimulationConfig().exit_log_file, "exit");
            EXPECT_EQ(am.getSimulationConfig().common_log_file, "common");
            EXPECT_EQ(am.getSimulationConfig().truncate_log_files, false);
            EXPECT_EQ(am.getSimulationConfig().random_seed, 654321u);
            EXPECT_EQ(am.getSimulationConfig().steps, 123u);
            EXPECT_EQ(am.getSimulationConfig().verbosity, Verbosity::Quiet);
            EXPECT_EQ(am.getSimulationConfig().timing, true);
#ifdef FLAMEGPU_VISUALISATION
            EXPECT_EQ(am.getSimulationConfig().console_mode, true);
#endif
            EXPECT_EQ(am.getCUDAConfig().device_id, 0);
            EXPECT_EQ(am.getCUDAConfig().inLayerConcurrency, false);
            // Check population data
            AgentVector pop_a_in(a, 5);
            AgentVector pop_b_in(b, 5);
            am.getPopulationData(pop_a_in);
            am.getPopulationData(pop_b_in, "2");
            // Valid agent none array vars
            ASSERT_EQ(pop_a_in.size(), pop_a_out.size());
            for (unsigned int i = 0; i < pop_a_in.size(); ++i) {
                const auto agent_in = pop_a_in[i];
                const auto agent_out = pop_a_out[i];
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
                // Limit values
                EXPECT_TRUE(std::isnan(agent_in.getVariable<float>("float_qnan")));
                EXPECT_TRUE(std::isnan(agent_in.getVariable<float>("float_snan")));
                EXPECT_EQ(agent_in.getVariable<float>("float_inf"), std::numeric_limits<float>::infinity());
                EXPECT_EQ(agent_in.getVariable<float>("float_inf_neg"), -std::numeric_limits<float>::infinity());
                EXPECT_TRUE(std::isnan(agent_in.getVariable<double>("double_qnan")));
                EXPECT_TRUE(std::isnan(agent_in.getVariable<double>("double_snan")));
                EXPECT_EQ(agent_in.getVariable<double>("double_inf"), std::numeric_limits<double>::infinity());
                EXPECT_EQ(agent_in.getVariable<double>("double_inf_neg"), -std::numeric_limits<double>::infinity());
            }
            // Valid agent array vars
            ASSERT_EQ(pop_b_in.size(), pop_b_out.size());
            for (unsigned int i = 0; i < pop_b_in.size(); ++i) {
                const auto agent_in = pop_b_in[i];
                const auto agent_out = pop_b_out[i];
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
FLAMEGPU_HOST_FUNCTION(DoNothing) {
    // Do nothing
}
TEST(IOTest2, AgentID_JSON_ExportImport) {
    // Create an agent pop, add it to the model, step so that they are assigned ids
    // Export pop
    // getPopData to an agent vector
    // Check that IDs are set to anything but unset ID.
    // Create new CUDASim, reimport pop
    // getPopData to an agent vector
    // Check that IDs are set to anything but unset ID.
    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_other", ID_NOT_SET);
    auto layer_a = model.newLayer();
    layer_a.addHostFunction(DoNothing);

    {
        AgentVector pop_in(agent, 100);
        for (auto a : pop_in) {
            ASSERT_EQ(a.getID(), ID_NOT_SET);
        }
        CUDASimulation sim(model);
        sim.setPopulationData(pop_in);
        sim.step();

        AgentVector pop_out(agent);
        sim.getPopulationData(pop_out);
        for (auto a : pop_out) {
            ASSERT_NE(a.getID(), ID_NOT_SET);
        }
        sim.exportData(JSON_FILE_NAME);
    }
    {
        CUDASimulation sim(model);
        sim.SimulationConfig().input_file = JSON_FILE_NAME;
        EXPECT_NO_THROW(sim.applyConfig());

        AgentVector pop_out(agent);
        sim.getPopulationData(pop_out);
        for (auto a : pop_out) {
            ASSERT_NE(a.getID(), ID_NOT_SET);
        }
    }
    // Cleanup
    ASSERT_EQ(::remove(JSON_FILE_NAME), 0);
}
TEST(IOTest2, AgentID_XML_ExportImport) {
    // Create an agent pop, add it to the model, step so that they are assigned ids
    // Export pop
    // getPopData to an agent vector
    // Check that IDs are set to anything but unset ID.
    // Create new CUDASim, reimport pop
    // getPopData to an agent vector
    // Check that IDs are set to anything but unset ID.
    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_other", ID_NOT_SET);
    auto layer_a = model.newLayer();
    layer_a.addHostFunction(DoNothing);

    {
        AgentVector pop_in(agent, 100);
        for (auto a : pop_in) {
            ASSERT_EQ(a.getID(), ID_NOT_SET);
        }
        CUDASimulation sim(model);
        sim.setPopulationData(pop_in);
        sim.step();

        AgentVector pop_out(agent);
        sim.getPopulationData(pop_out);
        for (auto a : pop_out) {
            ASSERT_NE(a.getID(), ID_NOT_SET);
        }
        sim.exportData(JSON_FILE_NAME);
    }
    {
        CUDASimulation sim(model);
        sim.SimulationConfig().input_file = JSON_FILE_NAME;
        EXPECT_NO_THROW(sim.applyConfig());

        AgentVector pop_out(agent);
        sim.getPopulationData(pop_out);
        for (auto a : pop_out) {
            ASSERT_NE(a.getID(), ID_NOT_SET);
        }
    }
    // Cleanup
    ASSERT_EQ(::remove(JSON_FILE_NAME), 0);
}
// Agent ID collision is detected on import from file

TEST(IOTest2, AgentID_FileInput_IDCollision) {
    const char* JSON_FILE_BODY = "{\"agents\":{\"agent\":{\"default\":[{\"_id\":1,\"id_other\":0},{\"_id\":2,\"id_other\":0},{\"_id\":1,\"id_other\":0}]}}}";  // 3 AGENTS, with ids 1,2,1
    // Manually create test.json, containing an agent pop with ID collision
    // Import pop and expect exception
    // Delete test.json
    // The actual part that handles the collision is input type agnostic, so don't need to test with XML too
    {
        std::ofstream myfile;
        myfile.open(JSON_FILE_NAME, std::ofstream::out | std::ofstream::trunc);
        myfile << JSON_FILE_BODY;
        myfile.close();
    }

    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_other", ID_NOT_SET);
    auto layer_a = model.newLayer();
    layer_a.addHostFunction(DoNothing);
    CUDASimulation sim(model);
    sim.SimulationConfig().input_file = JSON_FILE_NAME;
    EXPECT_THROW(sim.applyConfig(), exception::AgentIDCollision);
    // Cleanup
    ASSERT_EQ(::remove(JSON_FILE_NAME), 0);
}
}  // namespace test_io
}  // namespace flamegpu
