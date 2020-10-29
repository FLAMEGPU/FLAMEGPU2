#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"


namespace test_logging {
const char *MODEL_NAME = "Model";
const char *AGENT_NAME1 = "Agent1";
const char *FUNCTION_NAME1 = "Func1";
const char *HOST_FUNCTION_NAME1 = "Func2";

FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgNone, MsgNone) {
    // increment all variables
    FLAMEGPU->setVariable<float>("float_var", FLAMEGPU->getVariable<float>("float_var") + 1.0f);
    FLAMEGPU->setVariable<int>("int_var", FLAMEGPU->getVariable<int>("int_var") + 1);
    FLAMEGPU->setVariable<unsigned int>("uint_var", FLAMEGPU->getVariable<unsigned int>("uint_var") + 1);
    return ALIVE;
}
FLAMEGPU_STEP_FUNCTION(step_fn1) {
    // increment all properties
    FLAMEGPU->environment.setProperty<float>("float_prop", FLAMEGPU->environment.getProperty<float>("float_prop") + 1.0f);
    FLAMEGPU->environment.setProperty<int>("int_prop", FLAMEGPU->environment.getProperty<int>("int_prop") + 1);
    FLAMEGPU->environment.setProperty<unsigned int>("uint_prop", FLAMEGPU->environment.getProperty<unsigned int>("uint_prop") + 1);

    auto a = FLAMEGPU->environment.getProperty<float, 2>("float_prop_array");
    auto b = FLAMEGPU->environment.getProperty<int, 3>("int_prop_array");
    auto c = FLAMEGPU->environment.getProperty<unsigned int, 4>("uint_prop_array");
    FLAMEGPU->environment.setProperty<float, 2>("float_prop_array", {a[0] + 1.0f, a[1] + 1.0f});
    FLAMEGPU->environment.setProperty<int, 3>("int_prop_array", {b[0] + 1, b[1] + 1, b[2] + 1});
    FLAMEGPU->environment.setProperty<unsigned int, 4>("uint_prop_array", {c[0] + 1, c[1] + 1, c[2] + 1, c[3] + 1});
}
template<typename T>
void logAllAgent(AgentLoggingConfig &alcfg, const std::string &var_name) {
    alcfg.logMin<T>(var_name);
    alcfg.logMax<T>(var_name);
    alcfg.logMean<T>(var_name);
    alcfg.logStandardDev<T>(var_name);
    alcfg.logSum<T>(var_name);
}
TEST(LoggingTest, CUDASimulationStep) {
    /**
     * Ensure the expected data is logged when CUDASimulation::step() is called
     * Note: does not check files logged to disk
     */
    // Define model
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    a.newVariable<float>("float_var");
    a.newVariable<int>("int_var");
    a.newVariable<unsigned int>("uint_var");
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    m.newLayer().addAgentFunction(f1);
    m.addStepFunction(step_fn1);
    m.Environment().newProperty<float>("float_prop", 1.0f);
    m.Environment().newProperty<int>("int_prop", 1);
    m.Environment().newProperty<unsigned int>("uint_prop", 1);
    m.Environment().newProperty<float, 2>("float_prop_array", {1.0f, 2.0f});
    m.Environment().newProperty<int, 3>("int_prop_array", {2, 3, 4});
    m.Environment().newProperty<unsigned int, 4>("uint_prop_array", {3, 4, 5, 6});

    // Define logging configs
    LoggingConfig lcfg(m);
    AgentLoggingConfig alcfg = lcfg.agent(AGENT_NAME1);
    alcfg.logCount();
    logAllAgent<float>(alcfg, "float_var");
    logAllAgent<int>(alcfg, "int_var");
    logAllAgent<unsigned int>(alcfg, "uint_var");
    lcfg.logEnvironment("float_prop");
    lcfg.logEnvironment("int_prop");
    lcfg.logEnvironment("uint_prop");
    lcfg.logEnvironment("float_prop_array");
    lcfg.logEnvironment("int_prop_array");
    lcfg.logEnvironment("uint_prop_array");

    StepLoggingConfig slcfg(lcfg);
    slcfg.setFrequency(2);

    // Create agent population
    AgentPopulation pop(a, 101);
    for (int i = 0; i < 101; ++i) {
        auto instance = pop.getNextInstance();
        instance.setVariable<float>("float_var", static_cast<float>(i));
        instance.setVariable<int>("int_var", static_cast<int>(i+1));
        instance.setVariable<unsigned int>("uint_var", static_cast<unsigned int>(i+2));
    }

    // Run model
    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 10;
    sim.setStepLog(slcfg);
    sim.setExitLog(lcfg);
    sim.setPopulationData(pop);

    // Step log 5 individual times, and check step logs match expectations
    for (unsigned int i = 1; i <= 10; ++i) {
        sim.step();
        const auto &log = sim.getRunLog();
        const auto &steps = log.getStepLog();
        EXPECT_EQ(steps.size(), i/2);  // Step log frequency works as intended
    }
    {
        const auto &log = sim.getRunLog();
        const auto &steps = log.getStepLog();
        unsigned int step_index = 2;
        for (const auto &step : steps) {
            ASSERT_EQ(step.getStepCount(), step_index);
            // Agent step logging works
            EXPECT_EQ(step.getAgents().size(), 1u);
            auto agent_log = step.getAgent(AGENT_NAME1);
            EXPECT_EQ(101u, agent_log.getCount());

            EXPECT_EQ(100.0f + step_index, agent_log.getMax<float>("float_var"));
            EXPECT_EQ(static_cast<int>(101 + step_index), agent_log.getMax<int>("int_var"));
            EXPECT_EQ(102.0 + step_index, agent_log.getMax<unsigned int>("uint_var"));

            EXPECT_EQ(0.0f + step_index, agent_log.getMin<float>("float_var"));
            EXPECT_EQ(static_cast<int>(1 + step_index), agent_log.getMin<int>("int_var"));
            EXPECT_EQ(2.0 + step_index, agent_log.getMin<unsigned int>("uint_var"));

            EXPECT_EQ(50.0 + step_index, agent_log.getMean("float_var"));
            EXPECT_EQ(51.0 + step_index, agent_log.getMean("int_var"));
            EXPECT_EQ(52.0 + step_index, agent_log.getMean("uint_var"));

            EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("float_var")));  // Test value calculated with excel
            EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("int_var")));
            EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("uint_var")));

            EXPECT_EQ((50.0 + step_index) * 101, agent_log.getSum<float>("float_var"));
            EXPECT_EQ(static_cast<int>(51 + step_index) * 101, agent_log.getSum<int>("int_var"));
            EXPECT_EQ((52 + step_index) * 101, agent_log.getSum<unsigned int>("uint_var"));

            // Env step logging works
            ASSERT_EQ(step.getEnvironmentProperty<float>("float_prop"), 1.0f + step_index);
            ASSERT_EQ(step.getEnvironmentProperty<int>("int_prop"), static_cast<int>(1 + step_index));
            ASSERT_EQ(step.getEnvironmentProperty<unsigned int>("uint_prop"), 1 + step_index);

            const auto f_a = step.getEnvironmentProperty<float, 2>("float_prop_array");
            ASSERT_EQ(f_a[0], 1.0f + step_index);
            ASSERT_EQ(f_a[1], 2.0f + step_index);
            const auto i_a = step.getEnvironmentProperty<int, 3>("int_prop_array");
            ASSERT_EQ(i_a[0], static_cast<int>(2 + step_index));
            ASSERT_EQ(i_a[1], static_cast<int>(3 + step_index));
            ASSERT_EQ(i_a[2], static_cast<int>(4 + step_index));
            const auto u_a = step.getEnvironmentProperty<unsigned int, 4>("uint_prop_array");
            ASSERT_EQ(u_a[0], 3 + step_index);
            ASSERT_EQ(u_a[1], 4 + step_index);
            ASSERT_EQ(u_a[2], 5 + step_index);
            ASSERT_EQ(u_a[3], 6 + step_index);

            step_index+=2;
        }
    }
}
TEST(LoggingTest, CUDASimulationSimulate) {
    /**
     * Ensure the expected data is logged when CUDASimulation::simulate() is called
     * Note: does not check files logged to disk
     */
    // Define model
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    a.newVariable<float>("float_var");
    a.newVariable<int>("int_var");
    a.newVariable<unsigned int>("uint_var");
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    m.newLayer().addAgentFunction(f1);
    m.addStepFunction(step_fn1);
    m.Environment().newProperty<float>("float_prop", 1.0f);
    m.Environment().newProperty<int>("int_prop", 1);
    m.Environment().newProperty<unsigned int>("uint_prop", 1);
    m.Environment().newProperty<float, 2>("float_prop_array", {1.0f, 2.0f});
    m.Environment().newProperty<int, 3>("int_prop_array", {2, 3, 4});
    m.Environment().newProperty<unsigned int, 4>("uint_prop_array", {3, 4, 5, 6});

    // Define logging configs
    LoggingConfig lcfg(m);
    AgentLoggingConfig alcfg = lcfg.agent(AGENT_NAME1);
    alcfg.logCount();
    logAllAgent<float>(alcfg, "float_var");
    logAllAgent<int>(alcfg, "int_var");
    logAllAgent<unsigned int>(alcfg, "uint_var");
    lcfg.logEnvironment("float_prop");
    lcfg.logEnvironment("int_prop");
    lcfg.logEnvironment("uint_prop");
    lcfg.logEnvironment("float_prop_array");
    lcfg.logEnvironment("int_prop_array");
    lcfg.logEnvironment("uint_prop_array");

    StepLoggingConfig slcfg(lcfg);
    slcfg.setFrequency(2);

    // Create agent population
    AgentPopulation pop(a, 101);
    for (int i = 0; i < 101; ++i) {
        auto instance = pop.getNextInstance();
        instance.setVariable<float>("float_var", static_cast<float>(i));
        instance.setVariable<int>("int_var", static_cast<int>(i+1));
        instance.setVariable<unsigned int>("uint_var", static_cast<unsigned int>(i+2));
    }

    // Run model
    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 10;
    // sim.SimulationConfig().common_log_file = "commmon.json";
    // sim.SimulationConfig().step_log_file = "step.json";
    // sim.SimulationConfig().exit_log_file = "exit.json";
    sim.SimulationConfig().steps = 10;
    sim.setStepLog(slcfg);
    sim.setExitLog(lcfg);
    sim.setPopulationData(pop);
    // Call simulate(), and check step and exit logs match expectations
    sim.simulate();
    {  // Check step log
        const auto &log = sim.getRunLog();
        const auto &steps = log.getStepLog();
        EXPECT_EQ(steps.size(), 6);  // init log, + 5 logs from 10 steps
        unsigned int step_index = 0;
        for (const auto &step : steps) {
            ASSERT_EQ(step.getStepCount(), step_index);
            // Agent step logging works
            EXPECT_EQ(step.getAgents().size(), 1u);
            auto agent_log = step.getAgent(AGENT_NAME1);

            EXPECT_EQ(101u, agent_log.getCount());
            EXPECT_EQ(100.0f + step_index, agent_log.getMax<float>("float_var"));
            EXPECT_EQ(static_cast<int>(101 + step_index), agent_log.getMax<int>("int_var"));
            EXPECT_EQ(102.0 + step_index, agent_log.getMax<unsigned int>("uint_var"));

            EXPECT_EQ(0.0f + step_index, agent_log.getMin<float>("float_var"));
            EXPECT_EQ(static_cast<int>(1 + step_index), agent_log.getMin<int>("int_var"));
            EXPECT_EQ(2.0 + step_index, agent_log.getMin<unsigned int>("uint_var"));

            EXPECT_EQ(50.0 + step_index, agent_log.getMean("float_var"));
            EXPECT_EQ(51.0 + step_index, agent_log.getMean("int_var"));
            EXPECT_EQ(52.0 + step_index, agent_log.getMean("uint_var"));

            EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("float_var")));  // Test value calculated with excel
            EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("int_var")));
            EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("uint_var")));

            EXPECT_EQ((50.0 + step_index) * 101, agent_log.getSum<float>("float_var"));
            EXPECT_EQ(static_cast<int>(51 + step_index) * 101, agent_log.getSum<int>("int_var"));
            EXPECT_EQ((52 + step_index) * 101, agent_log.getSum<unsigned int>("uint_var"));

            // Env step logging works
            ASSERT_EQ(step.getEnvironmentProperty<float>("float_prop"), 1.0f + step_index);
            ASSERT_EQ(step.getEnvironmentProperty<int>("int_prop"), static_cast<int>(1 + step_index));
            ASSERT_EQ(step.getEnvironmentProperty<unsigned int>("uint_prop"), 1 + step_index);

            const auto f_a = step.getEnvironmentProperty<float, 2>("float_prop_array");
            ASSERT_EQ(f_a[0], 1.0f + step_index);
            ASSERT_EQ(f_a[1], 2.0f + step_index);
            const auto i_a = step.getEnvironmentProperty<int, 3>("int_prop_array");
            ASSERT_EQ(i_a[0], static_cast<int>(2 + step_index));
            ASSERT_EQ(i_a[1], static_cast<int>(3 + step_index));
            ASSERT_EQ(i_a[2], static_cast<int>(4 + step_index));
            const auto u_a = step.getEnvironmentProperty<unsigned int, 4>("uint_prop_array");
            ASSERT_EQ(u_a[0], 3 + step_index);
            ASSERT_EQ(u_a[1], 4 + step_index);
            ASSERT_EQ(u_a[2], 5 + step_index);
            ASSERT_EQ(u_a[3], 6 + step_index);

            step_index+=2;
        }
    }
    {  // Check exit log, should match final step log
        const auto &log = sim.getRunLog();
        const unsigned int step_index = 10;
        const auto &exit = log.getExitLog();
        ASSERT_EQ(exit.getStepCount(), step_index);
        // Agent step logging works
        EXPECT_EQ(exit.getAgents().size(), 1u);
        auto agent_log = exit.getAgent(AGENT_NAME1);
        EXPECT_EQ(101u, agent_log.getCount());

        EXPECT_EQ(100.0f + step_index, agent_log.getMax<float>("float_var"));
        EXPECT_EQ(static_cast<int>(101 + step_index), agent_log.getMax<int>("int_var"));
        EXPECT_EQ(102.0 + step_index, agent_log.getMax<unsigned int>("uint_var"));

        EXPECT_EQ(0.0f + step_index, agent_log.getMin<float>("float_var"));
        EXPECT_EQ(static_cast<int>(1 + step_index), agent_log.getMin<int>("int_var"));
        EXPECT_EQ(2.0 + step_index, agent_log.getMin<unsigned int>("uint_var"));

        EXPECT_EQ(50.0 + step_index, agent_log.getMean("float_var"));
        EXPECT_EQ(51.0 + step_index, agent_log.getMean("int_var"));
        EXPECT_EQ(52.0 + step_index, agent_log.getMean("uint_var"));

        EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("float_var")));  // Test value calculated with excel
        EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("int_var")));
        EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("uint_var")));

        EXPECT_EQ((50.0 + step_index) * 101, agent_log.getSum<float>("float_var"));
        EXPECT_EQ(static_cast<int>(51 + step_index) * 101, agent_log.getSum<int>("int_var"));
        EXPECT_EQ((52 + step_index) * 101, agent_log.getSum<unsigned int>("uint_var"));

        // Env step logging works
        ASSERT_EQ(exit.getEnvironmentProperty<float>("float_prop"), 1.0f + step_index);
        ASSERT_EQ(exit.getEnvironmentProperty<int>("int_prop"), static_cast<int>(1 + step_index));
        ASSERT_EQ(exit.getEnvironmentProperty<unsigned int>("uint_prop"), 1 + step_index);

        const auto f_a = exit.getEnvironmentProperty<float, 2>("float_prop_array");
        ASSERT_EQ(f_a[0], 1.0f + step_index);
        ASSERT_EQ(f_a[1], 2.0f + step_index);
        const auto i_a = exit.getEnvironmentProperty<int, 3>("int_prop_array");
        ASSERT_EQ(i_a[0], static_cast<int>(2 + step_index));
        ASSERT_EQ(i_a[1], static_cast<int>(3 + step_index));
        ASSERT_EQ(i_a[2], static_cast<int>(4 + step_index));
        const auto u_a = exit.getEnvironmentProperty<unsigned int, 4>("uint_prop_array");
        ASSERT_EQ(u_a[0], 3 + step_index);
        ASSERT_EQ(u_a[1], 4 + step_index);
        ASSERT_EQ(u_a[2], 5 + step_index);
        ASSERT_EQ(u_a[3], 6 + step_index);
    }
}
FLAMEGPU_INIT_FUNCTION(logging_ensemble_init) {
    const int instance_id  = FLAMEGPU->environment.getProperty<int>("instance_id");
    for (int i = instance_id; i < instance_id + 101; ++i) {
        auto instance = FLAMEGPU->newAgent(AGENT_NAME1);
        instance.setVariable<float>("float_var", static_cast<float>(i));
        instance.setVariable<int>("int_var", static_cast<int>(i+1));
        instance.setVariable<unsigned int>("uint_var", static_cast<unsigned int>(i+2));
    }
}
TEST(LoggingTest, CUDAEnsembleSimulate) {
    /**
     * Ensure the expected data is logged when CUDAEnsemble::simulate() is called
     * Note: does not check files logged to disk
     */
    // Define model
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    a.newVariable<float>("float_var");
    a.newVariable<int>("int_var");
    a.newVariable<unsigned int>("uint_var");
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    m.addInitFunction(logging_ensemble_init);
    m.newLayer().addAgentFunction(f1);
    m.addStepFunction(step_fn1);
    m.Environment().newProperty<int>("instance_id", 0);  // This will act as the modifier for ensemble instances, only impacting the init fn
    m.Environment().newProperty<float>("float_prop", 1.0f);
    m.Environment().newProperty<int>("int_prop", 1);
    m.Environment().newProperty<unsigned int>("uint_prop", 1);
    m.Environment().newProperty<float, 2>("float_prop_array", {1.0f, 2.0f});
    m.Environment().newProperty<int, 3>("int_prop_array", {2, 3, 4});
    m.Environment().newProperty<unsigned int, 4>("uint_prop_array", {3, 4, 5, 6});

    // Define logging configs
    LoggingConfig lcfg(m);
    AgentLoggingConfig alcfg = lcfg.agent(AGENT_NAME1);
    alcfg.logCount();
    logAllAgent<float>(alcfg, "float_var");
    logAllAgent<int>(alcfg, "int_var");
    logAllAgent<unsigned int>(alcfg, "uint_var");
    lcfg.logEnvironment("float_prop");
    lcfg.logEnvironment("instance_id");
    lcfg.logEnvironment("int_prop");
    lcfg.logEnvironment("uint_prop");
    lcfg.logEnvironment("float_prop_array");
    lcfg.logEnvironment("int_prop_array");
    lcfg.logEnvironment("uint_prop_array");

    StepLoggingConfig slcfg(lcfg);
    slcfg.setFrequency(2);

    // Set up the runplan
    RunPlanVec plan(m, 10);
    int i_id = 0;
    for (auto &p : plan) {
        p.setSteps(10);
        p.setProperty<int>("instance_id", i_id++);
        // p.setOutputSubdirectory(i_id%2 == 0 ? "a" : "b");
    }

    // Run model
    CUDAEnsemble sim(m);
    sim.Config().concurrent_runs = 5;
    sim.Config().silent = true;
    sim.Config().timing = false;
    // sim.Config().out_directory = "ensemble_out";
    // sim.Config().out_format = "json";
    sim.setStepLog(slcfg);
    sim.setExitLog(lcfg);
    // Call simulate(), and check step and exit logs match expectations
    sim.simulate(plan);
    {  // Check step log
        const auto &run_logs = sim.getLogs();
        i_id = 0;
        for (auto &log : run_logs) {
            {  // Check step log
                auto &steps = log.getStepLog();
                EXPECT_EQ(steps.size(), 6);  // init log, + 5 logs from 10 steps
                unsigned int step_index = 0;
                for (const auto &step : steps) {
                    // Log corresponds to the correct instance
                    ASSERT_EQ(step.getEnvironmentProperty<int>("instance_id"), i_id);
                    ASSERT_EQ(step.getStepCount(), step_index);
                    // Agent step logging works
                    EXPECT_EQ(step.getAgents().size(), 1u);
                    auto agent_log = step.getAgent(AGENT_NAME1);

                    EXPECT_EQ(101u, agent_log.getCount());
                    EXPECT_EQ(100.0f + step_index + i_id, agent_log.getMax<float>("float_var"));
                    EXPECT_EQ(static_cast<int>(101 + step_index + i_id), agent_log.getMax<int>("int_var"));
                    EXPECT_EQ(102.0 + step_index + i_id, agent_log.getMax<unsigned int>("uint_var"));

                    EXPECT_EQ(0.0f + step_index + i_id, agent_log.getMin<float>("float_var"));
                    EXPECT_EQ(static_cast<int>(1 + step_index + i_id), agent_log.getMin<int>("int_var"));
                    EXPECT_EQ(2.0 + step_index + i_id, agent_log.getMin<unsigned int>("uint_var"));

                    EXPECT_EQ(50.0 + step_index + i_id, agent_log.getMean("float_var"));
                    EXPECT_EQ(51.0 + step_index + i_id, agent_log.getMean("int_var"));
                    EXPECT_EQ(52.0 + step_index + i_id, agent_log.getMean("uint_var"));

                    EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("float_var")));  // Test value calculated with excel
                    EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("int_var")));
                    EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("uint_var")));

                    EXPECT_EQ((50.0 + step_index + i_id) * 101, agent_log.getSum<float>("float_var"));
                    EXPECT_EQ(static_cast<int>(51 + step_index + i_id) * 101, agent_log.getSum<int>("int_var"));
                    EXPECT_EQ((52 + step_index + i_id) * 101, agent_log.getSum<unsigned int>("uint_var"));

                    // Env step logging works
                    ASSERT_EQ(step.getEnvironmentProperty<float>("float_prop"), 1.0f + step_index);
                    ASSERT_EQ(step.getEnvironmentProperty<int>("int_prop"), static_cast<int>(1 + step_index));
                    ASSERT_EQ(step.getEnvironmentProperty<unsigned int>("uint_prop"), 1 + step_index);

                    const auto f_a = step.getEnvironmentProperty<float, 2>("float_prop_array");
                    ASSERT_EQ(f_a[0], 1.0f + step_index);
                    ASSERT_EQ(f_a[1], 2.0f + step_index);
                    const auto i_a = step.getEnvironmentProperty<int, 3>("int_prop_array");
                    ASSERT_EQ(i_a[0], static_cast<int>(2 + step_index));
                    ASSERT_EQ(i_a[1], static_cast<int>(3 + step_index));
                    ASSERT_EQ(i_a[2], static_cast<int>(4 + step_index));
                    const auto u_a = step.getEnvironmentProperty<unsigned int, 4>("uint_prop_array");
                    ASSERT_EQ(u_a[0], 3 + step_index);
                    ASSERT_EQ(u_a[1], 4 + step_index);
                    ASSERT_EQ(u_a[2], 5 + step_index);
                    ASSERT_EQ(u_a[3], 6 + step_index);

                    step_index+=2;
                }
            }
            {  // Check exit log, should match final step log
                const unsigned int step_index = 10;
                const auto &exit = log.getExitLog();
                ASSERT_EQ(exit.getStepCount(), step_index);
                // Agent step logging works
                EXPECT_EQ(exit.getAgents().size(), 1u);
                auto agent_log = exit.getAgent(AGENT_NAME1);
                EXPECT_EQ(101u, agent_log.getCount());

                EXPECT_EQ(100.0f + step_index + i_id, agent_log.getMax<float>("float_var"));
                EXPECT_EQ(static_cast<int>(101 + step_index + i_id), agent_log.getMax<int>("int_var"));
                EXPECT_EQ(102.0 + step_index + i_id, agent_log.getMax<unsigned int>("uint_var"));

                EXPECT_EQ(0.0f + step_index + i_id, agent_log.getMin<float>("float_var"));
                EXPECT_EQ(static_cast<int>(1 + step_index + i_id), agent_log.getMin<int>("int_var"));
                EXPECT_EQ(2.0 + step_index + i_id, agent_log.getMin<unsigned int>("uint_var"));

                EXPECT_EQ(50.0 + step_index + i_id, agent_log.getMean("float_var"));
                EXPECT_EQ(51.0 + step_index + i_id, agent_log.getMean("int_var"));
                EXPECT_EQ(52.0 + step_index + i_id, agent_log.getMean("uint_var"));

                EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("float_var")));  // Test value calculated with excel
                EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("int_var")));
                EXPECT_FLOAT_EQ(29.15476f, static_cast<float>(agent_log.getStandardDev("uint_var")));

                EXPECT_EQ((50.0 + step_index + i_id) * 101, agent_log.getSum<float>("float_var"));
                EXPECT_EQ(static_cast<int>(51 + step_index + i_id) * 101, agent_log.getSum<int>("int_var"));
                EXPECT_EQ((52 + step_index + i_id) * 101, agent_log.getSum<unsigned int>("uint_var"));

                // Env step logging works
                ASSERT_EQ(exit.getEnvironmentProperty<float>("float_prop"), 1.0f + step_index);
                ASSERT_EQ(exit.getEnvironmentProperty<int>("int_prop"), static_cast<int>(1 + step_index));
                ASSERT_EQ(exit.getEnvironmentProperty<unsigned int>("uint_prop"), 1 + step_index);

                const auto f_a = exit.getEnvironmentProperty<float, 2>("float_prop_array");
                ASSERT_EQ(f_a[0], 1.0f + step_index);
                ASSERT_EQ(f_a[1], 2.0f + step_index);
                const auto i_a = exit.getEnvironmentProperty<int, 3>("int_prop_array");
                ASSERT_EQ(i_a[0], static_cast<int>(2 + step_index));
                ASSERT_EQ(i_a[1], static_cast<int>(3 + step_index));
                ASSERT_EQ(i_a[2], static_cast<int>(4 + step_index));
                const auto u_a = exit.getEnvironmentProperty<unsigned int, 4>("uint_prop_array");
                ASSERT_EQ(u_a[0], 3 + step_index);
                ASSERT_EQ(u_a[1], 4 + step_index);
                ASSERT_EQ(u_a[2], 5 + step_index);
                ASSERT_EQ(u_a[3], 6 + step_index);
            }
            ++i_id;
        }
    }
}

}  // namespace test_logging
