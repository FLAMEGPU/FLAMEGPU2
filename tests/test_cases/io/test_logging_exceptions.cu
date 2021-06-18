#include "gtest/gtest.h"

#include "flamegpu/flamegpu.h"

// Need to test
// LogFrame
// AgentLogFrame

namespace flamegpu {


namespace test_logging_exception {
const char *MODEL_NAME = "Model";
const char *AGENT_NAME1 = "Agent1";
const char *AGENT_NAME2 = "Agent2";

TEST(LoggingExceptionTest, LoggerSupportedFileType) {
    // WriterFactory::createLogger() - UnsupportedFileType
    ModelDescription m(MODEL_NAME);
    m.newAgent(AGENT_NAME1);
    CUDASimulation sim(m);
    EXPECT_THROW(sim.exportLog("test.csv", true, true), UnsupportedFileType);
    EXPECT_THROW(sim.exportLog("test.html", true, true), UnsupportedFileType);
    EXPECT_NO_THROW(sim.exportLog("test.json", true, true));
    // Cleanup
    ASSERT_EQ(::remove("test.json"), 0);
}

TEST(LoggingExceptionTest, LoggingConfigExceptions) {
    // Define model
    ModelDescription m(MODEL_NAME);
    m.newAgent(AGENT_NAME1);
    m.Environment().newProperty<float>("float_prop", 1.0f);

    LoggingConfig lcfg(m);
    // Property doesn't exist
    EXPECT_THROW(lcfg.logEnvironment("int_prop"), InvalidEnvProperty);
    // Property does exist
    EXPECT_NO_THROW(lcfg.logEnvironment("float_prop"));
    // Property already marked for logging
    EXPECT_THROW(lcfg.logEnvironment("float_prop"), InvalidEnvProperty);
    // THIS DOES NOT WORK, cfg holds a copy of the ModelDescription, not a reference to it.
    // Add new property after lcfg made
    EXPECT_NO_THROW(m.Environment().newProperty<int>("int_prop", 1));
    // Property does not exist
    EXPECT_THROW(lcfg.logEnvironment("int_prop"), InvalidEnvProperty);

    // Agent does not exist
    EXPECT_THROW(lcfg.agent(AGENT_NAME2, "state2"), InvalidAgentName);
    // Agent state does not exist
    EXPECT_THROW(lcfg.agent(AGENT_NAME1, "state2"), InvalidAgentState);
    // Agent/State does exist
    EXPECT_NO_THROW(lcfg.agent(AGENT_NAME1, flamegpu::ModelData::DEFAULT_STATE));
    // THIS DOES NOT WORK, cfg holds a copy of the ModelDescription, not a reference to it.
    // Add new agent after lcfg
    AgentDescription &a2 = m.newAgent(AGENT_NAME2);
    a2.newState("state2");
    // Agent/State does not exist
    EXPECT_THROW(lcfg.agent(AGENT_NAME2, "state2"), InvalidAgentName);
}
TEST(LoggingExceptionTest, AgentLoggingConfigExceptions) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    a.newVariable<float>("float_var");
    a.newVariable<float, 2>("float_var_array");

    LoggingConfig lcfg(m);
    auto alcfg = lcfg.agent(AGENT_NAME1, flamegpu::ModelData::DEFAULT_STATE);

    // Log functions all pass to the same common method which contains the checks
    // Test 1, test them all (mean, standard dev, min, max, sum)
    // Bad variable name
    EXPECT_THROW(alcfg.logMean<int>("int_var"), InvalidAgentVar);
    // Type does not match variable name
    EXPECT_THROW(alcfg.logMean<int>("float_var"), InvalidVarType);
    // Array variables are not supported
    EXPECT_THROW(alcfg.logMean<float>("float_var_array"), InvalidVarType);
    // Variable is correct
    EXPECT_NO_THROW(alcfg.logMean<float>("float_var"));
    // Variable has already been marked for logging
    EXPECT_THROW(alcfg.logMean<float>("float_var"), InvalidArgument);
    // THIS DOES NOT WORK, cfg holds a copy of the ModelDescription, not a reference to it.
    // Add new agent var after log creation
    a.newVariable<int>("int_var");
    // Variable is not found
    EXPECT_THROW(alcfg.logMean<int>("int_var"), InvalidAgentVar);
}
TEST(LoggingExceptionTest, LogFrameExceptions) {
    // Define model
    ModelDescription m(MODEL_NAME);
    m.Environment().newProperty<float>("float_prop", 1.0f);
    m.Environment().newProperty<int>("int_prop", 1);
    m.Environment().newProperty<unsigned int>("uint_prop", 1);
    m.Environment().newProperty<float, 2>("float_prop_array", {1.0f, 2.0f});
    m.Environment().newProperty<int, 3>("int_prop_array", {2, 3, 4});
    m.Environment().newProperty<unsigned int, 4>("uint_prop_array", {3, 4, 5, 6});

    // Define logging configs
    LoggingConfig lcfg(m);

    StepLoggingConfig slcfg(lcfg);
    slcfg.logEnvironment("float_prop");
    slcfg.logEnvironment("uint_prop_array");

    // Run model
    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 1;
    sim.setStepLog(slcfg);

    sim.step();

    // Fetch log
    const auto &log = sim.getRunLog();
    const auto &steps = log.getStepLog();
    ASSERT_EQ(steps.size(), 1);
    auto &slog = *steps.begin();
    // Property wasn't logged
    EXPECT_THROW(slog.getEnvironmentProperty<float>("float_prop2"), InvalidEnvProperty);
    EXPECT_THROW(slog.getEnvironmentProperty<int>("int_prop"), InvalidEnvProperty);
    auto f1 = &LogFrame::getEnvironmentProperty<float, 2>;  // Hack for 2 arg templates inside macro fn
    auto f2 = &LogFrame::getEnvironmentProperty<int, 2>;  // Hack for 2 arg templates inside macro fn
    EXPECT_THROW((slog.*f1)("float_prop2"), InvalidEnvProperty);
    EXPECT_THROW((slog.*f2)("int_prop"), InvalidEnvProperty);
    // Property wrong type
    EXPECT_THROW(slog.getEnvironmentProperty<int>("float_prop"), InvalidEnvPropertyType);
    auto f3 = &LogFrame::getEnvironmentProperty<int, 4>;  // Hack for 2 arg templates inside macro fn
    EXPECT_THROW((slog.*f3)("uint_prop_array"), InvalidEnvPropertyType);
    // Property wrong length
    EXPECT_THROW(slog.getEnvironmentProperty<unsigned int>("uint_prop_array"), InvalidEnvPropertyType);
    auto f4 = &LogFrame::getEnvironmentProperty<unsigned int, 2>;  // Hack for 2 arg templates inside macro fn
    EXPECT_THROW((slog.*f4)("uint_prop_array"), InvalidEnvPropertyType);
    // Correct property settings work
    EXPECT_NO_THROW(slog.getEnvironmentProperty<float>("float_prop"));
    auto f5 = &LogFrame::getEnvironmentProperty<unsigned int, 4>;  // Hack for 2 arg templates inside macro fn
    EXPECT_NO_THROW((slog.*f5)("uint_prop_array"));
}
TEST(LoggingExceptionTest, AgentLogFrameExceptions) {
    // Define model
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME1);
    a.newVariable<float>("float_var");
    a.newVariable<int>("int_var");
    a.newVariable<unsigned int>("uint_var");
    a.newVariable<float, 2>("float_var_array");

    // Define logging configs
    LoggingConfig lcfg(m);
    StepLoggingConfig slcfg(lcfg);
    AgentLoggingConfig alcfg = slcfg.agent(AGENT_NAME1);
    alcfg.logMean<float>("float_var");
    alcfg.logStandardDev<float>("float_var");
    alcfg.logMin<int>("int_var");
    alcfg.logMax<int>("int_var");
    alcfg.logSum<unsigned int>("uint_var");

    // Run model
    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 1;
    sim.setStepLog(slcfg);
    sim.step();

    // Fetch log
    const auto &log = sim.getRunLog();
    const auto &steps = log.getStepLog();
    ASSERT_EQ(steps.size(), 1);
    auto &slog = *steps.begin();
    // Agent/state was not logged
    EXPECT_THROW(slog.getAgent("wrong_agent"), InvalidAgentState);
    EXPECT_THROW(slog.getAgent(AGENT_NAME1, "wrong_state"), InvalidAgentState);
    auto alog = slog.getAgent(AGENT_NAME1);
    // Count was not logged
    EXPECT_THROW(alog.getCount(), InvalidOperation);
    // Variable/Reduction wasn't logged
    EXPECT_THROW(alog.getMean("int_var"), InvalidAgentVar);
    EXPECT_THROW(alog.getStandardDev("double_var"), InvalidAgentVar);
    EXPECT_THROW(alog.getMin<float>("float_var"), InvalidAgentVar);
    EXPECT_THROW(alog.getMax<float>("float_var"), InvalidAgentVar);
    EXPECT_THROW(alog.getSum<unsigned int>("uint_prop_array"), InvalidAgentVar);
    // Property wrong type
    EXPECT_THROW(alog.getMin<float>("int_var"), InvalidVarType);
    EXPECT_THROW(alog.getMax<float>("int_var"), InvalidVarType);
    EXPECT_THROW(alog.getSum<int>("uint_var"), InvalidVarType);
    // Correct property settings work
    EXPECT_NO_THROW(alog.getMean("float_var"));
    EXPECT_NO_THROW(alog.getStandardDev("float_var"));
    EXPECT_NO_THROW(alog.getMin<int>("int_var"));
    EXPECT_NO_THROW(alog.getMax<int>("int_var"));
    EXPECT_NO_THROW(alog.getSum<unsigned int>("uint_var"));
}

}  // namespace test_logging_exception
}  // namespace flamegpu
