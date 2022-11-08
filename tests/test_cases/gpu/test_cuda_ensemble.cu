#include <thread>
#include <chrono>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace tests {
namespace test_cuda_ensemble {

TEST(TestCUDAEnsemble, constructor) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Declare a pointer
    flamegpu::CUDAEnsemble * ensemble = nullptr;
    // Use the ctor
    // explicit CUDAEnsemble(const ModelDescription& model, int argc = 0, const char** argv = nullptr);
    EXPECT_NO_THROW(ensemble = new flamegpu::CUDAEnsemble(model, 0, nullptr));
    EXPECT_NE(ensemble, nullptr);
    // Check a property
    EXPECT_EQ(ensemble->Config().timing, false);
    // Run the destructor ~CUDAEnsemble
    EXPECT_NO_THROW(delete ensemble);
    ensemble = nullptr;
    // Check with simple argparsing.
    const char *argv[2] = { "prog.exe", "--timing" };
    EXPECT_NO_THROW(ensemble = new flamegpu::CUDAEnsemble(model, sizeof(argv) / sizeof(char*), argv));
    EXPECT_EQ(ensemble->Config().timing, true);
    EXPECT_NO_THROW(delete ensemble);
    ensemble = nullptr;
}
TEST(TestCUDAEnsemble, EnsembleConfig) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Get a config object.
    // EnsembleConfig &Config()
    EXPECT_NO_THROW(ensemble.Config());
    auto &mutableConfig = ensemble.Config();
    // Get a const'd version.
    // const EnsembleConfig &getConfig()
    EXPECT_NO_THROW(ensemble.getConfig());
    auto &immutableConfig = ensemble.getConfig();
    // Check the default values are correct.
    EXPECT_EQ(immutableConfig.out_directory, "");
    EXPECT_EQ(immutableConfig.out_format, "json");
    EXPECT_EQ(immutableConfig.concurrent_runs, 4u);
    EXPECT_EQ(immutableConfig.devices, std::set<int>());  // @todo - this will need to change.
    EXPECT_EQ(immutableConfig.verbosity, Verbosity::Default);
    EXPECT_EQ(immutableConfig.timing, false);
    // Mutate the config. Note we cannot mutate the return from getConfig, and connot test this as it is a compialtion failure (requires ctest / standalone .cpp file)
    mutableConfig.out_directory = std::string("test");
    mutableConfig.out_format = std::string("xml");
    mutableConfig.concurrent_runs = 1;
    mutableConfig.devices = std::set<int>({0});
    mutableConfig.verbosity = Verbosity::Verbose;
    mutableConfig.timing = true;
    // Check via the const ref, this should show the same value as config was a reference, not a copy.
    EXPECT_EQ(immutableConfig.out_directory, "test");
    EXPECT_EQ(immutableConfig.out_format, "xml");
    EXPECT_EQ(immutableConfig.concurrent_runs, 1u);
    EXPECT_EQ(immutableConfig.devices, std::set<int>({0}));  // @todo - this will need to change.
    EXPECT_EQ(immutableConfig.verbosity, Verbosity::Verbose);
    EXPECT_EQ(immutableConfig.timing, true);
}
// This test causes `exit` so cannot be used.
/* TEST(TestCUDAEnsemble, DISABLED_initialise_help) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    const char *argv[2] = { "prog.exe", "--help" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
} */
TEST(TestCUDAEnsemble, initialise_out) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().out_directory, "");
    EXPECT_EQ(ensemble.getConfig().out_format, "json");
    const char *argv[4] = { "prog.exe", "--out", "test", "xml" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(ensemble.getConfig().out_directory, "test");
    EXPECT_EQ(ensemble.getConfig().out_format, "xml");
}
TEST(TestCUDAEnsemble, initialise_concurrent_runs) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().concurrent_runs, 4u);
    const char *argv[3] = { "prog.exe", "--concurrent", "2" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(ensemble.getConfig().concurrent_runs, 2u);
}
TEST(TestCUDAEnsemble, initialise_devices) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().devices, std::set<int>({}));
    const char *argv[3] = { "prog.exe", "--devices", "0" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(ensemble.getConfig().devices, std::set<int>({0}));
}
TEST(TestCUDAEnsemble, initialise_devices_wrong) {
    flamegpu::ModelDescription model("test");
    flamegpu::CUDAEnsemble ensemble(model);
    int device_ct = -1;
    EXPECT_EQ(cudaGetDeviceCount(&device_ct), cudaSuccess);
    ensemble.Config().devices = { device_ct };
    RunPlanVector plan(model, 1);
    // Sim with out of bounds device ID, get exception
    EXPECT_THROW(ensemble.simulate(plan), exception::InvalidCUDAdevice);
}
TEST(TestCUDAEnsemble, initialise_unknown_arg) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().devices, std::set<int>({}));
    const char* argv[2] = { "prog.exe", "--unknown" };
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(ensemble.initialise(sizeof(argv) / sizeof(char*), argv));  //  No exception but warning shoudl be raised
    std::string errors = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(errors.find("Warning: Unknown argument") != std::string::npos);  // Should be found
}
TEST(TestCUDAEnsemble, initialise_unknown_arg_quiet) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().devices, std::set<int>({}));
    const char* argv[3] = { "prog.exe", "--quiet", "--unknown" };
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(ensemble.initialise(sizeof(argv) / sizeof(char*), argv));  //  No exception but warning shoudl be raised
    std::string errors = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(errors.find("Warning: Unknown argument") == std::string::npos);  // Should NOT be found
}
TEST(TestCUDAEnsemble, initialise_unknown_arg_silenced) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().devices, std::set<int>({}));
    const char* argv[3] = { "prog.exe", "--silence-unknown-args", "--unknown" };
    testing::internal::CaptureStderr();
    EXPECT_NO_THROW(ensemble.initialise(sizeof(argv) / sizeof(char*), argv));  //  No exception but warning shoudl be raised
    std::string errors = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(errors.find("Warning: Unknown argument") == std::string::npos);  // Should NOT be found
}
TEST(TestCUDAEnsemble, initialise_quiet) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().verbosity, Verbosity::Default);
    const char *argv[2] = { "prog.exe", "--quiet" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(ensemble.getConfig().verbosity, Verbosity::Quiet);
}
TEST(TestCUDAEnsemble, initialise_default) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().verbosity, Verbosity::Default);
    const char* argv[1] = { "prog.exe" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(ensemble.getConfig().verbosity, Verbosity::Default);
}
TEST(TestCUDAEnsemble, initialise_verbose) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().verbosity, Verbosity::Default);
    const char* argv[2] = { "prog.exe", "--verbose" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(ensemble.getConfig().verbosity, Verbosity::Verbose);
}
TEST(TestCUDAEnsemble, initialise_timing) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().timing, false);
    const char *argv[2] = { "prog.exe", "--timing" };
    ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(ensemble.getConfig().timing, true);
}
TEST(TestCUDAEnsemble, initialise_error_level) {
    // Create a model
    flamegpu::ModelDescription model("test");
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Call initialise with different cli arguments, which will mutate values. Check they have the new value.
    EXPECT_EQ(ensemble.getConfig().error_level, CUDAEnsemble::EnsembleConfig::Slow);
    {
        const char* argv[3] = { "prog.exe", "-e", "0" };
        ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
        EXPECT_EQ(ensemble.getConfig().error_level, CUDAEnsemble::EnsembleConfig::Off);
    }
    {
        const char* argv[3] = { "prog.exe", "--error", "1" };
        ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
        EXPECT_EQ(ensemble.getConfig().error_level, CUDAEnsemble::EnsembleConfig::Slow);
    }
    {
        const char* argv[3] = { "prog.exe", "-e", "2" };
        ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
        EXPECT_EQ(ensemble.getConfig().error_level, CUDAEnsemble::EnsembleConfig::Fast);
    }
    {
        const char* argv[3] = { "prog.exe", "--error", "Off" };
        ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
        EXPECT_EQ(ensemble.getConfig().error_level, CUDAEnsemble::EnsembleConfig::Off);
    }
    {
        const char* argv[3] = { "prog.exe", "-e", "SLOW" };
        ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
        EXPECT_EQ(ensemble.getConfig().error_level, CUDAEnsemble::EnsembleConfig::Slow);
    }
    {
        const char* argv[3] = { "prog.exe", "--error", "fast" };
        ensemble.initialise(sizeof(argv) / sizeof(char*), argv);
        EXPECT_EQ(ensemble.getConfig().error_level, CUDAEnsemble::EnsembleConfig::Fast);
    }
}
// Agent function used to check the ensemble runs.
FLAMEGPU_AGENT_FUNCTION(simulateAgentFn, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Increment agent's counter by 1.
    FLAMEGPU->setVariable<int>("counter", FLAMEGPU->getVariable<int>("counter") + 1);
    return flamegpu::ALIVE;
}
FLAMEGPU_INIT_FUNCTION(simulateInit) {
    // Generate a basic pop
    const uint32_t POPULATION_TO_GENERATE = FLAMEGPU->environment.getProperty<uint32_t>("POPULATION_TO_GENERATE");
    auto agent = FLAMEGPU->agent("Agent");
    for (uint32_t i = 0; i < POPULATION_TO_GENERATE; ++i) {
        agent.newAgent().setVariable<uint32_t>("counter", 0);
    }
}
// File scoped variables to allow non-loggin based ensemble validation.
std::atomic<uint64_t> testSimulateSumOfSums = {0};
// File scoped atomics
FLAMEGPU_EXIT_FUNCTION(simulateExit) {
    uint64_t totalCounters = FLAMEGPU->agent("Agent").sum<uint32_t>("counter");
    // Add to the  file scoped atomic sum of sums.
    testSimulateSumOfSums += totalCounters;
}
TEST(TestCUDAEnsemble, simulate) {
    // Reset the atomic sum of sums to 0. Just in case.
    testSimulateSumOfSums = 0;
    // Number of simulations to run.
    constexpr uint32_t planCount = 2u;
    constexpr uint32_t populationSize = 32u;
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", populationSize, true);
    // Agent(s)
    flamegpu::AgentDescription &agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    agent.newFunction("simulateAgentFn", simulateAgentFn);
    // Control flow
    model.newLayer().addAgentFunction(simulateAgentFn);
    model.addInitFunction(simulateInit);
    model.addExitFunction(simulateExit);
    // Crete a small runplan, using a different number of steps per sim.
    uint64_t expectedResult = 0;
    flamegpu::RunPlanVector plans(model, planCount);
    for (uint32_t idx = 0; idx < plans.size(); idx++) {
        auto &plan = plans[idx];
        plan.setSteps(idx + 1);  // Can't have 0 steps without exit condition
        // Increment the expected result based on the number of steps.
        expectedResult += (idx + 1) * populationSize;
    }
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    // Simulate the ensemble,
    EXPECT_NO_THROW(ensemble.simulate(plans));
    // Get the sum of sums from the atomic.
    uint64_t atomicResult = testSimulateSumOfSums.load();
    // Compare against the epxected value
    EXPECT_EQ(atomicResult, expectedResult);

    // An exception should be thrown if the Plan and Ensemble are for different models.
    flamegpu::ModelDescription modelTwo("two");
    flamegpu::RunPlanVector modelTwoPlans(modelTwo, 1);
    EXPECT_THROW(ensemble.simulate(modelTwoPlans), flamegpu::exception::InvalidArgument);
    // Exceptions can also be thrown if output_directory cannot be created, but I'm unsure how to reliably test this cross platform.
}
TEST(TestCUDAEnsemble, verbosity) {
    // Reset the atomic sum of sums to 0. Just in case.
    testSimulateSumOfSums = 0;
    // Number of simulations to run.
    constexpr uint32_t planCount = 2u;
    constexpr uint32_t populationSize = 32u;
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", populationSize, true);
    // Agent(s)
    flamegpu::AgentDescription& agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    agent.newFunction("simulateAgentFn", simulateAgentFn);
    // Control flow
    model.newLayer().addAgentFunction(simulateAgentFn);
    model.addInitFunction(simulateInit);
    model.addExitFunction(simulateExit);
    // Crete a small runplan, using a different number of steps per sim.
    uint64_t expectedResult = 0;
    flamegpu::RunPlanVector plans(model, planCount);
    for (uint32_t idx = 0; idx < plans.size(); idx++) {
        auto& plan = plans[idx];
        plan.setSteps(idx + 1);  // Can't have 0 steps without exit condition
        // Increment the expected result based on the number of steps.
        expectedResult += (idx + 1) * populationSize;
    }
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Verbosity::Quiet
    {
        // Capture stderr and stdout
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        // Set verbosity level
        ensemble.Config().verbosity = Verbosity::Quiet;
        // Simulate the ensemble
        EXPECT_NO_THROW(ensemble.simulate(plans));
        // Get stderr and stdout
        std::string output = testing::internal::GetCapturedStdout();
        std::string errors = testing::internal::GetCapturedStderr();
        // Expect no warnings (stderr) or outputs in Quiet mode
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }
    // Verbosity::Default
    {
        // Capture stderr and stdout
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        // Set verbosity level
        ensemble.Config().verbosity = Verbosity::Default;
        // Simulate the ensemble
        EXPECT_NO_THROW(ensemble.simulate(plans));
        // Get stderr and stdout
        std::string output = testing::internal::GetCapturedStdout();
        std::string errors = testing::internal::GetCapturedStderr();
        // Expect no warnings (stderr) but an output reporing the number of completed simulation runs
        std::ostringstream  expect_output;
        EXPECT_TRUE(output.find("CUDAEnsemble progress") != std::string::npos);   // E.g. CUDAEnsemble progress: 1/2
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(output.find("Ensemble time elapsed") == std::string::npos);   // E.g. Ensemble time elapsed: 0.006000s
        EXPECT_TRUE(errors.empty());
    }
    // Verbosity::Verbose
    {
        // Capture stderr and stdout
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        // Set verbosity level
        ensemble.Config().verbosity = Verbosity::Verbose;
        // Simulate the ensemble
        EXPECT_NO_THROW(ensemble.simulate(plans));
        // Get stderr and stdout
        std::string output = testing::internal::GetCapturedStdout();
        std::string errors = testing::internal::GetCapturedStderr();
        // Expect no warnings (stderr) but outputs on progress and timing
        EXPECT_TRUE(output.find("CUDAEnsemble progress") != std::string::npos);   // E.g. CUDAEnsemble progress: 1/2
        EXPECT_TRUE(output.find("CUDAEnsemble completed") != std::string::npos);  // E.g. CUDAEnsemble completed 2 runs successfully!
        EXPECT_TRUE(output.find("Ensemble time elapsed") != std::string::npos);   // E.g. Ensemble time elapsed: 0.006000s
        EXPECT_TRUE(errors.empty());
    }
}
// Logging is more thoroughly tested in Logging. Here just make sure the methods work
TEST(TestCUDAEnsemble, setStepLog) {
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<float>("f", 0.f);
    // Add an agent so that the simulation can be ran, to check for presence of logs
    flamegpu::AgentDescription &agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    // Define the logging configuraiton.
    LoggingConfig lcfg(model);
    lcfg.logEnvironment("f");
    StepLoggingConfig slcfg(lcfg);
    slcfg.setFrequency(1);
    // Create a single run.
    auto plans = flamegpu::RunPlanVector(model, 1);
    plans[0].setSteps(1);
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    // Set the StepLog config.
    EXPECT_NO_THROW(ensemble.setStepLog(slcfg));
    // Run the ensemble, generating logs
    ensemble.simulate(plans);
    // Get the logs, checking the correct number are present.
    const auto &runLogs = ensemble.getLogs();
    EXPECT_EQ(runLogs.size(), plans.size());
    for (auto &log : runLogs) {
        auto &stepLogs = log.getStepLog();
        EXPECT_EQ(stepLogs.size(), 1 + 1);  // This is 1 + 1 due to the always present init log.
        uint32_t expectedStepCount = 0;
        for (const auto &stepLog : stepLogs) {
            ASSERT_EQ(stepLog.getStepCount(), expectedStepCount);
            expectedStepCount++;
        }
    }

    // An exception will be thrown if the step log config is for a different model.
    flamegpu::ModelDescription modelTwo("two");
    LoggingConfig lcfgTwo(modelTwo);
    StepLoggingConfig slcfgTwo(lcfgTwo);
    slcfgTwo.setFrequency(1);
    flamegpu::RunPlanVector modelTwoPlans(modelTwo, 1);
    EXPECT_THROW(ensemble.setStepLog(slcfgTwo), flamegpu::exception::InvalidArgument);
}
TEST(TestCUDAEnsemble, setExitLog) {
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<float>("f", 0.f);
    // Add an agent so that the simulation can be ran, to check for presence of logs
    flamegpu::AgentDescription &agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0u);
    // Define the logging configuraiton.
    LoggingConfig lcfg(model);
    lcfg.logEnvironment("f");
    // Create a single run.
    auto plans = flamegpu::RunPlanVector(model, 1u);
    plans[0].setSteps(1);
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    // Set the StepLog config.
    EXPECT_NO_THROW(ensemble.setExitLog(lcfg));
    // Run the ensemble, generating logs
    ensemble.simulate(plans);
    // Get the logs, checking the correct number are present.
    const auto &runLogs = ensemble.getLogs();
    EXPECT_EQ(runLogs.size(), plans.size());
    for (auto &log : runLogs) {
        const auto &exitLog = log.getExitLog();
        ASSERT_EQ(exitLog.getStepCount(), 1u);
    }

    // An exception will be thrown if the step log config is for a different model.
    flamegpu::ModelDescription modelTwo("two");
    LoggingConfig lcfgTwo(modelTwo);
    flamegpu::RunPlanVector modelTwoPlans(modelTwo, 1u);
    EXPECT_THROW(ensemble.setExitLog(lcfgTwo), flamegpu::exception::InvalidArgument);
}
TEST(TestCUDAEnsemble, getLogs) {
    // Create an ensemble with no logging enabled, but call getLogs
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    auto plans = flamegpu::RunPlanVector(model, 1);
    plans[0].setSteps(1);
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    EXPECT_NO_THROW(ensemble.getLogs());
    const auto &runLogs = ensemble.getLogs();
    EXPECT_EQ(runLogs.size(), 0u);
}
// Agent function used to check the ensemble runs.
FLAMEGPU_AGENT_FUNCTION(elapsedAgentFn, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Increment agent's counter by 1.
    FLAMEGPU->setVariable<int>("counter", FLAMEGPU->getVariable<int>("counter") + 1);
    return flamegpu::ALIVE;
}
FLAMEGPU_INIT_FUNCTION(elapsedInit) {
    // Generate a basic pop
    const uint32_t POPULATION_TO_GENERATE = FLAMEGPU->environment.getProperty<uint32_t>("POPULATION_TO_GENERATE");
    auto agent = FLAMEGPU->agent("Agent");
    for (uint32_t i = 0; i < POPULATION_TO_GENERATE; ++i) {
        agent.newAgent().setVariable<uint32_t>("counter", 0u);
    }
}
constexpr double sleepDurationSeconds = 0.5;
// File scoped atomics
FLAMEGPU_STEP_FUNCTION(elapsedStep) {
    // Sleep each thread for a duration of time.
    std::this_thread::sleep_for(std::chrono::duration<double>(sleepDurationSeconds));
}
TEST(TestCUDAEnsemble, getEnsembleElapsedTime) {
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", 1, true);
    // Agent(s)
    flamegpu::AgentDescription &agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    agent.newFunction("elapsedAgentFn", elapsedAgentFn);
    // Control flow
    model.newLayer().addAgentFunction(elapsedAgentFn);
    model.addInitFunction(elapsedInit);
    model.addStepFunction(elapsedStep);
    // Create a single run.
    auto plans = flamegpu::RunPlanVector(model, 1);
    plans[0].setSteps(1);
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    // Get the elapsed seconds before the sim has been executed
    EXPECT_NO_THROW(ensemble.getEnsembleElapsedTime());
    // Assert that it is LE zero.
    EXPECT_LE(ensemble.getEnsembleElapsedTime(), 0.);
    // Simulate the ensemble,
    EXPECT_NO_THROW(ensemble.simulate(plans));
    // Get the elapsed seconds before the sim has been executed
    double elapsedSeconds = 0.f;
    EXPECT_NO_THROW(elapsedSeconds = ensemble.getEnsembleElapsedTime());
    // Ensure the elapsed time is larger than a threshold.
    double threshold = sleepDurationSeconds * 0.8;
    EXPECT_GE(elapsedSeconds, threshold);
}
unsigned int tracked_err_ct;
unsigned int tracked_runs_ct;
FLAMEGPU_STEP_FUNCTION(throwException) {
    ++tracked_runs_ct;
    static int i = 0;
    if (++i % 2 == 0) {
        ++tracked_err_ct;
        THROW exception::UnknownInternalError("Dummy Exception");
    }
}
TEST(TestCUDAEnsemble, ErrorOff) {
    tracked_err_ct = 0;
    tracked_runs_ct = 0;
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", 1, true);
    // Agent(s)
    flamegpu::AgentDescription& agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    // Control flow
    model.addInitFunction(elapsedInit);
    model.addStepFunction(throwException);
    // Create a set of 10 Run plans
    const unsigned int ENSEMBLE_COUNT = 10;
    auto plans = flamegpu::RunPlanVector(model, ENSEMBLE_COUNT);
    plans.setSteps(1);
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    ensemble.Config().error_level = CUDAEnsemble::EnsembleConfig::Off;
    ensemble.Config().concurrent_runs = 1;  // Single device/no concurrency to ensure we get consistent data
    ensemble.Config().devices = {0};
    unsigned int reported_err_ct = 0;
    // Simulate the ensemble,
    EXPECT_NO_THROW(reported_err_ct = ensemble.simulate(plans));
    // Check correct number of fails is reported
    EXPECT_EQ(reported_err_ct, ENSEMBLE_COUNT / 2);
    EXPECT_EQ(tracked_err_ct, ENSEMBLE_COUNT / 2);
    EXPECT_EQ(tracked_runs_ct, ENSEMBLE_COUNT);
}
TEST(TestCUDAEnsemble, ErrorSlow) {
    tracked_err_ct = 0;
    tracked_runs_ct = 0;
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", 1, true);
    // Agent(s)
    flamegpu::AgentDescription& agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    // Control flow
    model.addInitFunction(elapsedInit);
    model.addStepFunction(throwException);
    // Create a set of 10 Run plans
    const unsigned int ENSEMBLE_COUNT = 10;
    auto plans = flamegpu::RunPlanVector(model, ENSEMBLE_COUNT);
    plans.setSteps(1);
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    ensemble.Config().error_level = CUDAEnsemble::EnsembleConfig::Slow;
    ensemble.Config().concurrent_runs = 1;  // Single device/no concurrency to ensure we get consistent data
    ensemble.Config().devices = { 0 };
    // Simulate the ensemble,
    EXPECT_THROW(ensemble.simulate(plans), exception::EnsembleError);
    // Check correct number of fails occurred (Unable to retrieve actual error count except from stderr with SLOW)
    EXPECT_EQ(tracked_err_ct, ENSEMBLE_COUNT / 2);
    EXPECT_EQ(tracked_runs_ct, ENSEMBLE_COUNT);
}
TEST(TestCUDAEnsemble, ErrorFast) {
    tracked_err_ct = 0;
    tracked_runs_ct = 0;
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", 1, true);
    // Agent(s)
    flamegpu::AgentDescription& agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    // Control flow
    model.addInitFunction(elapsedInit);
    model.addStepFunction(throwException);
    // Create a set of 10 Run plans
    const unsigned int ENSEMBLE_COUNT = 10;
    auto plans = flamegpu::RunPlanVector(model, ENSEMBLE_COUNT);
    plans.setSteps(1);
    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    ensemble.Config().error_level = CUDAEnsemble::EnsembleConfig::Fast;
    ensemble.Config().concurrent_runs = 1;  // Single device/no concurrency to ensure we get consistent data
    ensemble.Config().devices = { 0 };
    // Simulate the ensemble,
    EXPECT_THROW(ensemble.simulate(plans), exception::EnsembleError);
    // Check correct number of fails occurred (Fast kills ensemble as soon as first error occurs)
    EXPECT_EQ(tracked_err_ct, 1u);
    // The first run does not throw
    EXPECT_EQ(tracked_runs_ct, 2u);
}

/*
 * Test that use of a CUDA ensemble does not break existing device memory (e.g. an existing simulation) by incorrectly resetting devices.
 * This was an issue previously noticed within the python test suite, due to GC delay.
 * see https://github.com/FLAMEGPU/FLAMEGPU2/issues/939
 */
TEST(TestCUDAEnsemble, SimualteWithExistingCUDASimulation) {
    // Number of simulations to run.
    constexpr uint32_t planCount = 2u;
    constexpr uint32_t populationSize = 32u;
    // Create a model containing atleast one agent type and function.
    flamegpu::ModelDescription model("test");
    // Environmental constant for initial population
    model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", populationSize, true);
    // Agent(s)
    flamegpu::AgentDescription &agent = model.newAgent("Agent");
    agent.newVariable<uint32_t>("counter", 0);
    agent.newFunction("simulateAgentFn", simulateAgentFn);
    // Control flow
    model.newLayer().addAgentFunction(simulateAgentFn);
    model.addInitFunction(simulateInit);
    // Crete a small runplan, using a different number of steps per sim.
    uint64_t expectedResult = 0;
    flamegpu::RunPlanVector plans(model, planCount);
    for (uint32_t idx = 0; idx < plans.size(); idx++) {
        auto &plan = plans[idx];
        plan.setSteps(idx + 1);  // Can't have 0 steps without exit condition
        // Increment the expected result based on the number of steps.
        expectedResult += (idx + 1) * populationSize;
    }

    // Create and use but do not destroy a CUDASimulation, so it can be used to check for device reset issues later
    flamegpu::CUDASimulation simulation(model);
    simulation.SimulationConfig().steps = 1;
    simulation.simulate();

    // Create an ensemble
    flamegpu::CUDAEnsemble ensemble(model);
    // Make it quiet to avoid outputting during the test suite
    ensemble.Config().verbosity = Verbosity::Quiet;
    ensemble.Config().out_format = "";  // Suppress warning
    // Simulate the ensemble,
    EXPECT_NO_THROW(ensemble.simulate(plans));

    // At this point, the cudaSim should still be usable (and dtor-able()
    EXPECT_NO_THROW(simulation.step());
}

/*
 * Test that use of a CUDA ensemble does not break existing device memory (Explicit malloc) by incorrectly resetting devices.
 * This was an issue previously noticed within the python test suite, due to GC delay.
 * see https://github.com/FLAMEGPU/FLAMEGPU2/issues/939
 */
TEST(TestCUDAEnsemble, SimualteWithExistingCUDAMalloc) {
    // Allocate some arbitraty device memory.
    int * d_int = nullptr;
    gpuErrchk(cudaMalloc(&d_int, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    cudaPointerAttributes attributes = {};
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_int));
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    // Add extra layer of scope, so the ensemble get's dtor'd incase the dtor triggers a reset
    {
        // Number of simulations to run.
        constexpr uint32_t planCount = 2u;
        constexpr uint32_t populationSize = 32u;
        // Create a model containing atleast one agent type and function.
        flamegpu::ModelDescription model("test");
        // Environmental constant for initial population
        model.Environment().newProperty<uint32_t>("POPULATION_TO_GENERATE", populationSize, true);
        // Agent(s)
        flamegpu::AgentDescription &agent = model.newAgent("Agent");
        agent.newVariable<uint32_t>("counter", 0);
        agent.newFunction("simulateAgentFn", simulateAgentFn);
        // Control flow
        model.newLayer().addAgentFunction(simulateAgentFn);
        model.addInitFunction(simulateInit);
        // Crete a small runplan, using a different number of steps per sim.
        uint64_t expectedResult = 0;
        flamegpu::RunPlanVector plans(model, planCount);
        for (uint32_t idx = 0; idx < plans.size(); idx++) {
            auto &plan = plans[idx];
            plan.setSteps(idx + 1);  // Can't have 0 steps without exit condition
            // Increment the expected result based on the number of steps.
            expectedResult += (idx + 1) * populationSize;
        }
        // Create an ensemble
        flamegpu::CUDAEnsemble ensemble(model);
        // Make it quiet to avoid outputting during the test suite
        ensemble.Config().verbosity = Verbosity::Quiet;
        ensemble.Config().out_format = "";  // Suppress warning
        // Simulate the ensemble,
        EXPECT_NO_THROW(ensemble.simulate(plans));
    }

    // At this point, the manually allocated data should still be valid, i.e. cudaMemoryTypeDevice
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_int));
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    // Free explicit device memory, if it was valid (to get the correct error)
    if (attributes.type == cudaMemoryTypeDevice) {
        gpuErrchk(cudaFree(d_int));
    }
    d_int = nullptr;
}

}  // namespace test_cuda_ensemble
}  // namespace tests
}  // namespace flamegpu
