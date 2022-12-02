#include <chrono>
#include <filesystem>
#include <thread>
#include <set>

#include "flamegpu/flamegpu.h"
#include "flamegpu/util/detail/compute_capability.cuh"
#include "helpers/device_initialisation.h"
#include "flamegpu/util/Environment.h"
#include "flamegpu/io/Telemetry.h"


#include "gtest/gtest.h"

namespace flamegpu {
namespace tests {
namespace test_cuda_simulation {
    const char *MODEL_NAME = "Model";
    const char *MODEL_NAME2 = "Model2";
    const char *AGENT_NAME = "Agent";
    const char *AGENT_NAME2 = "Agent2";
    const char *FUNCTION_NAME = "Function";
    const char *LAYER_NAME = "Layer";
    const char VARIABLE_NAME[5] = "test";  // Have to define this in this form to use with compile time hash stuff
    __device__ const char dVARIABLE_NAME[5] = "test";  // Have to define this in this form to use with compile time hash stuff
    const int AGENT_COUNT = 10;
    const int MULTIPLIER = 3;
    __device__ const int dMULTIPLIER = 3;
    int externalCounter = 0;
FLAMEGPU_AGENT_FUNCTION(DeathTestFunc, MessageNone, MessageNone) {
    unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    // Agents with even value for 'x' die
    if (x % 2 == 0)
        return DEAD;
    return ALIVE;
}
FLAMEGPU_STEP_FUNCTION(IncrementCounter) {
    externalCounter++;
}
FLAMEGPU_STEP_FUNCTION(IncrementCounterSlow) {
    externalCounter++;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
FLAMEGPU_INIT_FUNCTION(InitIncrementCounterSlow) {
    externalCounter++;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}
FLAMEGPU_EXIT_FUNCTION(ExitIncrementCounterSlow) {
    externalCounter++;
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST(TestCUDASimulation, ApplyConfigDerivedContextCreation) {
    // Simply get the result from the method provided by the helper file.
    ASSERT_TRUE(getCUDASimulationContextCreationTestResult());
}
// Test that the CUDASimulation applyConfig_derived works for multiple GPU device_id values (if available)
TEST(TestCUDASimulation, AllDeviceIdValues) {
    // Get the number of devices
    int device_count = 1;
    if (cudaSuccess != cudaGetDeviceCount(&device_count) || device_count <= 0) {
        // Skip the test, if no CUDA or GPUs.
        return;
    }
    for (int i = 0; i < device_count; i++) {
        // Check if the specified device is allowed to run the tests to determine if the test should throw or not. This is system dependent so must be dynamic.
        bool shouldThrowCCException = !flamegpu::util::detail::compute_capability::checkComputeCapability(i);
        // Initialise and run a simple model on each device in the system. This test is pointless on single GPU machines.
        ModelDescription m(MODEL_NAME);
        m.newAgent(AGENT_NAME);
        // Scoping
        {
            CUDASimulation c(m);
            // Set the device ID
            c.CUDAConfig().device_id = i;
            c.SimulationConfig().steps = 1;
            //  Apply the config (and therefore set the device.)
            if (shouldThrowCCException) {
                // Should throw exception::InvalidCUDAComputeCapability if bad compute capability.
                EXPECT_THROW(c.applyConfig(), exception::InvalidCUDAComputeCapability);
                EXPECT_THROW(c.simulate(), exception::InvalidCUDAComputeCapability);
            } else {
                // Should not get any excpetions if CC is valid.
                EXPECT_NO_THROW(c.applyConfig());
                EXPECT_NO_THROW(c.simulate());
            }
        }
    }
    // Return to prior state for remaining tests.
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
}
TEST(TestSimulation, ArgParse_inputfile_long) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "--in", "test" };
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
    EXPECT_THROW(c.initialise(sizeof(argv)/sizeof(char*), argv), exception::UnsupportedFileType);  // cant detect filetype
    EXPECT_EQ(c.getSimulationConfig().input_file, argv[2]);
    // Blank init does not reset value to default
    EXPECT_THROW(c.initialise(0, nullptr), exception::UnsupportedFileType);  // cant detect filetype
    EXPECT_EQ(c.getSimulationConfig().input_file, argv[2]);
}
TEST(TestSimulationDeathTest, ArgParse_inputfile_short) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "-i", "I_DO_NOT_EXIST.xml" };
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
    EXPECT_EXIT(c.initialise(sizeof(argv) / sizeof(char*), argv), testing::ExitedWithCode(EXIT_FAILURE), ".*Loading input file '.*' failed!.*");  // File doesn't exist
}
TEST(TestSimulation, ArgParse_steps_long) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "--steps", "12" };
    EXPECT_EQ(c.getSimulationConfig().steps, 1u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
    // Blank does not reset value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
}
TEST(TestSimulation, ArgParse_steps_short) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "-s", "12" };
    EXPECT_EQ(c.getSimulationConfig().steps, 1u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
    // Blank does not reset value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
}
TEST(TestSimulation, ArgParse_randomseed_long) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "--random", "12" };
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
    // Blank does not reset value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
}
TEST(TestSimulation, ArgParse_randomseed_short) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "-r", "12" };
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
    // Blank does not reset value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
}
TEST(TestCUDASimulation, ArgParse_device_long) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "--device", "1200" };
    EXPECT_EQ(c.getCUDAConfig().device_id, 0);
    // Setting an invalid device ID is the only safe way to do this without making internal methods accessible
    // As can set to a valid device, we haven't build code for
    EXPECT_THROW(c.initialise(sizeof(argv) / sizeof(char*), argv), exception::InvalidCUDAdevice);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1200);
    // Blank init does not reset value to default
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_THROW(c.initialise(0, nullptr), exception::InvalidCUDAdevice);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1200);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}
TEST(TestCUDASimulation, ArgParse_device_short) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "-d", "1200" };
    EXPECT_EQ(c.getCUDAConfig().device_id, 0);
    // Setting an invalid device ID is the only safe way to do this without making internal methods accessible
    // As can set to a valid device, we haven't build code for
    EXPECT_THROW(c.initialise(sizeof(argv) / sizeof(char*), argv), exception::InvalidCUDAdevice);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1200);
    // Blank init does not reset value to default
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    EXPECT_THROW(c.initialise(0, nullptr), exception::InvalidCUDAdevice);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1200);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}
TEST(TestSimulation, ArgParse_unknown) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char* argv[2] = { "prog.exe", "--unknown" };
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
    testing::internal::CaptureStderr();
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    std::string errors = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(errors.find("Warning: Unknown argument") != std::string::npos);  // Should be found
}
TEST(TestSimulation, ArgParse_unknown_quiet) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char* argv[3] = { "prog.exe", "--quiet", "--unknown" };
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
    testing::internal::CaptureStderr();
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    std::string errors = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(errors.find("Warning: Unknown argument") == std::string::npos);  // Shoudl NOT be found
}
TEST(TestSimulation, ArgParse_unknown_silenced) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char* argv[3] = { "prog.exe", "--silence-unknown-args", "--unknown" };
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
    testing::internal::CaptureStderr();
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    std::string errors = testing::internal::GetCapturedStderr();
    EXPECT_TRUE(errors.find("Warning: Unknown argument") == std::string::npos);  // Shoudl NOT be found
}
TEST(TestSimulation, ArgParse_truncate) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    EXPECT_EQ(c.getSimulationConfig().truncate_log_files, false);
    const char* argv[2] = { "prog.exe", "--truncate" };
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().truncate_log_files, true);
}
TEST(TestSimulation, initialise_quiet) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    EXPECT_EQ(c.getSimulationConfig().verbosity, Verbosity::Default);
    const char* argv[2] = { "prog.exe", "--quiet" };
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().verbosity, Verbosity::Quiet);
}
TEST(TestSimulation, initialise_default) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    EXPECT_EQ(c.getSimulationConfig().verbosity, Verbosity::Default);
    const char* argv[1] = { "prog.exe" };
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().verbosity, Verbosity::Default);
}
TEST(TestSimulation, initialise_verbose) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    EXPECT_EQ(c.getSimulationConfig().verbosity, Verbosity::Default);
    const char* argv[2] = { "prog.exe", "--verbose" };
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().verbosity, Verbosity::Verbose);
}
FLAMEGPU_AGENT_FUNCTION(SetGetFn, MessageNone, MessageNone) {
    int i = FLAMEGPU->getVariable<int>(dVARIABLE_NAME);
    FLAMEGPU->setVariable<int>(dVARIABLE_NAME, i * dMULTIPLIER);
    return ALIVE;
}
TEST(TestCUDASimulation, SetGetPopulationData) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    m.newLayer(LAYER_NAME).addAgentFunction(a.newFunction(FUNCTION_NAME, SetGetFn));
    a.newVariable<int>(VARIABLE_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentVector::Agent i = pop[_i];
        i.setVariable<int>(VARIABLE_NAME, _i);
        EXPECT_THROW(i.setVariable<float>(VARIABLE_NAME, static_cast<float>(_i)), exception::InvalidVarType);
    }
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentVector::Agent i = pop[_i];
        EXPECT_EQ(i.getVariable<int>(VARIABLE_NAME), _i * MULTIPLIER);
        i.setVariable<int>(VARIABLE_NAME, _i * 2);
    }
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentVector::Agent i = pop[_i];
        EXPECT_EQ(i.getVariable<int>(VARIABLE_NAME), _i * MULTIPLIER * 2);
        EXPECT_THROW(i.getVariable<float>(VARIABLE_NAME), exception::InvalidVarType);
    }
}
TEST(TestCUDASimulation, SetGetPopulationData_InvalidAgent) {
    ModelDescription m2(MODEL_NAME2);
    AgentDescription a2 = m2.newAgent(AGENT_NAME2);
    ModelDescription m(MODEL_NAME);
    // AgentDescription a = m.newAgent(AGENT_NAME);

    AgentVector pop(a2, static_cast<unsigned int>(AGENT_COUNT));

    CUDASimulation c(m);
    EXPECT_THROW(c.setPopulationData(pop), exception::InvalidAgent);
    EXPECT_THROW(c.getPopulationData(pop), exception::InvalidAgent);
}
TEST(TestCUDASimulation, GetAgent) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    m.newLayer(LAYER_NAME).addAgentFunction(a.newFunction(FUNCTION_NAME, SetGetFn));
    a.newVariable<int>(VARIABLE_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentVector::Agent i = pop[_i];
        i.setVariable<int>(VARIABLE_NAME, _i);
    }
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    AgentInterface &agent = c.getAgent(AGENT_NAME);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        int host = 0;
        cudaMemcpy(&host, reinterpret_cast<int*>(agent.getStateVariablePtr(ModelData::DEFAULT_STATE, VARIABLE_NAME)) + _i, sizeof(int), cudaMemcpyDeviceToHost);
        EXPECT_EQ(host, _i * MULTIPLIER);
        host = _i * 2;
        cudaMemcpy(reinterpret_cast<int*>(agent.getStateVariablePtr(ModelData::DEFAULT_STATE, VARIABLE_NAME)) + _i, &host, sizeof(int), cudaMemcpyHostToDevice);
    }
    c.simulate();
    agent = c.getAgent(AGENT_NAME);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        int host = 0;
        cudaMemcpy(&host, reinterpret_cast<int*>(agent.getStateVariablePtr(ModelData::DEFAULT_STATE, VARIABLE_NAME)) + _i, sizeof(int), cudaMemcpyDeviceToHost);
        EXPECT_EQ(host, _i * 2 * MULTIPLIER);
    }
}

TEST(TestCUDASimulation, Step) {
    // Test that step does a single step
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounter);
    CUDASimulation c(m);
    c.setPopulationData(pop);
    externalCounter = 0;
    c.resetStepCounter();
    c.step();
    EXPECT_EQ(externalCounter, 1);
    EXPECT_EQ(c.getStepCounter(), 1u);
    externalCounter = 0;
    c.resetStepCounter();
    for (unsigned int i = 0; i < 5; ++i) {
        c.step();
    }
    EXPECT_EQ(externalCounter, 5);
    EXPECT_EQ(c.getStepCounter(), 5u);
}
FLAMEGPU_AGENT_FUNCTION(add_fn, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<int>("i", FLAMEGPU->getVariable<int>("i") + 1);
    FLAMEGPU->setVariable<int>("j", FLAMEGPU->getVariable<int>("j") + 1);
    return ALIVE;
}
TEST(TestCUDASimulation, SharedAgentFunction) {
    // Test that two different agents can share an agent function name/implementation
    ModelDescription model("test");

    auto agent1 = model.newAgent("a1");
    auto agent2 = model.newAgent("a2");

    agent1.newVariable<int>("i", 1);
    agent1.newVariable<int>("j", -1);
    agent2.newVariable<int>("i", -1);
    agent2.newVariable<int>("j", 1);

    auto a1f = agent1.newFunction("add", add_fn);
    auto a2f = agent2.newFunction("add", add_fn);

    auto layer = model.newLayer();
    layer.addAgentFunction(a1f);
    layer.addAgentFunction(a2f);

    CUDASimulation cudaSimulation(model);
    cudaSimulation.applyConfig();

    const unsigned int populationSize = 5;
    AgentVector pop1(agent1, populationSize);
    AgentVector pop2(agent2, populationSize);
    cudaSimulation.setPopulationData(pop1);
    cudaSimulation.setPopulationData(pop2);

    const unsigned int steps = 5;
    for (unsigned int i = 0; i < steps; ++i) {
        cudaSimulation.step();
    }

    cudaSimulation.getPopulationData(pop1);
    cudaSimulation.getPopulationData(pop2);
    for (unsigned int i = 0; i < populationSize; i++) {
        auto instance = pop1[i];
        EXPECT_EQ(instance.getVariable<int>("i"), 6);
        EXPECT_EQ(instance.getVariable<int>("j"), 4);
    }
    for (unsigned int i = 0; i < populationSize; i++) {
        auto instance = pop2[i];
        EXPECT_EQ(instance.getVariable<int>("i"), 4);
        EXPECT_EQ(instance.getVariable<int>("j"), 6);
    }
}
TEST(TestSimulation, Simulate) {
    // Simulation is abstract, so test via CUDASimulation
    // Depends on CUDASimulation::step()
    // Test that step does a single step
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounter);
    CUDASimulation c(m);
    c.setPopulationData(pop);
    externalCounter = 0;
    c.resetStepCounter();
    c.SimulationConfig().steps = 7;
    c.simulate();
    EXPECT_EQ(externalCounter, 7);
    EXPECT_EQ(c.getStepCounter(), 7u);
    externalCounter = 0;
    c.resetStepCounter();
    c.SimulationConfig().steps = 3;
    c.simulate();
    EXPECT_EQ(externalCounter, 3);
    EXPECT_EQ(c.getStepCounter(), 3u);
}

// Show that blank init resets the vals?

TEST(TestCUDASimulation, AgentDeath) {
    std::mt19937_64 generator;
    std::uniform_int_distribution<unsigned int> distribution(0, 12);
    // Test that step does a single step
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("x");
    a.newFunction("DeathFunc", DeathTestFunc).setAllowAgentDeath(true);
    m.newLayer().addAgentFunction(DeathTestFunc);
    CUDASimulation c(m);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    std::vector<unsigned int> expected_output;
    for (auto p : pop) {
        unsigned int rng = distribution(generator);
        p.setVariable<unsigned int>("x", rng);
        if (rng % 2 != 0)
            expected_output.push_back(rng);
    }
    c.setPopulationData(pop);
    c.SimulationConfig().steps = 1;
    c.simulate();
    c.getPopulationData(pop);
    EXPECT_EQ(static_cast<size_t>(pop.size()), expected_output.size());
    for (unsigned int i = 0; i < pop.size(); ++i) {
        // Check x is an expected value
        EXPECT_EQ(expected_output[i], pop[i].getVariable<unsigned int>("x"));
    }
}

// Test setting the seed with different values/types.
TEST(TestCUDASimulation, randomseedTypes) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription model(MODEL_NAME);
    AgentDescription a = model.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));

    CUDASimulation simulation(model);

    simulation.SimulationConfig().random_seed = 0;
    EXPECT_EQ(simulation.SimulationConfig().random_seed, 0);

    int32_t int32_v = INT32_MAX;
    simulation.SimulationConfig().random_seed = int32_v;
    EXPECT_EQ(simulation.SimulationConfig().random_seed, static_cast<uint64_t>(int32_v));

    uint32_t uint32_v = UINT32_MAX;
    simulation.SimulationConfig().random_seed = uint32_v;
    EXPECT_EQ(simulation.SimulationConfig().random_seed, static_cast<uint64_t>(uint32_v));

    int64_t int64_v = INT64_MAX;
    simulation.SimulationConfig().random_seed = int64_v;
    EXPECT_EQ(simulation.SimulationConfig().random_seed, static_cast<uint64_t>(int64_v));

    uint64_t uint64_v = UINT64_MAX;
    simulation.SimulationConfig().random_seed = uint64_v;
    EXPECT_EQ(simulation.SimulationConfig().random_seed, static_cast<uint64_t>(uint64_v));

    // No need to check for larger values in cudac++
}

// test the programatically accessible simulation time elapsed.
TEST(TestCUDASimulation, simulationElapsedTime) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounterSlow);

    CUDASimulation c(m);
    c.setPopulationData(pop);

    // Try getting the timer before running simulate, which should return 0
    EXPECT_EQ(c.getElapsedTimeSimulation(), 0.);
    // Call simulate to run 1 steps, which should take some length of time
    c.SimulationConfig().steps = 1;
    c.simulate();
    EXPECT_GT(c.getElapsedTimeSimulation(), 0.);

    // Then run 10 steps, which should be longer / not the same.
    double simulate1StepDuration = c.getElapsedTimeSimulation();
    c.SimulationConfig().steps = 10;
    c.simulate();
    double simulate10StepDuration = c.getElapsedTimeSimulation();
    EXPECT_GT(simulate10StepDuration, 0.);
    EXPECT_NE(simulate1StepDuration, simulate10StepDuration);
}


// test the programatically accessible simulation time elapsed.
TEST(TestCUDASimulation, initExitElapsedTime) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addInitFunction(InitIncrementCounterSlow);
    m.addStepFunction(IncrementCounterSlow);
    m.addExitFunction(ExitIncrementCounterSlow);

    CUDASimulation c(m);
    c.setPopulationData(pop);

    // Try getting the timer before running simulate, which should return 0
    EXPECT_EQ(c.getElapsedTimeSimulation(), 0.);
    EXPECT_EQ(c.getElapsedTimeInitFunctions(), 0.);
    EXPECT_EQ(c.getElapsedTimeExitFunctions(), 0.);
    // Call simulate to run 1 steps, which should take some length of time
    c.SimulationConfig().steps = 1;
    c.simulate();
    // Afterwards timers should be non 0.
    EXPECT_GT(c.getElapsedTimeSimulation(), 0.);
    EXPECT_GT(c.getElapsedTimeInitFunctions(), 0.);
    EXPECT_GT(c.getElapsedTimeExitFunctions(), 0.);
}

// test the programatically accessible per step simulation time.
TEST(TestCUDASimulation, stepElapsedTime) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounterSlow);

    CUDASimulation c(m);
    c.setPopulationData(pop);

    // Try getting the timer before running simulate, which should be empty.
    EXPECT_EQ(c.getElapsedTimeSteps().size(), 0.);
    // Or gettng an individual element which is out of boudns should have some kind of error.
    // EXPECT_GT(c.getElapsedTimeStep(1), 0.); // @todo

    // Call simulate to run 10 steps, which should take some length of time
    const unsigned int STEPS = 10u;
    c.SimulationConfig().steps = STEPS;
    c.simulate();

    std::vector<double> stepTimes = c.getElapsedTimeSteps();
    EXPECT_EQ(stepTimes.size(), STEPS);
    for (unsigned int step = 0; step < STEPS; step++) {
        EXPECT_GT(stepTimes.at(step), 0.);
        EXPECT_GT(c.getElapsedTimeStep(step), 0.);
    }
}

/* const char* rtc_empty_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MessageNone, MessageNone) {
    return ALIVE;
}
)###"; */
/**
* Test an empty agent function to ensure that the RTC library can successful build and run a minimal example
*/
/* TEST(TestCUDASimulation, RTCElapsedTime) {
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector p(a, static_cast<unsigned int>(AGENT_COUNT));
    a.newVariable<unsigned int>("x");

    // add RTC agent function
    AgentFunctionDescription rtcFunc = a.newRTCFunction("rtc_test_func", rtc_empty_agent_func);
    m.newLayer().addAgentFunction(rtcFunc);
    // Init pop
    for (unsigned int i = 0u; i < AGENT_COUNT; i++) {
        AgentVector::Agent instance = p[i];
        instance.setVariable<unsigned int>("x", static_cast<unsigned int>(i));
    }
    // Setup Model
    CUDASimulation s(m);
    s.setPopulationData(p);

    EXPECT_EQ(s.getElapsedTimeRTCInitialisation(), 0.);
    EXPECT_EQ(s.getElapsedTimeSimulation(), 0.);

} */


const char* rtc_empty_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an empty agent function to ensure that the RTC library can successful build and run a minimal example
 */
TEST(TestCUDASimulation, RTCElapsedTime) {
    ModelDescription m("m");
    AgentDescription agent = m.newAgent(AGENT_NAME);
    // add RTC agent function
    AgentFunctionDescription func = agent.newRTCFunction("rtc_test_func", rtc_empty_agent_func);
    func.setAllowAgentDeath(true);
    m.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector p(agent, AGENT_COUNT);
    CUDASimulation s(m);
    // The RTC initialisation occurs before anything try to  interact with the device, i.e. population generation so the timer should be 0 here
    EXPECT_EQ(s.getElapsedTimeRTCInitialisation(), 0.);
    s.SimulationConfig().steps = 1;
    s.setPopulationData(p);
    s.simulate();
    // Afterwards timers should be non 0.
    EXPECT_GT(s.getElapsedTimeRTCInitialisation(), 0.);
}

// test that we can have 2 instances of the same ModelDescription simultaneously
TEST(TestCUDASimulation, MultipleInstances) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounter);

    CUDASimulation c1(m);
    c1.setPopulationData(pop);
    // Set population data should trigger initialiseSingletons(), which is what leads to crash if EnvManager has matching name/id
    EXPECT_NO_THROW(CUDASimulation c2(m); c2.setPopulationData(pop););
}

FLAMEGPU_AGENT_FUNCTION(CopyID, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<id_t>("id_copy", FLAMEGPU->getID());
    return ALIVE;
}
TEST(TestCUDASimulation, AgentID_MultipleStatesUniqueIDs) {
    // Create agents via AgentVector to two agent states
    // Store agent IDs to an agent variable inside model
    // Export agents and check their IDs are unique
    // Also check that the id's copied during model match those at export

    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_copy");
    agent.newState("a");
    agent.newState("b");
    auto af_a = agent.newFunction("copy_id", CopyID);
    af_a.setInitialState("a");
    af_a.setEndState("a");
    auto af_b = agent.newFunction("copy_id2", CopyID);
    af_b.setInitialState("b");
    af_b.setEndState("b");


    auto layer = model.newLayer();
    layer.addAgentFunction(af_a);
    layer.addAgentFunction(af_b);

    AgentVector pop_in(agent, 100);

    CUDASimulation sim(model);
    sim.setPopulationData(pop_in, "a");
    sim.setPopulationData(pop_in, "b");

    sim.step();

    AgentVector pop_out_a(agent);
    AgentVector pop_out_b(agent);

    sim.getPopulationData(pop_out_a, "a");
    sim.getPopulationData(pop_out_b, "b");

    std::set<id_t> ids_original, ids_copy;

    for (auto a : pop_out_a) {
        ids_original.insert(a.getID());
        ids_copy.insert(a.getVariable<id_t>("id_copy"));
        ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));
    }
    for (auto a : pop_out_b) {
        ids_original.insert(a.getID());
        ids_copy.insert(a.getVariable<id_t>("id_copy"));
        ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));
    }
    ASSERT_EQ(ids_original.size(), pop_out_a.size() + pop_out_b.size());
    ASSERT_EQ(ids_copy.size(), pop_out_a.size() + pop_out_b.size());
}
/**
 * Test that CUDASimulation::simulate(RunPlan &), does override environment properties, steps and random seed
 */
unsigned int Simulate_RunPlan_stepcheck = 0;
FLAMEGPU_HOST_FUNCTION(Check_Simulate_RunPlan) {
    // Check env property has expected value
    EXPECT_EQ(FLAMEGPU->environment.getProperty<int>("int"), 25);
    auto t = FLAMEGPU->environment.getProperty<int, 3>("int3");
    std::array<int, 3 > t_check = { 6, 7, 8 };
    EXPECT_EQ(t, t_check);
    EXPECT_EQ(FLAMEGPU->random.getSeed(), 1233333llu);
    // Increment step counter
    ++Simulate_RunPlan_stepcheck;
}
TEST(TestCUDASimulation, Simulate_RunPlan) {
    ModelDescription m("m");
    m.newAgent(AGENT_NAME);
    m.Environment().newProperty<int>("int", 2);
    m.Environment().newProperty<int, 3>("int3", {56, 57, 58});
    m.newLayer().addHostFunction(Check_Simulate_RunPlan);

    CUDASimulation s(m);
    RunPlan plan(m);
    plan.setProperty<int>("int", 25);
    plan.setProperty<int, 3>("int3", {6, 7, 8});
    plan.setRandomSimulationSeed(1233333llu);
    plan.setSteps(5);
    // Run sim
    Simulate_RunPlan_stepcheck = 0;
    s.SimulationConfig().steps = 1;
    s.SimulationConfig().random_seed = 12;
    EXPECT_NO_THROW(s.simulate(plan));
    // Afterwards check right number of steps ran
    EXPECT_EQ(Simulate_RunPlan_stepcheck, 5u);
    // Check that config was returned to it's initial value
    EXPECT_EQ(s.SimulationConfig().steps, 1u);
    EXPECT_EQ(s.SimulationConfig().random_seed, 12);
}
TEST(TestCUDASimulation, Simulate_RunPlan_WrongEnv) {
    ModelDescription m("m");
    m.newAgent(AGENT_NAME);
    m.Environment().newProperty<int>("int", 2);
    m.Environment().newProperty<int, 3>("int3", { 56, 57, 58 });
    m.newLayer().addHostFunction(Check_Simulate_RunPlan);
    ModelDescription m2("m");
    m2.newAgent(AGENT_NAME);
    m2.Environment().newProperty<int>("int", 2);
    m2.Environment().newProperty<int, 4>("int3", { 56, 57, 58, 59 });
    m2.newLayer().addHostFunction(Check_Simulate_RunPlan);

    CUDASimulation s(m);
    RunPlan plan(m2);
    // Run sim
    EXPECT_THROW(s.simulate(plan), exception::InvalidArgument);
}
FLAMEGPU_HOST_FUNCTION(Check_setEnvironmentProperty) {
    // Check env property has expected value
    EXPECT_EQ(FLAMEGPU->environment.getProperty<int>("int"), 25);
    auto t2 = FLAMEGPU->environment.getProperty<int, 2>("int2");
    std::array<int, 2> t_check2 = { 1, -1 };
    EXPECT_EQ(t2, t_check2);
    auto t = FLAMEGPU->environment.getProperty<int, 3>("int3");
    std::array<int, 3> t_check = { 6, 7, 8 };
    EXPECT_EQ(t, t_check);

#ifdef USE_GLM
    const std::array<glm::ivec3, 3> ivec3_3_check2 =
    { glm::ivec3{ 41, 42, 43 }, glm::ivec3{44, 45, 46}, glm::ivec3{47, 48, 49} };
    EXPECT_EQ(FLAMEGPU->environment.getProperty<glm::ivec3>("ivec3"), glm::ivec3(31, 32, 33));
    EXPECT_EQ((FLAMEGPU->environment.getProperty<glm::ivec3, 3>)("ivec33"), ivec3_3_check2);
    EXPECT_EQ(FLAMEGPU->environment.getProperty<glm::ivec3>("ivec32", 0), glm::ivec3(7, 8, 9));
    EXPECT_EQ(FLAMEGPU->environment.getProperty<glm::ivec3>("ivec32", 1), glm::ivec3(4, 5, 6));
#endif
}
TEST(TestCUDASimulation, setEnvironmentProperty) {
    ModelDescription m("m");
    m.newAgent(AGENT_NAME);
    const std::array<int, 3> t_check = { 56, 57, 58 };
    m.Environment().newProperty<int>("int", 2);
    m.Environment().newProperty<int, 2>("int2", { -1, 1 });
    m.Environment().newProperty<int, 3>("int3", t_check);
#ifdef USE_GLM
    const glm::ivec3 ivec3_1_check = glm::ivec3{ 1, 2, 3 };
    const std::array<glm::ivec3, 2> ivec3_2_check = { glm::ivec3{4, 5, 6}, glm::ivec3{7, 8, 9} };
    const std::array<glm::ivec3, 3> ivec3_3_check =
        { glm::ivec3{ 11, 12, 13 }, glm::ivec3{14, 15, 16}, glm::ivec3{17, 18, 19} };
    m.Environment().newProperty<glm::ivec3>("ivec3", ivec3_1_check);
    m.Environment().newProperty<glm::ivec3, 2>("ivec32", ivec3_2_check);
    m.Environment().newProperty<glm::ivec3, 3>("ivec33", ivec3_3_check);
#endif
    m.newLayer().addHostFunction(Check_setEnvironmentProperty);

    CUDASimulation s(m);
    s.SimulationConfig().steps = 1;
    // Test the getters work
    EXPECT_EQ(s.getEnvironmentProperty<int>("int"), 2);
    EXPECT_EQ((s.getEnvironmentProperty<int, 3>)("int3"), t_check);
    EXPECT_EQ(s.getEnvironmentProperty<int>("int2", 0), -1);
    EXPECT_EQ(s.getEnvironmentProperty<int>("int2", 1), 1);
#ifdef USE_GLM
    EXPECT_EQ(s.getEnvironmentProperty<glm::ivec3>("ivec3"), ivec3_1_check);
    EXPECT_EQ((s.getEnvironmentProperty<glm::ivec3, 3>)("ivec33"), ivec3_3_check);
    EXPECT_EQ(s.getEnvironmentProperty<glm::ivec3>("ivec32", 0), ivec3_2_check[0]);
    EXPECT_EQ(s.getEnvironmentProperty<glm::ivec3>("ivec32", 1), ivec3_2_check[1]);
#endif
    // Test the setters work
    s.setEnvironmentProperty<int>("int", 25);
    s.setEnvironmentProperty<int, 3>("int3", { 6, 7, 8 });
    s.setEnvironmentProperty<int>("int2", 0, 1);
    s.setEnvironmentProperty<int>("int2", 1, -1);
#ifdef USE_GLM
    s.setEnvironmentProperty<glm::ivec3>("ivec3", glm::ivec3{ 31, 32, 33 });
    const std::array<glm::ivec3, 3> ivec3_3_check2 =
    { glm::ivec3{ 41, 42, 43 }, glm::ivec3{44, 45, 46}, glm::ivec3{47, 48, 49} };
    s.setEnvironmentProperty<glm::ivec3, 3>("ivec33", ivec3_3_check2);
    s.setEnvironmentProperty<glm::ivec3>("ivec32", 0, ivec3_2_check[1]);
    s.setEnvironmentProperty<glm::ivec3>("ivec32", 1, ivec3_2_check[0]);
#endif
    // Test the exceptions work
    EXPECT_THROW(s.CUDASimulation::setEnvironmentProperty<int>("float", 2), exception::InvalidEnvProperty);  // Bad name
    EXPECT_THROW(s.CUDASimulation::setEnvironmentProperty<int>("int3", 3), exception::OutOfBoundsException);  // Bad length
    EXPECT_THROW(s.CUDASimulation::setEnvironmentProperty<float>("int", 3), exception::InvalidEnvPropertyType);  // Bad type
    EXPECT_THROW(s.CUDASimulation::getEnvironmentProperty<int>("float"), exception::InvalidEnvProperty);  // Bad name
    EXPECT_THROW(s.CUDASimulation::getEnvironmentProperty<int>("int3"), exception::OutOfBoundsException);  // Bad length
    EXPECT_THROW(s.CUDASimulation::getEnvironmentProperty<float>("int"), exception::InvalidEnvPropertyType);  // Bad type
    const std::array<float, 3> tf3 = { 56.0f, 57.0f, 58.0f };
    const std::array<int, 5> ti5 = { 56, 57, 58, 59, 60 };
    EXPECT_THROW((s.CUDASimulation::setEnvironmentProperty<int, 3>)("float", t_check), exception::InvalidEnvProperty);  // Bad name
    EXPECT_THROW((s.CUDASimulation::setEnvironmentProperty<int, 5>)("int3", ti5), exception::OutOfBoundsException);  // Bad length
    EXPECT_THROW((s.CUDASimulation::setEnvironmentProperty<float, 3>)("int3", tf3), exception::InvalidEnvPropertyType);  // Bad type
    EXPECT_THROW((s.CUDASimulation::getEnvironmentProperty<int, 3>)("float"), exception::InvalidEnvProperty);  // Bad name
    EXPECT_THROW((s.CUDASimulation::getEnvironmentProperty<int, 5>)("int3"), exception::OutOfBoundsException);  // Bad length
    EXPECT_THROW((s.CUDASimulation::getEnvironmentProperty<float, 3>)("int3"), exception::InvalidEnvPropertyType);  // Bad type
    EXPECT_THROW((s.CUDASimulation::getEnvironmentProperty<int>)("int3", 4), exception::OutOfBoundsException);  // Out of bounds
#ifdef USE_GLM
    EXPECT_THROW((s.CUDASimulation::setEnvironmentProperty<glm::ivec3>)("ivec32", 3, {}), exception::OutOfBoundsException);  // Out of bounds
    EXPECT_THROW((s.CUDASimulation::setEnvironmentProperty<glm::ivec3>)("ivec33", 4, {}), exception::OutOfBoundsException);  // Out of bounds
    EXPECT_THROW((s.CUDASimulation::getEnvironmentProperty<glm::ivec3>)("ivec32", 3), exception::OutOfBoundsException);  // Out of bounds
    EXPECT_THROW((s.CUDASimulation::getEnvironmentProperty<glm::ivec3>)("ivec33", 4), exception::OutOfBoundsException);  // Out of bounds
#endif
    // Run sim
    s.simulate();
}

/*
 * Test that use of a CUDASimulation does not break existing device memory (e.g. an existing CUDAEnsemble) by incorrectly resetting devices.
 * This was an issue previously noticed within the python test suite, due to GC delay.
 * see https://github.com/FLAMEGPU/FLAMEGPU2/issues/939
 */
TEST(TestCUDASimulation, SimulationWithExistingCUDAMalloc) {
    // Allocate some arbitraty device memory.
    int * d_int = nullptr;
    gpuErrchk(cudaMalloc(&d_int, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    cudaPointerAttributes attributes = {};
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_int));
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    // Add extra layer of scope, so the ensemble get's dtor'd incase the dtor triggers a reset
    {
        ModelDescription m(MODEL_NAME);
        AgentDescription a = m.newAgent(AGENT_NAME);
        AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
        m.addStepFunction(IncrementCounter);
        // Instanciate a CUDASimulation of the model
        CUDASimulation c(m);
        c.setPopulationData(pop);
        c.SimulationConfig().steps = 1;
        // Assert that using the simulation does not trigger an exception
        c.simulate();
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
// Test the verbosity levels to ensure correct levels of output
TEST(TestCUDASimulation, simulationVerbosity) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentVector pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounterSlow);
    // Create a simulation (verbosity not set until simulate so no config outputs expected)
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.SimulationConfig().steps = 1;
    // Verbosity::Quiet
    {
        // Capture stderr and stdout
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        // Set verbosity level
        c.SimulationConfig().verbosity = Verbosity::Quiet;
        // Simulate
        EXPECT_NO_THROW(c.simulate());
        // Get stderr and stdout
        std::string output = testing::internal::GetCapturedStdout();
        std::string errors = testing::internal::GetCapturedStderr();
        // Expect no warnings (stderr) or outputs in Quiet mode
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }
    // DEFAULT
    {
        // Capture stderr and stdout
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        // Set verbosity level
        c.SimulationConfig().verbosity = Verbosity::Default;
        // Simulate
        c.resetStepCounter();
        EXPECT_NO_THROW(c.simulate());
        // Get stderr and stdout
        std::string output = testing::internal::GetCapturedStdout();
        std::string errors = testing::internal::GetCapturedStderr();
        // Expect no warnings (stderr) and no output
        EXPECT_TRUE(output.empty());
        EXPECT_TRUE(errors.empty());
    }
    // Verbosity::Verbose
    {
        // Capture stderr and stdout
        testing::internal::CaptureStdout();
        testing::internal::CaptureStderr();
        // Set verbosity level
        c.SimulationConfig().verbosity = Verbosity::Verbose;
        // Simulate
        c.resetStepCounter();
        EXPECT_NO_THROW(c.simulate());
        // Get stderr and stdout
        std::string output = testing::internal::GetCapturedStdout();
        std::string errors = testing::internal::GetCapturedStderr();
        // Expect no warnings (stderr) but updates on progress and timing
        EXPECT_TRUE(output.find("Init Function Processing time") != std::string::npos);
        EXPECT_TRUE(output.find("Processing Simulation Step 0") != std::string::npos);
        EXPECT_TRUE(output.find("Step 0 Processing time") != std::string::npos);
        EXPECT_TRUE(output.find("Exit Function Processing time") != std::string::npos);
        EXPECT_TRUE(output.find("Total Processing time") != std::string::npos);
        EXPECT_TRUE(errors.empty());
    }
}
TEST(TestCUDASimulation, TruncationOff_Step) {
    ModelDescription m("test");
    m.newAgent("agent");
    StepLoggingConfig slc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = false;
    e.SimulationConfig().step_log_file = "test_truncate.json";
    e.setStepLog(slc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Run sim
    EXPECT_THROW(e.simulate(), exception::FileAlreadyExists);
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOff_Exit) {
    ModelDescription m("test");
    m.newAgent("agent");
    LoggingConfig elc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = false;
    e.SimulationConfig().exit_log_file = "test_truncate.json";
    e.SimulationConfig().steps = 1;
    e.setExitLog(elc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Run sim
    EXPECT_THROW(e.simulate(), exception::FileAlreadyExists);
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOff_Common) {
    ModelDescription m("test");
    m.newAgent("agent");
    StepLoggingConfig slc(m);
    LoggingConfig elc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = false;
    e.SimulationConfig().common_log_file = "test_truncate.json";
    e.SimulationConfig().steps = 1;
    e.setStepLog(slc);
    e.setExitLog(elc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Run sim
    EXPECT_THROW(e.simulate(), exception::FileAlreadyExists);
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOff_exportLog) {
    ModelDescription m("test");
    m.newAgent("agent");
    StepLoggingConfig slc(m);
    LoggingConfig elc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = false;
    e.SimulationConfig().steps = 1;
    e.setStepLog(slc);
    e.setExitLog(elc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Run sim
    EXPECT_NO_THROW(e.simulate());
    EXPECT_THROW(e.exportLog("test_truncate.json", true, true, true, true, true), exception::FileAlreadyExists);
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOff_exportData) {
    ModelDescription m("test");
    m.newAgent("agent");
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = false;
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Run sim
    EXPECT_THROW(e.exportData("test_truncate.json", true), exception::FileAlreadyExists);
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOn_Step) {
    ModelDescription m("test");
    m.newAgent("agent");
    StepLoggingConfig slc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = true;
    e.SimulationConfig().step_log_file = "test_truncate.json";
    e.setStepLog(slc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Sanity check on the file we created
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_EQ(std::string("test"), std::string(word));
    }
    // Run sim
    EXPECT_NO_THROW(e.simulate());
    // Read back the file, check it nolonger has contents "test"
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_NE(std::string("test"), std::string(word));
    }
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOn_Exit) {
    ModelDescription m("test");
    m.newAgent("agent");
    LoggingConfig elc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = true;
    e.SimulationConfig().exit_log_file = "test_truncate.json";
    e.SimulationConfig().steps = 1;
    e.setExitLog(elc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Sanity check on the file we created
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_EQ(std::string("test"), std::string(word));
    }
    // Run sim
    EXPECT_NO_THROW(e.simulate());
    // Read back the file, check it nolonger has contents "test"
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_NE(std::string("test"), std::string(word));
    }
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOn_Common) {
    ModelDescription m("test");
    m.newAgent("agent");
    StepLoggingConfig slc(m);
    LoggingConfig elc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = true;
    e.SimulationConfig().common_log_file = "test_truncate.json";
    e.SimulationConfig().steps = 1;
    e.setStepLog(slc);
    e.setExitLog(elc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Sanity check on the file we created
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_EQ(std::string("test"), std::string(word));
    }
    // Run sim
    EXPECT_NO_THROW(e.simulate());
    // Read back the file, check it nolonger has contents "test"
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_NE(std::string("test"), std::string(word));
    }
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOn_exportLog) {
    ModelDescription m("test");
    m.newAgent("agent");
    StepLoggingConfig slc(m);
    LoggingConfig elc(m);
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = true;
    e.SimulationConfig().steps = 1;
    e.setStepLog(slc);
    e.setExitLog(elc);
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Sanity check on the file we created
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_EQ(std::string("test"), std::string(word));
    }
    // Run sim
    EXPECT_NO_THROW(e.simulate());
    EXPECT_NO_THROW(e.exportLog("test_truncate.json", true, true, true, true, true));
    // Read back the file, check it nolonger has contents "test"
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_NE(std::string("test"), std::string(word));
    }
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
TEST(TestCUDASimulation, TruncationOn_exportData) {
    ModelDescription m("test");
    m.newAgent("agent");
    CUDASimulation e(m);
    e.SimulationConfig().truncate_log_files = true;
    // Create an empty file at the output location
    {
        std::ofstream os("test_truncate.json", std::ios_base::trunc);
        os << "test";
    }
    // Sanity check on the file we created
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_EQ(std::string("test"), std::string(word));
    }
    // Run sim
    EXPECT_NO_THROW(e.exportData("test_truncate.json", true));
    // Read back the file, check it nolonger has contents "test"
    {
        std::ifstream is("test_truncate.json");
        char word[5];
        is.getline(word, 5);
        EXPECT_NE(std::string("test"), std::string(word));
    }
    // Cleanup
    std::filesystem::remove("test_truncate.json");
}
// Test setting the telemetry value via runtime function call
TEST(TestCUDASimulation, simulationTelemetryFunction) {
    // Define a simple model - doesn't need to do anything
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    CUDASimulation c(m);
    const bool telemetry = c.SimulationConfig().telemetry;
    ASSERT_FALSE(telemetry);    // Telemetry must be disabled during test suite (set in test main)
    // set telemetry at runtime which will override any global/cmake values
    c.shareUsageStatistics(true);
    EXPECT_TRUE(c.SimulationConfig().telemetry);
}

// Test telemetry global variable.
// If the global varibale is set to 'true' (or more specifically not a false value) then simulation dervived objects should set the telemetry value to true
// There are various 'false' values that are supported to specifically disable telemetry
// Not possible to test the cmake value as this may be changed by user but cmake var and environment var are respected equally
// This test is also valid for Ensembles as it belongs to the generic Simulation class
TEST(TestCUDASimulation, simulationTelemetryEnvironemt) {
    // Define a simple model - doesn't need to do anything
    ModelDescription m(MODEL_NAME);
    AgentDescription a = m.newAgent(AGENT_NAME);
    // Enable telemetry globally via "True" value
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS", "True");
    CUDASimulation c(m);
    EXPECT_TRUE(c.SimulationConfig().telemetry);  // Telemetry should have been enabled during initiaalisation due to global variable
    // check telemetry global Off values priduce expected result
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS", "Off");
    EXPECT_FALSE(flamegpu::io::Telemetry::globalTelemetryEnabled());
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS", "OFF");
    EXPECT_FALSE(flamegpu::io::Telemetry::globalTelemetryEnabled());
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS", "FALSE");
    EXPECT_FALSE(flamegpu::io::Telemetry::globalTelemetryEnabled());
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS", "0");
    EXPECT_FALSE(flamegpu::io::Telemetry::globalTelemetryEnabled());
    // Reset telemetry globally to Off
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_SHARE_USAGE_STATISTICS", "False");
    ASSERT_FALSE(flamegpu::io::Telemetry::globalTelemetryEnabled());  // Dont continue unless telemetry has been disabled globally
}
}  // namespace test_cuda_simulation
}  // namespace tests
}  // namespace flamegpu
