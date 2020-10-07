#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"
#include "flamegpu/util/compute_capability.cuh"
#include "helpers/device_initialisation.h"

#include "gtest/gtest.h"

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
FLAMEGPU_AGENT_FUNCTION(DeathTestFunc, MsgNone, MsgNone) {
    unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    // Agents with even value for 'x' die
    if (x % 2 == 0)
        return DEAD;
    return ALIVE;
}
FLAMEGPU_STEP_FUNCTION(IncrementCounter) {
    externalCounter++;
}

TEST(TestCUDASimulation, ApplyConfigDerivedContextCreation) {
    // Simply get the result from the method provided by the helper file.
    ASSERT_TRUE(getCUDASimulationContextCreationTestResult());
    // Reset the device, just to be sure.
    ASSERT_EQ(cudaSuccess, cudaDeviceReset());
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
        bool shouldThrowCCException = !util::compute_capability::checkComputeCapability(i);
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
                // Should throw InvalidCUDAComputeCapability if bad compute capability.
                EXPECT_THROW(c.applyConfig(), InvalidCUDAComputeCapability);
                EXPECT_THROW(c.simulate(), InvalidCUDAComputeCapability);
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
    EXPECT_THROW(c.initialise(sizeof(argv)/sizeof(char*), argv), UnsupportedFileType);  // cant detect filetype
    EXPECT_EQ(c.getSimulationConfig().input_file, argv[2]);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
}
TEST(TestSimulation, ArgParse_inputfile_short) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "-i", "I_DO_NOT_EXIST.xml" };
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
    EXPECT_THROW(c.initialise(sizeof(argv) / sizeof(char*), argv), InvalidInputFile);  // File doesn't exist
    EXPECT_EQ(c.getSimulationConfig().input_file, argv[2]);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().input_file, "");
}
TEST(TestSimulation, ArgParse_steps_long) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "--steps", "12" };
    EXPECT_EQ(c.getSimulationConfig().steps, 0u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().steps, 0u);
}
TEST(TestSimulation, ArgParse_steps_short) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "-s", "12" };
    EXPECT_EQ(c.getSimulationConfig().steps, 0u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().steps, 0u);
}
TEST(TestSimulation, ArgParse_randomseed_long) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "--random", "12" };
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
}
TEST(TestSimulation, ArgParse_randomseed_short) {
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "-r", "12" };
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
}
TEST(TestCUDASimulation, ArgParse_device_long) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ModelDescription m(MODEL_NAME);
    CUDASimulation c(m);
    const char *argv[3] = { "prog.exe", "--device", "1200" };
    EXPECT_EQ(c.getCUDAConfig().device_id, 0);
    // Setting an invalid device ID is the only safe way to do this without making internal methods accessible
    // As can set to a valid device, we haven't build code for
    EXPECT_THROW(c.initialise(sizeof(argv) / sizeof(char*), argv), InvalidCUDAdevice);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1200);
    // Blank init resets value to default
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getCUDAConfig().device_id, 0);
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
    EXPECT_THROW(c.initialise(sizeof(argv) / sizeof(char*), argv), InvalidCUDAdevice);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1200);
    // Blank init resets value to default
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getCUDAConfig().device_id, 0);
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
}
FLAMEGPU_AGENT_FUNCTION(SetGetFn, MsgNone, MsgNone) {
    int i = FLAMEGPU->getVariable<int>(dVARIABLE_NAME);
    FLAMEGPU->setVariable<int>(dVARIABLE_NAME, i * dMULTIPLIER);
    return ALIVE;
}
TEST(TestCUDASimulation, SetGetPopulationData) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    m.newLayer(LAYER_NAME).addAgentFunction(a.newFunction(FUNCTION_NAME, SetGetFn));
    a.newVariable<int>(VARIABLE_NAME);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentInstance i = pop.getNextInstance();
        i.setVariable<int>(VARIABLE_NAME, _i);
        EXPECT_THROW(i.setVariable<float>(VARIABLE_NAME, static_cast<float>(_i)), InvalidVarType);
    }
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentInstance i = pop.getInstanceAt(_i);
        EXPECT_EQ(i.getVariable<int>(VARIABLE_NAME), _i * MULTIPLIER);
        i.setVariable<int>(VARIABLE_NAME, _i * 2);
    }
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentInstance i = pop.getInstanceAt(_i);
        EXPECT_EQ(i.getVariable<int>(VARIABLE_NAME), _i * MULTIPLIER * 2);
        EXPECT_THROW(i.getVariable<float>(VARIABLE_NAME), InvalidVarType);
    }
}
TEST(TestCUDASimulation, SetGetPopulationData_InvalidCudaAgent) {
    ModelDescription m2(MODEL_NAME2);
    AgentDescription &a2 = m2.newAgent(AGENT_NAME2);
    ModelDescription m(MODEL_NAME);
    // AgentDescription &a = m.newAgent(AGENT_NAME);

    AgentPopulation pop(a2, static_cast<unsigned int>(AGENT_COUNT));

    CUDASimulation c(m);
    EXPECT_THROW(c.setPopulationData(pop), InvalidCudaAgent);
    EXPECT_THROW(c.getPopulationData(pop), InvalidCudaAgent);
}
TEST(TestCUDASimulation, GetAgent) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    m.newLayer(LAYER_NAME).addAgentFunction(a.newFunction(FUNCTION_NAME, SetGetFn));
    a.newVariable<int>(VARIABLE_NAME);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentInstance i = pop.getNextInstance();
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
    AgentDescription &a = m.newAgent(AGENT_NAME);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
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
FLAMEGPU_AGENT_FUNCTION(add_fn, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<int>("i", FLAMEGPU->getVariable<int>("i") + 1);
    FLAMEGPU->setVariable<int>("j", FLAMEGPU->getVariable<int>("j") + 1);
    return ALIVE;
}
TEST(TestCUDASimulation, SharedAgentFunction) {
    // Test that two different agents can share an agent function name/implementation
    ModelDescription model("test");

    auto &agent1 = model.newAgent("a1");
    auto &agent2 = model.newAgent("a2");

    agent1.newVariable<int>("i", 1);
    agent1.newVariable<int>("j", -1);
    agent2.newVariable<int>("i", -1);
    agent2.newVariable<int>("j", 1);

    auto &a1f = agent1.newFunction("add", add_fn);
    auto &a2f = agent2.newFunction("add", add_fn);

    auto &layer = model.newLayer();
    layer.addAgentFunction(a1f);
    layer.addAgentFunction(a2f);

    CUDASimulation cudaSimulation(model);
    cudaSimulation.applyConfig();

    const unsigned int populationSize = 5;
    AgentPopulation pop1(agent1, populationSize);
    AgentPopulation pop2(agent2, populationSize);
    for (unsigned int i = 0; i < populationSize; i++) {
        pop1.getNextInstance();
        pop2.getNextInstance();
    }
    cudaSimulation.setPopulationData(pop1);
    cudaSimulation.setPopulationData(pop2);

    const unsigned int steps = 5;
    for (unsigned int i = 0; i < steps; ++i) {
        cudaSimulation.step();
    }

    cudaSimulation.getPopulationData(pop1);
    cudaSimulation.getPopulationData(pop2);
    for (unsigned int i = 0; i < populationSize; i++) {
        auto instance = pop1.getInstanceAt(i);
        EXPECT_EQ(instance.getVariable<int>("i"), 6);
        EXPECT_EQ(instance.getVariable<int>("j"), 4);
    }
    for (unsigned int i = 0; i < populationSize; i++) {
        auto instance = pop2.getInstanceAt(i);
        EXPECT_EQ(instance.getVariable<int>("i"), 4);
        EXPECT_EQ(instance.getVariable<int>("j"), 6);
    }
}
TEST(TestSimulation, Simulate) {
    // Simulation is abstract, so test via CUDASimulation
    // Depends on CUDASimulation::step()
    // Test that step does a single step
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
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
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, 12);
    // Test that step does a single step
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("x");
    a.newFunction("DeathFunc", DeathTestFunc).setAllowAgentDeath(true);
    m.newLayer().addAgentFunction(DeathTestFunc);
    CUDASimulation c(m);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
    std::vector<unsigned int> expected_output;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        auto p = pop.getNextInstance();
        unsigned int rng = distribution(generator);
        p.setVariable<unsigned int>("x", rng);
        if (rng % 2 != 0)
            expected_output.push_back(rng);
    }
    c.setPopulationData(pop);
    c.SimulationConfig().steps = 1;
    c.simulate();
    c.getPopulationData(pop);
    EXPECT_EQ(static_cast<size_t>(pop.getCurrentListSize()), expected_output.size());
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        // Check x is an expected value
        EXPECT_EQ(expected_output[i], ai.getVariable<unsigned int>("x"));
    }
}

// test the programatically accessible simulation time elapsed.
TEST(TestCUDASimulation, getSimulationElapsedTime) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounter);

    CUDASimulation c(m);
    c.setPopulationData(pop);

    // Try getting the timer before running simulate, which should return 0
    EXPECT_EQ(c.getSimulationElapsedTime(), 0.0f);
    // Call simulate to run 1 steps, which should take some length of time
    c.SimulationConfig().steps = 1;
    c.simulate();
    EXPECT_GT(c.getSimulationElapsedTime(), 0.0f);

    // Then run 10 steps, which should be longer / not the same.
    float simulate1StepDuration = c.getSimulationElapsedTime();
    c.SimulationConfig().steps = 10;
    c.simulate();
    float simulate10StepDuration = c.getSimulationElapsedTime();
    EXPECT_GT(simulate10StepDuration, 0.0f);
    EXPECT_NE(simulate1StepDuration, simulate10StepDuration);
}
// test that we can have 2 instances of the same ModelDescription simultaneously
TEST(TestCUDASimulation, MultipleInstances) {
    // Define a simple model - doesn't need to do anything other than take some time.
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounter);

    CUDASimulation c1(m);
    c1.setPopulationData(pop);
    // Set population data should trigger initialiseSingletons(), which is what leads to crash if EnvManager has matching name/id
    EXPECT_NO_THROW(CUDASimulation c2(m); c2.setPopulationData(pop););
}

}  // namespace test_cuda_simulation
