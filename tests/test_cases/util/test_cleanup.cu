#include <cuda_runtime.h>

#include <vector>
#include "flamegpu/util/cleanup.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"
namespace flamegpu {
namespace tests {
namespace test_cleanup {
    const char *MODEL_NAME = "Model";
    const char *AGENT_NAME = "Agent";
    const int AGENT_COUNT = 10;
    const int PLAN_COUNT = 2;

FLAMEGPU_INIT_FUNCTION(initfn) {
    auto agent = FLAMEGPU->agent(AGENT_NAME);
    for (uint32_t i = 0; i < AGENT_COUNT; ++i) {
        agent.newAgent();
    }
}
FLAMEGPU_AGENT_FUNCTION(alive, MessageNone, MessageNone) {
    return ALIVE;
}

const char* rtc_alive = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_alive, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}
)###";

// Test the getting of a device's compute capability.
TEST(TestCleanup, Explicit) {
    // Allocate some arbitraty device memory.
    int * d_int = nullptr;
    gpuErrchk(cudaMalloc(&d_int, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    cudaPointerAttributes attributes = {};
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_int));
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);

    // Call the cleanup method
    flamegpu::util::cleanup();

    // Assert that the pointer is no logner valid - i.e. the device was actually reset
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_int));
    EXPECT_NE(attributes.type, cudaMemoryTypeDevice);

    // Free explicit device memory, if it was valid (to get the correct error)
    if (attributes.type == cudaMemoryTypeDevice) {
        gpuErrchk(cudaFree(d_int));
    }
    d_int = nullptr;
}

// Test cleaning up with a CUDASim in scope, i.e. the dtor will fire after the CUDASim. E.g. this test that a CUDASimulation can be used in the main method.
TEST(TestCleanup, CUDASimulation) {
    // Define a model and a pop.
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    flamegpu::AgentFunctionDescription aliveDesc = agent.newFunction("alive", alive);
    model.addInitFunction(initfn);
    model.addExecutionRoot(aliveDesc);
    model.generateLayers();

    // Instanciate a cudaSimulation, using new so that we can explcitly trigger the dtor. First check that it works as intended
    flamegpu::CUDASimulation * simulation = new flamegpu::CUDASimulation(model);
    simulation->SimulationConfig().steps = 1;
    simulation->simulate();

    EXPECT_NO_THROW(delete simulation);
    simulation = nullptr;

    // Then try again with a cleanup in the middle. This is to effectively test what happens cleanup is used in the same scope
    simulation = new flamegpu::CUDASimulation(model);
    simulation->SimulationConfig().steps = 1;
    simulation->simulate();
    // cleanup
    EXPECT_NO_THROW(flamegpu::util::cleanup());
    // Explicitly trigger the dtor. At thsi point any device code the simulation used is invalid, so any device-touching part of simulation will fail.
    EXPECT_NO_THROW(delete simulation);
    simulation = nullptr;
}

// Test cleaning up with a CUDASim in scope, i.e. the dtor will fire after the CUDASim. E.g. this test that a CUDASimulation can be used in the main method.
TEST(TestCleanup, CUDASimulationRTC) {
    // Define a model and a pop.
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    flamegpu::AgentFunctionDescription func = agent.newRTCFunction("rtc_alive", rtc_alive);
    model.addInitFunction(initfn);
    model.addExecutionRoot(aliveDesc);
    model.generateLayers();

    // Instanciate a cudaSimulation, using new so that we can explcitly trigger the dtor. First check that it works as intended
    flamegpu::CUDASimulation * simulation = new flamegpu::CUDASimulation(model);
    simulation->SimulationConfig().steps = 1;
    simulation->simulate();

    EXPECT_NO_THROW(delete simulation);
    simulation = nullptr;

    // Then try again with a cleanup in the middle. This is to effectively test what happens cleanup is used in the same scope
    simulation = new flamegpu::CUDASimulation(model);
    simulation->SimulationConfig().steps = 1;
    simulation->simulate();
    // cleanup
    EXPECT_NO_THROW(flamegpu::util::cleanup());
    // Explicitly trigger the dtor. At thsi point any device code the simulation used is invalid, so any device-touching part of simulation will fail.
    EXPECT_NO_THROW(delete simulation);
    simulation = nullptr;
}


// Test cleaning up with a CUDAEnsemble in scope, i.e. the dtor will fire after the CUDASim. E.g. this test that a CUDASimulation can be used in the main method.
TEST(TestCleanup, CUDAEnsemble) {
    // Define a model and a pop.
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    flamegpu::AgentFunctionDescription aliveDesc = agent.newFunction("alive", alive);
    model.addInitFunction(initfn);
    model.addExecutionRoot(aliveDesc);
    model.generateLayers();

    // Create a vctor of runplans
    flamegpu::RunPlanVector plans(model, PLAN_COUNT);
    for (uint32_t idx = 0; idx < plans.size(); idx++) {
        plans[idx].setSteps(1);
    }

    // Instanciate and run a cudaEnsemble, using new so that we can explcitly trigger the dtor. First check that it works as intended
    flamegpu::CUDAEnsemble * ensemble = new flamegpu::CUDAEnsemble(model);
    ensemble->Config().verbosity = Verbosity::Quiet;
    ensemble->simulate(plans);

    EXPECT_NO_THROW(delete ensemble);
    ensemble = nullptr;

    // Then try again with a cleanup prior to destruction. This is to effectively test what happens when cleanup is used in the same scope
    ensemble = new flamegpu::CUDAEnsemble(model);
    ensemble->Config().verbosity = Verbosity::Quiet;
    ensemble->simulate(plans);
    // cleanup
    EXPECT_NO_THROW(flamegpu::util::cleanup());
    // Explicitly trigger the dtor. At thsi point any device code the ensemble used is invalid, so any device-touching part of ensemble will fail.
    EXPECT_NO_THROW(delete ensemble);
    ensemble = nullptr;
}

}  // namespace test_cleanup
}  // namespace tests
}  // namespace flamegpu
