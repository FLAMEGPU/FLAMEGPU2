#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_cuda_agent_model {
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
TEST(TestSimulation, ArgParse_inputfile_long) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "--in", "test" };
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, "");
    EXPECT_THROW(c.initialise(sizeof(argv)/sizeof(char*), argv), UnsupportedFileType);  // cant detect filetype
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, argv[2]);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, "");
}
TEST(TestSimulation, ArgParse_inputfile_short) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "-i", "test.xml" };
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, "");
    EXPECT_THROW(c.initialise(sizeof(argv) / sizeof(char*), argv), InvalidInputFile);  // File doesn't exist
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, argv[2]);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, "");
}
TEST(TestSimulation, ArgParse_steps_long) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
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
    CUDAAgentModel c(m);
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
    CUDAAgentModel c(m);
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
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "-r", "12" };
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
    // Blank init resets value to default
    c.initialise(0, nullptr);
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
}
TEST(TestCUDAAgentModel, ArgParse_device_long) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
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
TEST(TestCUDAAgentModel, ArgParse_device_short) {
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
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
FLAMEGPU_AGENT_FUNCTION(SetGetFn) {
    int i = FLAMEGPU->getVariable<int>(dVARIABLE_NAME);
    FLAMEGPU->setVariable<int>(dVARIABLE_NAME, i * dMULTIPLIER);
    return ALIVE;
}
TEST(TestCUDAAgentModel, SetGetPopulationData) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    m.newLayer(LAYER_NAME).addAgentFunction(a.newFunction(FUNCTION_NAME, SetGetFn));
    a.newVariable<int>(VARIABLE_NAME);
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentInstance i = pop.getNextInstance();
        i.setVariable<int>(VARIABLE_NAME, _i);
        EXPECT_THROW(i.setVariable<float>(VARIABLE_NAME, _i), InvalidVarType);
    }
    CUDAAgentModel c(m);
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
TEST(TestCUDAAgentModel, SetGetPopulationData_InvalidCudaAgent) {
    ModelDescription m2(MODEL_NAME2);
    AgentDescription &a2 = m2.newAgent(AGENT_NAME2);
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);

    AgentPopulation pop(a2, (unsigned int)AGENT_COUNT);

    CUDAAgentModel c(m);
    EXPECT_THROW(c.setPopulationData(pop), InvalidCudaAgent);
    EXPECT_THROW(c.getPopulationData(pop), InvalidCudaAgent);
}
TEST(TestCUDAAgentModel, GetAgent) {
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    m.newLayer(LAYER_NAME).addAgentFunction(a.newFunction(FUNCTION_NAME, SetGetFn));
    a.newVariable<int>(VARIABLE_NAME);
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    for (int _i = 0; _i < AGENT_COUNT; ++_i) {
        AgentInstance i = pop.getNextInstance();
        i.setVariable<int>(VARIABLE_NAME, _i);
    }
    CUDAAgentModel c(m);
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

// Show that blank init resets the vals?

}  // namespace test_cuda_agent_model
