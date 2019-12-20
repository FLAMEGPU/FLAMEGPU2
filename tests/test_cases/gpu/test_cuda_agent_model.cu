#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_cuda_agent_model {
    const char *MODEL_NAME = "Model";
TEST(TestSimulation, ArgParse_inputfile_long) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "--in", "test" };
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, "");
    EXPECT_THROW(c.initialise(sizeof(argv)/sizeof(char*), argv), TinyXMLError);  // File doesn't exist
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, argv[2]);
}
TEST(TestSimulation, ArgParse_inputfile_short) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "-i", "test" };
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, "");
    EXPECT_THROW(c.initialise(sizeof(argv) / sizeof(char*), argv), TinyXMLError);  // File doesn't exist
    EXPECT_EQ(c.getSimulationConfig().xml_input_file, argv[2]);
}
TEST(TestSimulation, ArgParse_steps_long) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "--steps", "12" };
    EXPECT_EQ(c.getSimulationConfig().steps, 0u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
}
TEST(TestSimulation, ArgParse_steps_short) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "-s", "12" };
    EXPECT_EQ(c.getSimulationConfig().steps, 0u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().steps, 12u);
}
TEST(TestSimulation, ArgParse_randomseed_long) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "--random", "12" };
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
}
TEST(TestSimulation, ArgParse_randomseed_short) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "-r", "12" };
    EXPECT_NE(c.getSimulationConfig().random_seed, 12u);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getSimulationConfig().random_seed, 12u);
}
TEST(TestCUDAAgentModel, ArgParse_device_long) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "--device", "1" };
    EXPECT_EQ(c.getCUDAConfig().device_id, 0);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1);
}
TEST(TestCUDAAgentModel, ArgParse_device_short) {
    ModelDescription m(MODEL_NAME);
    CUDAAgentModel c(m);
    const char *argv[3] = { "prog.exe", "-d", "1" };
    EXPECT_EQ(c.getCUDAConfig().device_id, 0);
    c.initialise(sizeof(argv) / sizeof(char*), argv);
    EXPECT_EQ(c.getCUDAConfig().device_id, 1);
}

// Show that blank init resets the vals?

}  // namespace test_cuda_agent_model
