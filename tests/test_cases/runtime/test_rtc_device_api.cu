#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_rtc_device_api {
const unsigned int AGENT_COUNT = 64;

const char* rtc_empty_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an empty agent function to ensure that the RTC library can successful build and run a minimal example
 */
TEST(DeviceRTCAPITest, AgentFunction_empty) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    // add RTC agent function
    AgentFunctionDescription &func = agent.newRTCFunction("rtc_test_func", rtc_empty_agent_func);
    func.setAllowAgentDeath(true);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", static_cast<float>(i));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure agent function compiles and runs
    cuda_model.step();
}

/**
 * Test an empty agent function to ensure that the RTC library can successful build and run a minimal example
 */
TEST(DeviceRTCAPITest, AgentFunction_differentName) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    // add RTC agent function
    AgentFunctionDescription &func = agent.newRTCFunction("other_name", rtc_empty_agent_func);
    func.setAllowAgentDeath(true);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", static_cast<float>(i));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure agent function compiles and runs
    cuda_model.step();
}

const char* rtc_error_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    compile_error_here;
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an simple RTC function with an error to ensure that a compile error is thrown
 * NOTE: This error can not be run as it leaves unfreed CUDA memory which will be reported in future tests. Suggest that we have a safe shutdown method of a model to handle this.
 */
/*
TEST(DeviceRTCAPITest, AgentFunction_compile_error) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_error_agent_func);
    func.setAllowAgentDeath(true);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", static_cast<float>(i));
    }
    // Setup Model
    EXPECT_THROW({
        // expected to throw an exception before running due to agent function compile error
        CUDASimulation cuda_model(model);
        cuda_model.setPopulationData(init_population);
        // Run 1 step to ensure agent function compiles and runs
        cuda_model.step();
    }, InvalidAgentFunc);
}
*/

const char* rtc_death_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    if (threadIdx.x % 2 == 0)
        return flamegpu::DEAD;
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an RTC function to ensure death is processed correctly
 */
TEST(DeviceRTCAPITest, AgentFunction_death) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_death_agent_func);
    func.setAllowAgentDeath(true);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    // Check population size is half of initial
    EXPECT_EQ(population.size(), AGENT_COUNT / 2);
}


const char* rtc_get_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    if (id % 2 == 0)
        return flamegpu::DEAD;
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an RTC function to ensure that getVaribale function works correctly. Expected result is that all even id agents are killed.
 */
TEST(DeviceRTCAPITest, AgentFunction_get) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_get_agent_func);
    func.setAllowAgentDeath(true);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    // Check population size is half of initial (which is only possible if get has returned the correct id)
    EXPECT_EQ(population.size(), AGENT_COUNT / 2);
}

const char* rtc_getset_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->setVariable<int>("id_out", id);
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an RTC function to ensure that the setVariable function works correctly. Expected result is 'id' is copied to 'id_out'
 */
TEST(DeviceRTCAPITest, AgentFunction_getset) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    agent.newVariable<int>("id_out");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_getset_agent_func);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("id_out", 0);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    for (int i = 0; i < static_cast<int>(population.size()); i++) {
        AgentVector::Agent instance = population[i];
        // Check neighbouring vars are correct
        EXPECT_EQ(instance.getVariable<int>("id"), i);
        EXPECT_EQ(instance.getVariable<int>("id_out"), i);
    }
}

const char* rtc_array_get_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    FLAMEGPU->setVariable<int>("a1", FLAMEGPU->getVariable<int, 4>("array_var", 0));
    FLAMEGPU->setVariable<int>("a2", FLAMEGPU->getVariable<int, 4>("array_var", 1));
    FLAMEGPU->setVariable<int>("a3", FLAMEGPU->getVariable<int, 4>("array_var", 2));
    FLAMEGPU->setVariable<int>("a4", FLAMEGPU->getVariable<int, 4>("array_var", 3));
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an RTC function to ensure that the getVariable function works correctly for array variables. Expected result is 'array_var' values are copied into a1, a2, a3 and a4.
 */
TEST(DeviceRTCAPITest, AgentFunction_array_get) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<int>("a1");
    agent.newVariable<int>("a2");
    agent.newVariable<int>("a3");
    agent.newVariable<int>("a4");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_array_get_agent_func);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
        instance.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    for (AgentVector::Agent instance : population) {
        int j = instance.getVariable<int>("id");
        // Check scalar variables have been set correctly by agent function (which has read them from an array)
        EXPECT_EQ(instance.getVariable<int>("a1"), 2 + j);
        EXPECT_EQ(instance.getVariable<int>("a2"), 4 + j);
        EXPECT_EQ(instance.getVariable<int>("a3"), 8 + j);
        EXPECT_EQ(instance.getVariable<int>("a4"), 16 + j);
    }
}

const char* rtc_array_set_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    // Read variables a1 to a4
    // Store as array in array_var
    int a0 = 0;
    int a1 = FLAMEGPU->getVariable<int>("a1");
    int a2 = FLAMEGPU->getVariable<int>("a2");
    int a3 = FLAMEGPU->getVariable<int>("a3");
    int a4 = FLAMEGPU->getVariable<int>("a4");
    FLAMEGPU->setVariable<int, 5>("array_var", 0, a0);
    FLAMEGPU->setVariable<int, 5>("array_var", 1, a1);
    FLAMEGPU->setVariable<int, 5>("array_var", 2, a2);
    FLAMEGPU->setVariable<int, 5>("array_var", 3, a3);
    FLAMEGPU->setVariable<int, 5>("array_var", 4, a4);
    return flamegpu::ALIVE;
}
)###";
/**
 * Test an RTC function to ensure that the setVariable function works correctly for array variables. Expected result is a1, a2, a3 and a4 are copied into 'array_var'.
 * Also includes a test to ensure that scalar variables can not use the array get and set functions of the API.
 */
TEST(DeviceRTCAPITest, AgentFunction_array_set) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    agent.newVariable<int, 5>("array_var");
    agent.newVariable<int>("a0");
    agent.newVariable<int>("a1");
    agent.newVariable<int>("a2");
    agent.newVariable<int>("a3");
    agent.newVariable<int>("a4");
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_array_set_agent_func);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("a0", i);
        instance.setVariable<int>("a1", 2 + i);
        instance.setVariable<int>("a2", 4 + i);
        instance.setVariable<int>("a3", 8 + i);
        instance.setVariable<int>("a4", 16 + i);
        instance.setVariable<int, 5>("array_var", {0, 0, 0, 0, 0 });
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    for (unsigned int i = 0; i < population.size(); ++i) {
        AgentVector::Agent instance = population[i];
        int j = instance.getVariable<int>("id");
        // Check array_var has been set from scalar variables
        std::array<int, 5> array_var = instance.getVariable<int, 5>("array_var");
        EXPECT_EQ(array_var[0], 0);
        EXPECT_EQ(array_var[1], 2 + j);
        EXPECT_EQ(array_var[2], 4 + j);
        EXPECT_EQ(array_var[3], 8 + j);
        EXPECT_EQ(array_var[4], 16 + j);
        // Value should not have been set by agent function as the value is scalar and the setter used was for an array
        EXPECT_EQ(instance.getVariable<int>("a0"), static_cast<int>(i));
    }
}

const char* rtc_msg_out_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_msg_out_func, flamegpu::MsgNone, flamegpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return flamegpu::ALIVE;
}
)###";

const char* rtc_msg_in_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_msg_in_func, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return flamegpu::ALIVE;
}
)###";
/**
 * Tests RTC behaviour of messages (using brute force messaging) to ensure correct mapping of message variables to RTC device pointers
 * As messages are derived from a common CUDAMessage type there is no need to perform the same test with every message type.
 */
TEST(DeviceRTCAPITest, AgentFunction_msg_bruteforce) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("rtc_msg_out_func", rtc_msg_out_func);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("rtc_msg_in_func", rtc_msg_in_func);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}

const char* rtc_rand_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_rand_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    FLAMEGPU->setVariable<float>("a", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("b", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("c", FLAMEGPU->random.uniform<float>());
    return flamegpu::ALIVE;
}
)###";
/**
 * Test agent random functions to ensure that random values are returned by RTC implementation. Implemented from AgentRandomTest.AgentRandomCheck Test Model 1. No need to check seed as this is done in the orginal test.
 */
TEST(DeviceRTCAPITest, AgentFunction_random) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<float>("a");
    agent.newVariable<float>("b");
    agent.newVariable<float>("c");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_rand_func", rtc_rand_func);
    model.newLayer().addAgentFunction(func);
    // Init pop with 0 values for variables
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("a", 0);
        instance.setVariable<float>("b", 0);
        instance.setVariable<float>("c", 0);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    for (unsigned int i = 0; i < population.size(); i++) {
        AgentVector::Agent instance = population[i];
        // Check for random values
        float a = instance.getVariable<float>("a");
        float b = instance.getVariable<float>("b");
        float c = instance.getVariable<float>("c");
        if (i != 0) {
            // Check that the value has changed
            EXPECT_NE(a, 0);
            EXPECT_NE(b, 0);
            EXPECT_NE(c, 0);
            // Check that no error value is returned
            EXPECT_NE(a, -1);
            EXPECT_NE(b, -1);
            EXPECT_NE(c, -1);
        }
        // Multiple calls get multiple random numbers
        EXPECT_TRUE(a != b);
        EXPECT_TRUE(b != c);
        EXPECT_TRUE(a != c);
    }
}


const char* rtc_env_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_env_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    // get environment variable and set it as an agent variable (environment variable does not change)
    int e1_out = FLAMEGPU->environment.getProperty<int>("e1");
    int e1_exists = FLAMEGPU->environment.containsProperty("e1");
    FLAMEGPU->setVariable<int>("e1_out", e1_out);
    FLAMEGPU->setVariable<int>("e1_exists", e1_exists);
    // get stepped environment variable and add it to agent variable (environment variable is increased in step function)
    int e2 = FLAMEGPU->environment.getProperty<int>("e2");
    int e2_out = FLAMEGPU->getVariable<int>("e2_out") + e2;
    FLAMEGPU->setVariable<int>("e2_out", e2_out);
    // get array variables and set in agent variable arrays
    for (int i=0; i<32; i++) {
        int e_temp1 = FLAMEGPU->environment.getProperty<int>("e_array_1", i);
        int e_temp2 = FLAMEGPU->environment.getProperty<int>("e_array_2", i);
        // set values in agent arrary
        FLAMEGPU->setVariable<int, 32>("e_array_out_1", i, e_temp1);
        FLAMEGPU->setVariable<int, 32>("e_array_out_2", i, e_temp2);
    }
    return flamegpu::ALIVE;
}
)###";

FLAMEGPU_STEP_FUNCTION(etc_env_step) {
    // Test Set + Get for scalar (set returns previous value)
    FLAMEGPU->environment.setProperty<int>("e2", 400);
    // Test Set + Get for set by array
    std::array<int, 32> e_array_1;
    for (int i = 0; i < 32; i++) {
        e_array_1[i] = i;  // fill array values
    }
    FLAMEGPU->environment.setProperty<int, 32>("e_array_1", e_array_1);
    // Test Set + Get for set by array index
    std::array<int, 32> e_array_2;
    for (int i = 0; i < 32; i++) {
        e_array_2[i] = i;  // fill array values
        FLAMEGPU->environment.setProperty<int>("e_array_2", i, e_array_2[i]);
    }
}
/**
 * Test agent environment functions.
 */
TEST(DeviceRTCAPITest, AgentFunction_env) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("e1_out");
    agent.newVariable<int>("e1_exists");
    agent.newVariable<int>("e2_out");
    agent.newVariable<int, 32>("e_array_out_1");
    agent.newVariable<int, 32>("e_array_out_2");
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_env_func", rtc_env_func);
    model.newLayer().addAgentFunction(func);
    // empty array
    std::array<int, 32> zero_array;
    zero_array.fill(0);
    // create some environment variables
    EnvironmentDescription& env = model.Environment();
    env.newProperty<int>("e1", 100);
    env.newProperty<int>("e2", 200);
    env.newProperty<int, 32>("e_array_1", zero_array);
    env.newProperty<int, 32>("e_array_2", zero_array);
    // Init pop with 0 values for variables
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("e1_out", 0);
        instance.setVariable<int>("e1_exists", false);
        instance.setVariable<int>("e2_out", 0);
        // set the agent array variables to 0

        instance.setVariable<int, 32>("e_array_out_1", zero_array);
        instance.setVariable<int, 32>("e_array_out_2", zero_array);
    }
    // add step function to increase environment variable
    model.addStepFunction(etc_env_step);
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Set the number of steps to 2
    cuda_model.SimulationConfig().steps = 2;
    cuda_model.simulate();
    // Recover data from device
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    for (AgentVector::Agent instance : population) {
        // Check constant environment values have been updated
        int e1_out = instance.getVariable<int>("e1_out");
        int e1_exists = instance.getVariable<int>("e1_exists");
        // Check that the value is correct and that the variable is reported as existing
        EXPECT_EQ(e1_out, 100);
        EXPECT_TRUE(e1_exists);
        // check that the stepped value has been reported correctly (e.g. should be 200 (from step 1) + 400 (from step 2)
        int e2_out = instance.getVariable<int>("e2_out");
        EXPECT_EQ(e2_out, 600);
        // check the agent array outputs match the environment variables set in the step function
        std::array<int, 32> e_array_out_1 = instance.getVariable<int, 32>("e_array_out_1");
        std::array<int, 32> e_array_out_2 = instance.getVariable<int, 32>("e_array_out_2");
        for (int j = 0; j < 32; j++) {
            EXPECT_EQ(e_array_out_1[j], j);
            EXPECT_EQ(e_array_out_2[j], j);
        }
    }
}

const char* rtc_agent_output_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_agent_output_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
    if (threadIdx.x % 2 == 0) {
        FLAMEGPU->agent_out.setVariable<unsigned int>("x", id + 12);
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
    }
    return flamegpu::ALIVE;
}
)###";
/**
 * Test rtc agent output by optionally setting an agent output and checking for the correct population size and agent values
 */
TEST(DeviceRTCAPITest, AgentFunction_agent_output) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<unsigned int>("x");
    agent.newVariable<unsigned int>("id");
    // add RTC agent function with an agent output
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_agent_output_func", rtc_agent_output_func);
    func.setAllowAgentDeath(true);
    func.setAgentOutput(agent);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<unsigned int>("x", i + 1);
        instance.setVariable<unsigned int>("id", i);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step
    cuda_model.step();
    // Test output
    AgentVector population(agent);
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(), (unsigned int)(AGENT_COUNT * 1.5));
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (AgentVector::Agent ai : population) {
        const unsigned int id = ai.getVariable<unsigned int>("id");
        const unsigned int val = ai.getVariable<unsigned int>("x") - id;
        if (val == 1)
            is_1++;
        else if (val == 12)
            is_12++;
    }
    EXPECT_EQ(is_1, AGENT_COUNT);
    EXPECT_EQ(is_12, AGENT_COUNT / 2);
}

const char* rtc_func_cond_non_rtc_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->setVariable<int>("id_out", id);
    return flamegpu::ALIVE;
}
)###";
FLAMEGPU_AGENT_FUNCTION_CONDITION(odd_only) {
    return FLAMEGPU->getVariable<int>("id") % 2;
}
/**
 * Test an RTC function to an agent function condition (where the condition is not compiled using RTC)
 */
TEST(DeviceRTCAPITest, AgentFunction_cond_non_rtc) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    agent.newVariable<int>("id_out");
    // add a new state for condition
    agent.newState("default");
    agent.newState("odd_state");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_func_cond_non_rtc_func);
    // set output state (input state is default)
    func.setInitialState("default");
    func.setEndState("odd_state");
    func.setFunctionCondition(odd_only);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("id_out", 0);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population_default(agent);
    AgentVector population_odd_state(agent);
    cuda_model.getPopulationData(population_default, "default");
    cuda_model.getPopulationData(population_odd_state, "odd_state");
    // Check population size is half of initial
    EXPECT_EQ(population_default.size(), AGENT_COUNT / 2);
    EXPECT_EQ(population_odd_state.size(), AGENT_COUNT / 2);
    // check default state
    for (AgentVector::Agent ai : population_default) {
        int id_out = ai.getVariable<int>("id_out");
        // even id agent in default state should not have updated their id_out value
        EXPECT_EQ(id_out, 0);
    }
    // check odd state
    for (AgentVector::Agent ai : population_odd_state) {
        int id = ai.getVariable<int>("id");
        int id_out = ai.getVariable<int>("id_out");
        // odd id agent should have updated their id_out value
        EXPECT_EQ(id_out, id);
    }
}

const char* rtc_func_cond_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MsgNone, flamegpu::MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->setVariable<int>("id_out", id);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_func_cond = R"###(
FLAMEGPU_AGENT_FUNCTION_CONDITION(odd_only) {
    return FLAMEGPU->getVariable<int>("id") % 2;
}
)###";
/**
 * Test an RTC function to an agent function condition (where the condition IS compiled using RTC)
 */
TEST(DeviceRTCAPITest, AgentFunction_cond_rtc) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    agent.newVariable<int>("id_out");
    // add a new state for condition
    agent.newState("default");
    agent.newState("odd_state");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_func_cond_func);
    // set output state (input state is default)
    func.setInitialState("default");
    func.setEndState("odd_state");
    func.setRTCFunctionCondition(rtc_func_cond);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("id_out", 0);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentVector population_default(agent);
    AgentVector population_odd_state(agent);
    cuda_model.getPopulationData(population_default, "default");
    cuda_model.getPopulationData(population_odd_state, "odd_state");
    // Check population size is half of initial
    EXPECT_EQ(population_default.size(), AGENT_COUNT / 2);
    EXPECT_EQ(population_odd_state.size(), AGENT_COUNT / 2);
    // check default state
    for (AgentVector::Agent ai : population_default) {
        int id_out = ai.getVariable<int>("id_out");
        // even id agent in default state should not have updated their id_out value
        EXPECT_EQ(id_out, 0);
    }
    // check odd state
    for (AgentVector::Agent ai : population_odd_state) {
        int id = ai.getVariable<int>("id");
        int id_out = ai.getVariable<int>("id_out");
        // odd id agent should have updated their id_out value
        EXPECT_EQ(id_out, id);
    }
}


/**
 * Test device_api::getStepCounter() in RTC code.
 */
const char* rtc_testGetStepCounter = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_testGetStepCounter, flamegpu::MsgNone, flamegpu::MsgNone) {
    FLAMEGPU->setVariable<unsigned int>("step", FLAMEGPU->getStepCounter());
    return flamegpu::ALIVE;
}
)###";
TEST(DeviceRTCAPITest, getStepCounter) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("step");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_testGetStepCounter", rtc_testGetStepCounter);
    model.newLayer().addAgentFunction(func);
    // Init pop
    const unsigned int agentCount = 1;
    AgentVector init_population(agent, agentCount);
    for (int i = 0; i< static_cast<int>(agentCount); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<unsigned int>("step", 0);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(init_population);

    const unsigned int STEPS = 2;
    for (unsigned int step = 0; step < STEPS; step++) {
        cuda_model.step();
        // Recover data from device
        AgentVector population(agent);
        cuda_model.getPopulationData(population);
        // Check data is correct.
        for (AgentVector::Agent instance : population) {
            // Check neighbouring vars are correct
            EXPECT_EQ(instance.getVariable<unsigned int>("step"), step);
        }
    }
}

}  // namespace test_rtc_device_api
}  // namespace flamegpu
