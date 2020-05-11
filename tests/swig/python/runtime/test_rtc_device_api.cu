#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_rtc_device_api {
const unsigned int AGENT_COUNT = 64;

const char* rtc_empty_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", static_cast<float>(i));
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", static_cast<float>(i));
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure agent function compiles and runs
    cuda_model.step();
}

const char* rtc_error_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    compile_error_here;
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", static_cast<float>(i));
    }
    // Setup Model
    EXPECT_THROW({
        // expected to throw an exception before running due to agent function compile error
        CUDAAgentModel cuda_model(model);
        cuda_model.setPopulationData(init_population);
        // Run 1 step to ensure agent function compiles and runs
        cuda_model.step();
    }, InvalidAgentFunc);
}
*/

const char* rtc_death_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    if (threadIdx.x % 2 == 0)
        return DEAD;
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    // Check population size is half of initial
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT / 2);
}


const char* rtc_get_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    if (id % 2 == 0)
        return DEAD;
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    // Check population size is half of initial (which is only possible if get has returned the correct id)
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT / 2);
}

const char* rtc_getset_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->setVariable<int>("id_out", id);
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("id_out", 0);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    for (int i = 0; i < static_cast<int>(population.getCurrentListSize()); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        // Check neighbouring vars are correct
        EXPECT_EQ(instance.getVariable<int>("id"), i);
        EXPECT_EQ(instance.getVariable<int>("id_out"), i);
    }
}

const char* rtc_array_get_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    FLAMEGPU->setVariable<int>("a1", FLAMEGPU->getVariable<int, 4>("array_var", 0));
    FLAMEGPU->setVariable<int>("a2", FLAMEGPU->getVariable<int, 4>("array_var", 1));
    FLAMEGPU->setVariable<int>("a3", FLAMEGPU->getVariable<int, 4>("array_var", 2));
    FLAMEGPU->setVariable<int>("a4", FLAMEGPU->getVariable<int, 4>("array_var", 3));
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<int>("id", i);
        instance.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        int j = instance.getVariable<int>("id");
        // Check scalar variables have been set correctly by agent function (which has read them from an array)
        EXPECT_EQ(instance.getVariable<int>("a1"), 2 + j);
        EXPECT_EQ(instance.getVariable<int>("a2"), 4 + j);
        EXPECT_EQ(instance.getVariable<int>("a3"), 8 + j);
        EXPECT_EQ(instance.getVariable<int>("a4"), 16 + j);
    }
}

const char* rtc_array_set_agent_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    // Read variables a1 to a4
    // Store as array in array_var
    int a0 = FLAMEGPU->getVariable<int, 2>("a1", 1);   // should not be able to read scalar value as array (expecting 0)
    int a1 = FLAMEGPU->getVariable<int>("a1");
    int a2 = FLAMEGPU->getVariable<int>("a2");
    int a3 = FLAMEGPU->getVariable<int>("a3");
    int a4 = FLAMEGPU->getVariable<int>("a4");
    FLAMEGPU->setVariable<int, 5>("array_var", 0, a0);
    FLAMEGPU->setVariable<int, 5>("array_var", 1, a1);
    FLAMEGPU->setVariable<int, 5>("array_var", 2, a2);
    FLAMEGPU->setVariable<int, 5>("array_var", 3, a3);
    FLAMEGPU->setVariable<int, 5>("array_var", 4, a4);
    FLAMEGPU->setVariable<int, 2>("a0", 0, 10);           // should not be possible (no value should be written)
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("a0", i);
        instance.setVariable<int>("a1", 2 + i);
        instance.setVariable<int>("a2", 4 + i);
        instance.setVariable<int>("a3", 8 + i);
        instance.setVariable<int>("a4", 16 + i);
        instance.setVariable<int, 5>("array_var", {0, 0, 0, 0, 0 });
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    for (int i = 0; i < static_cast<int>(population.getCurrentListSize()); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        int j = instance.getVariable<int>("id");
        // Check array_var has been set from scalar variables
        std::array<int, 5> array_var = instance.getVariable<int, 5>("array_var");
        EXPECT_EQ(array_var[0], 0);
        EXPECT_EQ(array_var[1], 2 + j);
        EXPECT_EQ(array_var[2], 4 + j);
        EXPECT_EQ(array_var[3], 8 + j);
        EXPECT_EQ(array_var[4], 16 + j);
        // Value should not have been set by agent function as the value is scalar and the setter used was for an array
        EXPECT_EQ(instance.getVariable<int>("a0"), i);
    }
}

const char* rtc_msg_out_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_msg_out_func, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
)###";

const char* rtc_msg_in_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_msg_in_func, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
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
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDAAgentModel c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}

const char* rtc_rand_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_rand_func, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<float>("a", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("b", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("c", FLAMEGPU->random.uniform<float>());
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("a", 0);
        instance.setVariable<float>("b", 0);
        instance.setVariable<float>("c", 0);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
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
FLAMEGPU_AGENT_FUNCTION(rtc_env_func, MsgNone, MsgNone) {
    // get environment variable and set it as an agent variable (environment variable does not change)
    int e1_out = FLAMEGPU->environment.get<int>("e1");
    int e1_exists = FLAMEGPU->environment.contains("e1");
    FLAMEGPU->setVariable<int>("e1_out", e1_out);
    FLAMEGPU->setVariable<int>("e1_exists", e1_exists);
    // get stepped environment variable and add it to agent variable (environment variable is increased in step function)
    int e2 = FLAMEGPU->environment.get<int>("e2");
    int e2_out = FLAMEGPU->getVariable<int>("e2_out") + e2;
    FLAMEGPU->setVariable<int>("e2_out", e2_out);
    // get array variables and set in agent variable arrays
    for (int i=0; i<32; i++) {
        int e_temp1 = FLAMEGPU->environment.get<int>("e_array_1", i);
        int e_temp2 = FLAMEGPU->environment.get<int>("e_array_2", i);
        // set values in agent arrary
        FLAMEGPU->setVariable<int, 32>("e_array_out_1", i, e_temp1);
        FLAMEGPU->setVariable<int, 32>("e_array_out_2", i, e_temp2);
    }
    return ALIVE;
}
)###";

FLAMEGPU_STEP_FUNCTION(etc_env_step) {
    // Test Set + Get for scalar (set returns previous value)
    FLAMEGPU->environment.set<int>("e2", 400);
    // Test Set + Get for set by array
    std::array<int, 32> e_array_1;
    for (int i = 0; i < 32; i++) {
        e_array_1[i] = i;  // fill array values
    }
    FLAMEGPU->environment.set<int, 32>("e_array_1", e_array_1);
    // Test Set + Get for set by array index
    std::array<int, 32> e_array_2;
    for (int i = 0; i < 32; i++) {
        e_array_2[i] = i;  // fill array values
        FLAMEGPU->environment.set<int>("e_array_2", i, e_array_2[i]);
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
    env.add<int>("e1", 100);
    env.add<int>("e2", 200);
    env.add<int, 32>("e_array_1", zero_array);
    env.add<int, 32>("e_array_2", zero_array);
    // Init pop with 0 values for variables
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
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
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Set the number of steps to 2
    cuda_model.SimulationConfig().steps = 2;
    cuda_model.simulate();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
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
FLAMEGPU_AGENT_FUNCTION(rtc_agent_output_func, MsgNone, MsgNone) {
    unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
    if (threadIdx.x % 2 == 0) {
        FLAMEGPU->agent_out.setVariable<unsigned int>("x", id + 12);
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
    }
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<unsigned int>("x", i + 1);
        instance.setVariable<unsigned int>("id", i);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step
    cuda_model.step();
    // Test output
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), (unsigned int)(AGENT_COUNT * 1.5));
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
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
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->setVariable<int>("id_out", id);
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("id_out", 0);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    // Check population size is half of initial
    EXPECT_EQ(population.getCurrentListSize("default"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("odd_state"), AGENT_COUNT / 2);
    // check default state
    for (int i = 0; i < static_cast<int>(population.getCurrentListSize("default")); i++) {
        AgentInstance ai = population.getInstanceAt(i, "default");
        int id_out = ai.getVariable<int>("id_out");
        // even id agent in default state should not have updated their id_out value
        EXPECT_EQ(id_out, 0);
    }
    // check odd state
    for (int i = 0; i < static_cast<int>(population.getCurrentListSize("odd_state")); i++) {
        AgentInstance ai = population.getInstanceAt(i, "odd_state");
        int id = ai.getVariable<int>("id");
        int id_out = ai.getVariable<int>("id_out");
        // odd id agent should have updated their id_out value
        EXPECT_EQ(id_out, id);
    }
}

const char* rtc_func_cond_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->setVariable<int>("id_out", id);
    return ALIVE;
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
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<int>("id", i);
        instance.setVariable<int>("id_out", 0);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent);
    cuda_model.getPopulationData(population);
    // Check population size is half of initial
    EXPECT_EQ(population.getCurrentListSize("default"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("odd_state"), AGENT_COUNT / 2);
    // check default state
    for (int i = 0; i < static_cast<int>(population.getCurrentListSize("default")); i++) {
        AgentInstance ai = population.getInstanceAt(i, "default");
        int id_out = ai.getVariable<int>("id_out");
        // even id agent in default state should not have updated their id_out value
        EXPECT_EQ(id_out, 0);
    }
    // check odd state
    for (int i = 0; i < static_cast<int>(population.getCurrentListSize("odd_state")); i++) {
        AgentInstance ai = population.getInstanceAt(i, "odd_state");
        int id = ai.getVariable<int>("id");
        int id_out = ai.getVariable<int>("id_out");
        // odd id agent should have updated their id_out value
        EXPECT_EQ(id_out, id);
    }
}



}  // namespace test_rtc_device_api
