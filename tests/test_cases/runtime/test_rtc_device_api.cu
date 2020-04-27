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
    // Do nothing, but ensure variables are made available on device
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
    // Do nothing, but ensure variables are made available on device
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
    // Do nothing, but ensure variables are made available on device
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
    // Do nothing, but ensure variables are made available on device
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
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
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
    // Do nothing, but ensure variables are made available on device
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
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
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

}  // namespace test_rtc_device_api
