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
 * Test an empty agent function to ensure that the RTC library can successfull build and run a minimal example
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
 */
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
        // expected to throw an excpetion before running due to agent function compile error
        CUDAAgentModel cuda_model(model);
        cuda_model.setPopulationData(init_population);
        // Run 1 step to ensure agent function compiles and runs
        cuda_model.step();
    }, InvalidAgentFunc);
}

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
    int a1 = FLAMEGPU->getVariable<int>("a1");
    int a2 = FLAMEGPU->getVariable<int>("a2");
    int a3 = FLAMEGPU->getVariable<int>("a3");
    int a4 = FLAMEGPU->getVariable<int>("a4");
    FLAMEGPU->setVariable<int, 4>("array_var", 0, a1);
    FLAMEGPU->setVariable<int, 4>("array_var", 1, a2);
    FLAMEGPU->setVariable<int, 4>("array_var", 2, a3);
    FLAMEGPU->setVariable<int, 4>("array_var", 3, a4);
    return ALIVE;
}
)###";
/**
 * Test an RTC function to ensure that the setVariable function works correctly for array variables. Expected result is a1, a2, a3 and a4 are copied into 'array_var'.
 */
TEST(DeviceRTCAPITest, AgentFunction_array_set) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    agent.newVariable<int, 4>("array_var");
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
        instance.setVariable<int>("a1", 2 + i);
        instance.setVariable<int>("a2", 4 + i);
        instance.setVariable<int>("a3", 8 + i);
        instance.setVariable<int>("a4", 16 + i);
        instance.setVariable<int, 4>("array_var", { 0, 0, 0, 0 });
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
        std::array<int, 4> array_var = instance.getVariable<int, 4>("array_var");
        EXPECT_EQ(array_var[0], 2 + j);
        EXPECT_EQ(array_var[1], 4 + j);
        EXPECT_EQ(array_var[2], 8 + j);
        EXPECT_EQ(array_var[3], 16 + j);
    }
}

}  // namespace test_rtc_device_api
