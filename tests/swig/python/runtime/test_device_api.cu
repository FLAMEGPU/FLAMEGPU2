#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_device_api {
    const unsigned int AGENT_COUNT = 1024;
FLAMEGPU_AGENT_FUNCTION(agent_fn_ad_array, MsgNone, MsgNone) {
    if (threadIdx.x % 2 == 0)
        return DEAD;
    return ALIVE;
}
TEST(DeviceAPITest, AgentDeath_array) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription &func = agent.newFunction("some_function", agent_fn_ad_array);
    func.setAllowAgentDeath(true);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
        instance.setVariable<float>("y", 14.0f);
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
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarntee
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        // Check neighbouring vars are correct
        EXPECT_EQ(instance.getVariable<float>("x"), 12.0f);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
        int j = instance.getVariable<int>("id");
        // Check array sets are correct
        auto output_array = instance.getVariable<int, 4>("array_var");
        EXPECT_EQ(output_array[0], 2 + j);
        EXPECT_EQ(output_array[1], 4 + j);
        EXPECT_EQ(output_array[2], 8 + j);
        EXPECT_EQ(output_array[3], 16 + j);
    }
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_da_set, MsgNone, MsgNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    FLAMEGPU->setVariable<int, 4>("array_var", 0, 2 + FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->setVariable<int, 4>("array_var", 1, 4 + FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->setVariable<int, 4>("array_var", 2, 8 + FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->setVariable<int, 4>("array_var", 3, 16 + FLAMEGPU->getVariable<int>("id"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_da_get, MsgNone, MsgNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    FLAMEGPU->setVariable<int>("a1", FLAMEGPU->getVariable<int, 4>("array_var", 0));
    FLAMEGPU->setVariable<int>("a2", FLAMEGPU->getVariable<int, 4>("array_var", 1));
    FLAMEGPU->setVariable<int>("a3", FLAMEGPU->getVariable<int, 4>("array_var", 2));
    FLAMEGPU->setVariable<int>("a4", FLAMEGPU->getVariable<int, 4>("array_var", 3));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_da_arrayunsuitable, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<int>("array_var", 0);
    FLAMEGPU->setVariable<int>("array_var", 0);
    FLAMEGPU->setVariable<int>("array_var", 0);
    FLAMEGPU->setVariable<int>("array_var", 0);
    FLAMEGPU->setVariable<int>("var", FLAMEGPU->getVariable<int>("array_var"));
    return ALIVE;
}
TEST(DeviceAPITest, ArraySet) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription &func = agent.newFunction("some_function", agent_fn_da_set);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<float>("y", 14.0f);
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent, AGENT_COUNT);
    cuda_model.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarntee
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        // Check neighbouring vars are correct
        EXPECT_EQ(instance.getVariable<float>("x"), 12.0f);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
        int j = instance.getVariable<int>("id");
        // Check array sets are correct
        auto output_array = instance.getVariable<int, 4>("array_var");
        EXPECT_EQ(output_array[0], 2 + j);
        EXPECT_EQ(output_array[1], 4 + j);
        EXPECT_EQ(output_array[2], 8 + j);
        EXPECT_EQ(output_array[3], 16 + j);
    }
}
TEST(DeviceAPITest, ArrayGet) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    agent.newVariable<int>("a1");
    agent.newVariable<int>("a2");
    agent.newVariable<int>("a3");
    agent.newVariable<int>("a4");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription &func = agent.newFunction("some_function", agent_fn_da_get);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
        instance.setVariable<float>("y", 14.0f);
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent, AGENT_COUNT);
    cuda_model.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarntee
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        // Check neighbouring vars are correct
        EXPECT_EQ(instance.getVariable<float>("x"), 12.0f);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
        int j = instance.getVariable<int>("id");
        // Check array sets are correct
        EXPECT_EQ(instance.getVariable<int>("a1"), 2 + j);
        EXPECT_EQ(instance.getVariable<int>("a2"), 4 + j);
        EXPECT_EQ(instance.getVariable<int>("a3"), 8 + j);
        EXPECT_EQ(instance.getVariable<int>("a4"), 16 + j);
    }
}
TEST(DeviceAPITest, ArrayUnsuitable) {
    // Cant pass array var to arraySet, will not change variable
    // Can't pass array var to get, so 0 will be returned
    const std::array<int, 4> TEST_REFERENCE = { 3, 5, 9, 17 };
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var", TEST_REFERENCE);
    agent.newVariable<int>("var", 1);
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription &func = agent.newFunction("some_function", agent_fn_da_arrayunsuitable);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cuda_model.step();
    // Recover data from device
    AgentPopulation population(agent, AGENT_COUNT);
    cuda_model.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarntee
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        auto test = instance.getVariable<int, 4>("array_var");
        EXPECT_EQ(test, TEST_REFERENCE);
        EXPECT_EQ(instance.getVariable<int>("var"), 0);
    }
}

}  // namespace test_device_api
