#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_device_api {
    const unsigned int AGENT_COUNT = 1024;
FLAMEGPU_AGENT_FUNCTION(agent_fn_ad_array, MessageNone, MessageNone) {
    if (threadIdx.x % 2 == 0)
        return DEAD;
    return ALIVE;
}
TEST(DeviceAPITest, AgentDeath_array) {
    ModelDescription model("model");
    AgentDescription agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", agent_fn_ad_array);
    func.setAllowAgentDeath(true);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
        instance.setVariable<float>("y", 14.0f);
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cudaSimulation.step();
    // Recover data from device
    AgentVector population(agent);
    cudaSimulation.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarntee
    EXPECT_EQ(population.size(), AGENT_COUNT / 2);
    for (AgentVector::Agent instance : population) {
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
FLAMEGPU_AGENT_FUNCTION(agent_fn_da_set, MessageNone, MessageNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    FLAMEGPU->setVariable<int, 4>("array_var", 0, 2 + FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->setVariable<int, 4>("array_var", 1, 4 + FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->setVariable<int, 4>("array_var", 2, 8 + FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->setVariable<int, 4>("array_var", 3, 16 + FLAMEGPU->getVariable<int>("id"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_da_get, MessageNone, MessageNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    FLAMEGPU->setVariable<int>("a1", FLAMEGPU->getVariable<int, 4>("array_var", 0));
    FLAMEGPU->setVariable<int>("a2", FLAMEGPU->getVariable<int, 4>("array_var", 1));
    FLAMEGPU->setVariable<int>("a3", FLAMEGPU->getVariable<int, 4>("array_var", 2));
    FLAMEGPU->setVariable<int>("a4", FLAMEGPU->getVariable<int, 4>("array_var", 3));
    return ALIVE;
}
TEST(DeviceAPITest, ArraySet) {
    ModelDescription model("model");
    AgentDescription agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", agent_fn_da_set);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<float>("y", 14.0f);
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cudaSimulation.step();
    // Recover data from device
    AgentVector population(agent, AGENT_COUNT);
    cudaSimulation.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarantee
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (AgentVector::Agent instance : population) {
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
    AgentDescription agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    agent.newVariable<int>("a1");
    agent.newVariable<int>("a2");
    agent.newVariable<int>("a3");
    agent.newVariable<int>("a4");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", agent_fn_da_get);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
        instance.setVariable<float>("y", 14.0f);
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cudaSimulation.step();
    // Recover data from device
    AgentVector population(agent, AGENT_COUNT);
    cudaSimulation.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarantee
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (AgentVector::Agent instance : population) {
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
#ifdef FLAMEGPU_USE_GLM
FLAMEGPU_AGENT_FUNCTION(agent_fn_da_set_glm, MessageNone, MessageNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    const int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->setVariable<glm::ivec4>("array_var", glm::ivec4(2 + id, 4 + id, 8 + id, 16 + id));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_da_get_glm, MessageNone, MessageNone) {
    // Read array from `array_var`
    // Store it's values back in `a1` -> `a4`
    const glm::ivec4 t = FLAMEGPU->getVariable<glm::ivec4>("array_var");
    FLAMEGPU->setVariable<int>("a1", t.x);
    FLAMEGPU->setVariable<int>("a2", t.y);
    FLAMEGPU->setVariable<int>("a3", t.z);
    FLAMEGPU->setVariable<int>("a4", t.w);
    return ALIVE;
}
TEST(DeviceAPITest, ArraySet_glm) {
    ModelDescription model("model");
    AgentDescription agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<glm::ivec4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", agent_fn_da_set_glm);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<float>("y", 14.0f);
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cudaSimulation.step();
    // Recover data from device
    AgentVector population(agent, AGENT_COUNT);
    cudaSimulation.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarantee
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (AgentVector::Agent instance : population) {
        // Check neighbouring vars are correct
        EXPECT_EQ(instance.getVariable<float>("x"), 12.0f);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
        int j = instance.getVariable<int>("id");
        // Check array sets are correct
        auto output_array = instance.getVariable<glm::ivec4>("array_var");
        EXPECT_EQ(output_array[0], 2 + j);
        EXPECT_EQ(output_array[1], 4 + j);
        EXPECT_EQ(output_array[2], 8 + j);
        EXPECT_EQ(output_array[3], 16 + j);
    }
}
TEST(DeviceAPITest, ArrayGet_glm) {
    ModelDescription model("model");
    AgentDescription agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<glm::ivec4>("array_var");
    agent.newVariable<float>("y");
    agent.newVariable<int>("id");
    agent.newVariable<int>("a1");
    agent.newVariable<int>("a2");
    agent.newVariable<int>("a3");
    agent.newVariable<int>("a4");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", agent_fn_da_get_glm);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<glm::ivec4>("array_var", glm::ivec4(2 + i, 4 + i, 8 + i, 16 + i));
        instance.setVariable<float>("y", 14.0f);
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    cudaSimulation.step();
    // Recover data from device
    AgentVector population(agent, AGENT_COUNT);
    cudaSimulation.getPopulationData(population);
    // Check data is intact
    // Might need to go more complicate and give different agents different values
    // They should remain in order for such a basic function, but can't guarantee
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (AgentVector::Agent instance : population) {
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
#else
// Mark that test is disabled
TEST(DeviceAPITest, DISABLED_ArraySet_glm) { }
TEST(DeviceAPITest, DISABLED_ArrayGet_glm) { }
#endif

// Test device_api::getStepCounter()
FLAMEGPU_AGENT_FUNCTION(agent_testGetStepCounter, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("step", FLAMEGPU->getStepCounter());
    return ALIVE;
}
TEST(DeviceAPITest, getStepCounter) {
    ModelDescription model("model");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("step");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", agent_testGetStepCounter);
    model.newLayer().addAgentFunction(func);
    // Init pop
    const unsigned int agentCount = 1;
    AgentVector init_population(agent, agentCount);
    for (int i = 0; i< static_cast<int>(agentCount); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<unsigned int>("step", 0);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);

    const unsigned int STEPS = 2;
    for (unsigned int step = 0; step < STEPS; step++) {
        cudaSimulation.step();
        // Recover data from device
        AgentVector population(agent);
        cudaSimulation.getPopulationData(population);
        // Check data is correct.
        for (AgentVector::Agent instance : population) {
            // Check neighbouring vars are correct
            EXPECT_EQ(instance.getVariable<unsigned int>("step"), step);
        }
    }
}

// Only run the function based on the step accessed in the function condition.
// I.e. only run the function once.
FLAMEGPU_AGENT_FUNCTION_CONDITION(condition_testGetStepCounter) {
    return FLAMEGPU->getStepCounter() == 0;
}
FLAMEGPU_AGENT_FUNCTION(condition_testGetStepCounterFunction, MessageNone, MessageNone) {
    // Increment the counter of the number of times the agent ran the function.
    unsigned int count = FLAMEGPU->getVariable<unsigned int>("count");
    FLAMEGPU->setVariable<unsigned int>("count", count + 1);
    return ALIVE;
}
TEST(DeviceAPITest, getStepCounterFunctionCondition) {
    ModelDescription model("model");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("count");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", condition_testGetStepCounterFunction);
    func.setFunctionCondition(condition_testGetStepCounter);
    model.newLayer().addAgentFunction(func);
    // Init pop
    const unsigned int agentCount = 1;
    AgentVector init_population(agent, agentCount);
    for (int i = 0; i< static_cast<int>(agentCount); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<unsigned int>("count", 0);
    }
    const unsigned int STEPS = 4;
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);

    // RUN STEPS steps of simulation.
    cudaSimulation.SimulationConfig().steps = STEPS;
    cudaSimulation.simulate();

    // Recover data from device
    AgentVector population(agent);
    cudaSimulation.getPopulationData(population);
    // Check data is correct.
    const unsigned int EXPECTED_COUNT = 1;
    for (AgentVector::Agent instance : population) {
        // Check neighbouring vars are correct
        EXPECT_EQ(instance.getVariable<unsigned int>("count"), EXPECTED_COUNT);
    }
}

FLAMEGPU_AGENT_FUNCTION(check_agent_name_state_fn, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<int>("correct_name", static_cast<int>(FLAMEGPU->isAgent("agent")));
    FLAMEGPU->setVariable<int>("wrong_name", static_cast<int>(FLAMEGPU->isAgent("agent3")));
    FLAMEGPU->setVariable<int>("correct_state", static_cast<int>(FLAMEGPU->isState("state")));
    FLAMEGPU->setVariable<int>("wrong_state", static_cast<int>(FLAMEGPU->isState("state5")));
    return ALIVE;
}

TEST(DeviceAPITest, check_agent_name_state) {
    ModelDescription model("model");
    AgentDescription agent = model.newAgent("agent");
    agent.newState("state");
    agent.newState("state8");
    agent.newVariable<int>("correct_name", -1);
    agent.newVariable<int>("wrong_name", -1);
    agent.newVariable<int>("correct_state", -1);
    agent.newVariable<int>("wrong_state", -1);
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription func = agent.newFunction("some_function", check_agent_name_state_fn);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);

    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population, "state");

    // Run 1 step to ensure data is pushed to device
    cudaSimulation.step();
    // Recover data from device
    AgentVector population(agent, AGENT_COUNT);
    cudaSimulation.getPopulationData(population, "state");
    // Check results are correct
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (const auto &instance : population) {
        EXPECT_EQ(instance.getVariable<int>("correct_name"), 1);
        EXPECT_EQ(instance.getVariable<int>("wrong_name"), 0);
        EXPECT_EQ(instance.getVariable<int>("correct_state"), 1);
        EXPECT_EQ(instance.getVariable<int>("wrong_state"), 0);
    }
}

}  // namespace test_device_api
}  // namespace flamegpu
