#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_agent_array_variables {
unsigned int AGENT_COUNT = 1024;
//////////////// These tests will eventually be moved to TestAgentPop
/**
 * Tests that an agent variable array can be set and recovered before and after the model
 * Data is not set/get or written to
 */
FLAMEGPU_AGENT_FUNCTION(agent_fn_ap1, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
TEST(AgentArrayVariablesTest, SetViaAgentInstance) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription &func = agent.newFunction("some_function", agent_fn_ap1);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (unsigned int i = 0; i< AGENT_COUNT; i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<int, 4>("array_var", {2, 4, 8, 16});
        instance.setVariable<float>("y", 14.0f);
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
        EXPECT_EQ(instance.getVariable<float>("x"), 12.0f);
        auto output_array = instance.getVariable<int, 4>("array_var");
        auto test_array = std::array<int, 4>({ 2, 4, 8, 16 });
        EXPECT_EQ(output_array, test_array);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
    }
}
TEST(AgentArrayVariablesTest, SetViaAgentInstance2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x");
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription &func = agent.newFunction("some_function", agent_fn_ap1);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (unsigned int i = 0; i< AGENT_COUNT; i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        instance.setVariable<float>("x", 12.0f);
        instance.setVariable<int, 4>("array_var", 0, 2);
        instance.setVariable<int, 4>("array_var", 1, 4);
        instance.setVariable<int, 4>("array_var", 2, 8);
        instance.setVariable<int, 4>("array_var", 3, 16);
        instance.setVariable<float>("y", 14.0f);
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
        EXPECT_EQ(instance.getVariable<float>("x"), 12.0f);
        auto test_array = std::array<int, 4>({ 2, 4, 8, 16 });
        auto output_val = instance.getVariable<int, 4>("array_var", 0);
        EXPECT_EQ(output_val, test_array[0]);
        output_val = instance.getVariable<int, 4>("array_var", 1);
        EXPECT_EQ(output_val, test_array[1]);
        output_val = instance.getVariable<int, 4>("array_var", 2);
        EXPECT_EQ(output_val, test_array[2]);
        output_val = instance.getVariable<int, 4>("array_var", 3);
        EXPECT_EQ(output_val, test_array[3]);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
    }
}
TEST(AgentArrayVariablesTest, AgentInstance_ArrayTypeWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Init pop
    AgentPopulation init_population(agent, 1);
    AgentInstance instance = init_population.getNextInstance("default");
#define float_4 float, 4
    EXPECT_THROW(instance.setVariable<float_4>("array_var", { }), InvalidVarType);
    EXPECT_THROW(instance.setVariable<float_4>("array_var", 0, 2), InvalidVarType);
    EXPECT_THROW(instance.getVariable<float_4>("array_var"), InvalidVarType);
    EXPECT_THROW(instance.getVariable<float_4>("array_var", 0), InvalidVarType);
#undef int_4
}
TEST(AgentArrayVariablesTest, AgentPopArrayLenWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int>("x");
    agent.newVariable<int, 4>("array_var");
    // Init pop
    AgentPopulation init_population(agent, 1);
    AgentInstance instance = init_population.getNextInstance("default");
#define int_5 int, 5
    EXPECT_THROW(instance.setVariable<int_5>("x", {}), InvalidAgentVar);
    EXPECT_THROW(instance.setVariable<int_5>("x", 0, 2), InvalidAgentVar);
    EXPECT_THROW(instance.getVariable<int_5>("x"), InvalidAgentVar);
    EXPECT_THROW(instance.getVariable<int_5>("x", 0), InvalidAgentVar);
    EXPECT_THROW(instance.setVariable<int_5>("array_var", {}), InvalidAgentVar);
    EXPECT_THROW(instance.setVariable<int_5>("array_var", 0, 2), InvalidAgentVar);
    EXPECT_THROW(instance.getVariable<int_5>("array_var"), InvalidAgentVar);
    EXPECT_THROW(instance.getVariable<int_5>("array_var", 0), InvalidAgentVar);
#undef int_5
}

/////////// These tests will eventually be moved to device api or something
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
TEST(AgentArrayVariablesTest, DeviceAPI_ArraySet) {
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
TEST(AgentArrayVariablesTest, DeviceAPI_ArrayGet) {
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

////////////// These tests will eventually be moved to FGPU Host api
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_sum1) {
    FLAMEGPU->agent("agent_name").sum<int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_sum2) {
    FLAMEGPU->agent("agent_name").sum<int, int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_min) {
    FLAMEGPU->agent("agent_name").min<int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_max) {
    FLAMEGPU->agent("agent_name").max<int>("array_var");
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_count) {
    FLAMEGPU->agent("agent_name").count<int>("array_var", 0);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_hist1) {
    FLAMEGPU->agent("agent_name").histogramEven<int>("array_var", 10, 0, 9);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_hist2) {
    FLAMEGPU->agent("agent_name").histogramEven<int, int>("array_var", 10, 0, 9);
}
FLAMEGPU_CUSTOM_REDUCTION(SampleReduction, a, b) {
        return a + b;
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_reduce) {
    FLAMEGPU->agent("agent_name").reduce<int>("array_var", SampleReduction, 0);
}
FLAMEGPU_CUSTOM_TRANSFORM(SampleTransform, a) {
    return a + 1;
}
FLAMEGPU_STEP_FUNCTION(ArrayVarNotSupported_transformReduce) {
    FLAMEGPU->agent("agent_name").transformReduce<int>("array_var", SampleTransform, SampleReduction, 0);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_sum1) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_sum1);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_sum2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_sum2);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_min) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_min);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_max) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_max);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_count) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_count);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_hist1) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_hist1);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_hist2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_hist2);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_reduce) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_reduce);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
TEST(AgentArrayVariablesTest, HostReduceApi_ArrayVarNotSupported_transformReduce) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Add the function to be tested
    model.addStepFunction(ArrayVarNotSupported_transformReduce);
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cuda_model.step(), UnsupportedVarType);
}
////////////////// These tests will eventually be move to agent death?
// Do agent variables remain intact through agent death

FLAMEGPU_AGENT_FUNCTION(agent_fn_ad_array, MsgNone, MsgNone) {
    if (threadIdx.x % 2 == 0)
        return DEAD;
    return ALIVE;
}
TEST(AgentArrayVariablesTest, AgentDeath_array) {
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



//////////////// These tests will eventually be moved to host agent birth
TEST(AgentArrayVariablesTest, SetViaHostAgentBirth) { }
//////////////// These tests will eventually be moved to device agent birth
TEST(AgentArrayVariablesTest, SetViaDeviceAgentBirth) { }
}  // namespace test_agent_array_variables
