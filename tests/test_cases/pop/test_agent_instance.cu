/**
* Tests of features of Agent Instance
*
* Tests cover:
* > does agent pop create new agents with init values set
* > Do setVariable()/getVariable() throw exceptions on bad name/type
* > Does setVariable() work
*/
#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_agent_population {
const unsigned int INIT_AGENT_COUNT = 10;

TEST(AgentInstanceTest, DefaultVariableValue) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
       AgentInstance instance = population.getNextInstance();
       EXPECT_EQ(instance.getVariable<float>("x"), 0.0f);
       EXPECT_EQ(instance.getVariable<float>("default"), 15.0f);
    }
}
TEST(AgentInstanceTest, GetterBadVarName) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        EXPECT_THROW(instance.getVariable<float>("nope"), InvalidAgentVar);
        EXPECT_THROW(instance.getVariable<float>("this is not valid"), InvalidAgentVar);
    }
}
TEST(AgentInstanceTest, GetterBadVarType) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        EXPECT_THROW(instance.getVariable<int64_t>("x"), InvalidVarType);
        EXPECT_THROW(instance.getVariable<unsigned int>("default"), InvalidVarType);
    }
}
TEST(AgentInstanceTest, SetterBadVarName) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        EXPECT_THROW(instance.setVariable<float>("nope", 1.0f), InvalidAgentVar);
        EXPECT_THROW(instance.setVariable<float>("this is not valid", 1.0f), InvalidAgentVar);
    }
}
TEST(AgentInstanceTest, SetterBadVarType) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        EXPECT_THROW(instance.setVariable<int64_t>("x", static_cast<int64_t>(1)), InvalidVarType);
        EXPECT_THROW(instance.setVariable<unsigned int>("default", 1u), InvalidVarType);
    }
}
TEST(AgentInstanceTest, SetterAndGetterWork) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("x");
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<unsigned int>("x", i);
        EXPECT_EQ(instance.getVariable<unsigned int>("x"), i);
    }
}

    FLAMEGPU_AGENT_FUNCTION(agent_fn_ap1, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
// agent array variable tests
const unsigned int AGENT_COUNT = 1024;
TEST(AgentInstanceTest, SetViaAgentInstance) {
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
        std::array<int, 4> test_array = { 2, 4, 8, 16 };
        EXPECT_EQ(output_array, test_array);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
    }
}
TEST(AgentInstanceTest, SetViaAgentInstance2) {
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
        instance.setVariable<int>("array_var", 0, 2);
        instance.setVariable<int>("array_var", 1, 4);
        instance.setVariable<int>("array_var", 2, 8);
        instance.setVariable<int>("array_var", 3, 16);
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
        std::array<int, 4> test_array = { 2, 4, 8, 16 };
        auto output_val = instance.getVariable<int>("array_var", 0);
        EXPECT_EQ(output_val, test_array[0]);
        output_val = instance.getVariable<int>("array_var", 1);
        EXPECT_EQ(output_val, test_array[1]);
        output_val = instance.getVariable<int>("array_var", 2);
        EXPECT_EQ(output_val, test_array[2]);
        output_val = instance.getVariable<int>("array_var", 3);
        EXPECT_EQ(output_val, test_array[3]);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
    }
}
TEST(AgentInstanceTest, AgentInstance_ArrayDefaultWorks) {
    const std::array<int, 4> TEST_REFERENCE = { 2, 4, 8, 16 };
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<float>("x", 12.0f);
    agent.newVariable<int, 4>("array_var", TEST_REFERENCE);
    agent.newVariable<float>("y", 13.0f);
    // Do nothing, but ensure variables are made available on device
    AgentFunctionDescription &func = agent.newFunction("some_function", agent_fn_ap1);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentPopulation init_population(agent, AGENT_COUNT);
    for (unsigned int i = 0; i< AGENT_COUNT; i++) {
        AgentInstance instance = init_population.getNextInstance("default");
        const auto test = instance.getVariable<int, 4>("array_var");
        ASSERT_EQ(test, TEST_REFERENCE);
        EXPECT_EQ(instance.getVariable<float>("x"), 12.0f);
        EXPECT_EQ(instance.getVariable<float>("y"), 13.0f);
    }
}
TEST(AgentInstanceTest, AgentInstance_ArrayTypeWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Init pop
    AgentPopulation init_population(agent, 1);
    AgentInstance instance = init_population.getNextInstance("default");

    // Use function ptr, can't do more than 1 template arg inside macro
    auto setVarArray4 = &AgentInstance::setVariable<float, 4>;
    auto getVarArray4 = &AgentInstance::getVariable<float, 4>;
    // Check for expected exceptions
    EXPECT_THROW((instance.*setVarArray4)("array_var", { }), InvalidVarType);
    EXPECT_THROW((instance.*getVarArray4)("array_var"), InvalidVarType);
    EXPECT_THROW(instance.setVariable<float>("array_var", 0, 2), InvalidVarType);
    EXPECT_THROW(instance.getVariable<float>("array_var", 0), InvalidVarType);
}
TEST(AgentInstanceTest, AgentInstance_ArrayLenWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int>("x");
    agent.newVariable<int, 4>("array_var");
    // Init pop
    AgentPopulation init_population(agent, 1);
    AgentInstance instance = init_population.getNextInstance("default");
    // Use function ptr, can't do more than 1 template arg inside macro
    auto setVarArray5 = &AgentInstance::setVariable<int, 5>;
    auto getVarArray5 = &AgentInstance::getVariable<int, 5>;
    // Check for expected exceptions
    EXPECT_THROW((instance.*setVarArray5)("x", {}), InvalidVarArrayLen);
    EXPECT_THROW((instance.*getVarArray5)("x"), InvalidVarArrayLen);
    EXPECT_THROW((instance.*setVarArray5)("array_var", {}), InvalidVarArrayLen);
    EXPECT_THROW((instance.*getVarArray5)("array_var"), InvalidVarArrayLen);
    EXPECT_THROW(instance.setVariable<int>("x", 10, 0), OutOfRangeVarArray);
    EXPECT_THROW(instance.getVariable<int>("x", 1), OutOfRangeVarArray);
    EXPECT_THROW(instance.setVariable<int>("array_var", 10, 0), OutOfRangeVarArray);
    EXPECT_THROW(instance.getVariable<int>("array_var", 10), OutOfRangeVarArray);
}
TEST(AgentInstanceTest, AgentInstance_ArrayNameWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Init pop
    AgentPopulation init_population(agent, 1);
    AgentInstance instance = init_population.getNextInstance("default");

    // Use function ptr, can't do more than 1 template arg inside macro
    auto setVarArray4 = &AgentInstance::setVariable<float, 4>;
    auto getVarArray4 = &AgentInstance::getVariable<float, 4>;
    // Check for expected exceptions
    EXPECT_THROW((instance.*setVarArray4)("array_varAAAAAA", {}), InvalidAgentVar);
    EXPECT_THROW((instance.*getVarArray4)("array_varAAAAAA"), InvalidAgentVar);
    EXPECT_THROW(instance.setVariable<int>("array_varAAAAAA", 0, 2), InvalidAgentVar);
    EXPECT_THROW(instance.getVariable<int>("array_varAAAAAA", 0), InvalidAgentVar);
}
TEST(AgentInstanceTest, AgentInstance_ArrayNotSuitable) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Init pop
    AgentPopulation init_population(agent, 1);
    AgentInstance instance = init_population.getNextInstance("default");
    EXPECT_THROW(instance.setVariable<int>("array_var", 0), InvalidAgentVar);
    EXPECT_THROW(instance.getVariable<int>("array_var"), InvalidAgentVar);
}
}  // namespace test_agent_population
