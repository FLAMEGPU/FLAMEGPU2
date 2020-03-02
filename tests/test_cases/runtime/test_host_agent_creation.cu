/**
* Tests of features on host agent creation
*
* Tests cover:
* > agent output from init/step/host-layer/exit condition
* > agent output to empty/existing pop, multiple states/agents
* > host function birthed agents have default values set
* > Exception thrown if setting/getting wrong variable name/type
* > getVariable() works
*/

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


namespace test_host_agent_creation {
const unsigned int INIT_AGENT_COUNT = 512;
const unsigned int NEW_AGENT_COUNT = 512;
FLAMEGPU_STEP_FUNCTION(BasicOutput) {
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i)
        FLAMEGPU->newAgent("agent").setVariable<float>("x", 1.0f);
}
FLAMEGPU_EXIT_CONDITION(BasicOutputCdn) {
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i)
        FLAMEGPU->newAgent("agent").setVariable<float>("x", 1.0f);
    return CONTINUE;  // New agents wont be created if EXIT is passed
}
FLAMEGPU_STEP_FUNCTION(OutputState) {
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i)
        FLAMEGPU->newAgent("agent", "b").setVariable<float>("x", 1.0f);
}
FLAMEGPU_STEP_FUNCTION(OutputMultiAgent) {
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i) {
        FLAMEGPU->newAgent("agent", "b").setVariable<float>("x", 1.0f);
        FLAMEGPU->newAgent("agent2").setVariable<float>("y", 2.0f);
    }
}
FLAMEGPU_STEP_FUNCTION(BadVarName) {
    FLAMEGPU->newAgent("agent").setVariable<float>("nope", 1.0f);
}
FLAMEGPU_STEP_FUNCTION(BadVarType) {
    FLAMEGPU->newAgent("agent").setVariable<int64_t>("x", static_cast<int64_t>(1.0f));
}
FLAMEGPU_STEP_FUNCTION(Getter) {
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i) {
        auto newAgt = FLAMEGPU->newAgent("agent");
        newAgt.setVariable<float>("x", newAgt.getVariable<float>("default"));
    }
}
FLAMEGPU_STEP_FUNCTION(GetBadVarName) {
    FLAMEGPU->newAgent("agent").getVariable<float>("nope");
}
FLAMEGPU_STEP_FUNCTION(GetBadVarType) {
    FLAMEGPU->newAgent("agent").getVariable<int64_t>("x");
}
TEST(HostAgentCreationTest, FromInit) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addInitFunction(BasicOutput);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 12.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x");
        if (val == 1.0f)
            is_1++;
        else if (val == 12.0f)
            is_12++;
    }
    EXPECT_EQ(is_12, INIT_AGENT_COUNT);
    EXPECT_EQ(is_1, NEW_AGENT_COUNT);
}
TEST(HostAgentCreationTest, FromStep) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(BasicOutput);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 12.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x");
        if (val == 1.0f)
            is_1++;
        else if (val == 12.0f)
            is_12++;
    }
    EXPECT_EQ(is_12, INIT_AGENT_COUNT);
    EXPECT_EQ(is_1, NEW_AGENT_COUNT);
}
TEST(HostAgentCreationTest, FromHostLayer) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.newLayer().addHostFunction(BasicOutput);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 12.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x");
        if (val == 1.0f)
            is_1++;
        else if (val == 12.0f)
            is_12++;
    }
    EXPECT_EQ(is_12, INIT_AGENT_COUNT);
    EXPECT_EQ(is_1, NEW_AGENT_COUNT);
}
TEST(HostAgentCreationTest, FromExitCondition) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addExitCondition(BasicOutputCdn);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 12.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x");
        if (val == 1.0f)
            is_1++;
        else if (val == 12.0f)
            is_12++;
    }
    EXPECT_EQ(is_12, INIT_AGENT_COUNT);
    EXPECT_EQ(is_1, NEW_AGENT_COUNT);
}
TEST(HostAgentCreationTest, FromStepEmptyPop) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(BasicOutput);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"));
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x");
        if (val == 1.0f)
            is_1++;
    }
    EXPECT_EQ(is_1, NEW_AGENT_COUNT);
}
TEST(HostAgentCreationTest, FromStepMultiState) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    model.addStepFunction(OutputState);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", 12.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), INIT_AGENT_COUNT);
    EXPECT_EQ(population.getCurrentListSize("b"), NEW_AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(12.0f, ai.getVariable<float>("x"));
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(1.0f, ai.getVariable<float>("x"));
    }
}
TEST(HostAgentCreationTest, FromStepMultiAgent) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent2.newVariable<float>("y");
    model.addStepFunction(OutputMultiAgent);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(agent, INIT_AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < INIT_AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", 12.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    cuda_model.getPopulationData(population);
    AgentPopulation population2(agent2);
    cuda_model.getPopulationData(population2);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), INIT_AGENT_COUNT);
    EXPECT_EQ(population.getCurrentListSize("b"), NEW_AGENT_COUNT);
    EXPECT_EQ(population2.getCurrentListSize(), NEW_AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(12.0f, ai.getVariable<float>("x"));
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(1.0f, ai.getVariable<float>("x"));
    }
    for (unsigned int i = 0; i < population2.getCurrentListSize(); ++i) {
        AgentInstance ai = population2.getInstanceAt(i);
        EXPECT_EQ(2.0f, ai.getVariable<float>("y"));
    }
}
TEST(HostAgentCreationTest, DefaultVariableValue) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    model.addStepFunction(BasicOutput);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    AgentPopulation population(model.Agent("agent"), NEW_AGENT_COUNT);
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(),  NEW_AGENT_COUNT);
    unsigned int is_15 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("default");
        if (val == 15.0f)
            is_15++;
    }
    EXPECT_EQ(is_15, NEW_AGENT_COUNT);
}
TEST(HostAgentCreationTest, BadVarName) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(BadVarName);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    // Execute model
    EXPECT_THROW(cuda_model.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, BadVarType) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(BadVarType);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    // Execute model
    EXPECT_THROW(cuda_model.step(), InvalidVarType);
}
TEST(HostAgentCreationTest, GetterWorks) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    model.addStepFunction(Getter);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    // Execute model
    cuda_model.SimulationConfig().steps = 1;
    cuda_model.applyConfig();
    cuda_model.simulate();
    // Test output
    AgentPopulation population(model.Agent("agent"), NEW_AGENT_COUNT);
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), NEW_AGENT_COUNT);
    unsigned int is_15 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x");
        if (val == 15.0f)
            is_15++;
    }
    // Every host created agent has had their default loaded from "default" and stored in "x"
    EXPECT_EQ(is_15, NEW_AGENT_COUNT);
}
TEST(HostAgentCreationTest, GetterBadVarName) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(GetBadVarName);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    // Execute model
    EXPECT_THROW(cuda_model.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, GetterBadVarType) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(GetBadVarType);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    // Execute model
    EXPECT_THROW(cuda_model.step(), InvalidVarType);
}

// array variable stuff
const unsigned int AGENT_COUNT = 1024;
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth) {
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = FLAMEGPU->newAgent("agent_name");
        a.setVariable<unsigned int>("id", i);
        a.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
        a.setVariable<int>("array_var2", 0, 3 + i);
        a.setVariable<int>("array_var2", 1, 5 + i);
        a.setVariable<int>("array_var2", 2, 9 + i);
        a.setVariable<int>("array_var2", 3, 17 + i);
        a.setVariable<float>("y", 14.0f + i);
    }
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirthSetGet) {
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = FLAMEGPU->newAgent("agent_name");
        a.setVariable<unsigned int>("id", i);
        // Set
        a.setVariable<int, 4>("array_var", { 2 + i, 4 + i, 8 + i, 16 + i });
        a.setVariable<int>("array_var2", 0, 3 + i);
        a.setVariable<int>("array_var2", 1, 5 + i);
        a.setVariable<int>("array_var2", 2, 9 + i);
        a.setVariable<int>("array_var2", 3, 17 + i);
        a.setVariable<float>("y", 14.0f + i);
        // GetSet
        a.setVariable<int, 4>("array_var", a.getVariable<int, 4>("array_var"));
        a.setVariable<int>("array_var2", 0, a.getVariable<int>("array_var2", 0));
        a.setVariable<int>("array_var2", 1, a.getVariable<int>("array_var2", 1));
        a.setVariable<int>("array_var2", 2, a.getVariable<int>("array_var2", 2));
        a.setVariable<int>("array_var2", 3, a.getVariable<int>("array_var2", 3));
        a.setVariable<float>("y", a.getVariable<float>("y"));
    }
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_DefaultWorks) {
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        FLAMEGPU->newAgent("agent_name");
    }
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_LenWrong) {
    FLAMEGPU->newAgent("agent_name").setVariable<int, 5>("array_var", {});
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_LenWrong2) {
    FLAMEGPU->newAgent("agent_name").setVariable<int>("array_var", 5, 0);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_TypeWrong) {
    FLAMEGPU->newAgent("agent_name").setVariable<float, 4>("array_var", {});
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_TypeWrong2) {
    FLAMEGPU->newAgent("agent_name").setVariable<float>("array_var", 4, 0.0F);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_NameWrong) {
    FLAMEGPU->newAgent("agent_name").setVariable<int, 4>("array_varAAAAAA", {});
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_NameWrong2) {
    FLAMEGPU->newAgent("agent_name").setVariable<int>("array_varAAAAAA", 4, 0);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_ArrayNotSuitableSet) {
    FLAMEGPU->newAgent("agent_name").setVariable<int>("array_var", 12);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_ArrayNotSuitableGet) {
    FLAMEGPU->newAgent("agent_name").getVariable<int>("array_var");
}
TEST(HostAgentCreationTest, HostAgentBirth_ArraySet) {
    const std::array<int, 4> TEST_REFERENCE = { 2, 4, 8, 16 };
    const std::array<int, 4> TEST_REFERENCE2 = { 3, 5, 9, 17 };
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<unsigned int>("id", UINT_MAX);
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<int, 4>("array_var2");
    agent.newVariable<float>("y", 13.0f);
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth);
    CUDAAgentModel sim(model);
    sim.step();
    AgentPopulation population(agent);
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        const unsigned int j = instance.getVariable<unsigned int>("id");
        // Check array sets are correct
        auto array1 = instance.getVariable<int, 4>("array_var");
        auto array2 = instance.getVariable<int, 4>("array_var2");
        for (unsigned int k = 0; k < 4; ++k) {
            array1[k] -= j;
            array2[k] -= j;
        }
        EXPECT_EQ(array1, TEST_REFERENCE);
        EXPECT_EQ(array2, TEST_REFERENCE2);
        EXPECT_EQ(instance.getVariable<float>("y"), 14 + j);
    }
}
TEST(HostAgentCreationTest, HostAgentBirth_ArraySetGet) {
    const std::array<int, 4> TEST_REFERENCE = { 2, 4, 8, 16 };
    const std::array<int, 4> TEST_REFERENCE2 = { 3, 5, 9, 17 };
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<unsigned int>("id", UINT_MAX);
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<int, 4>("array_var2");
    agent.newVariable<float>("y", 13.0f);
    // Run the init function
    model.addStepFunction(ArrayVarHostBirthSetGet);
    CUDAAgentModel sim(model);
    sim.step();
    AgentPopulation population(agent);
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        const unsigned int j = instance.getVariable<unsigned int>("id");
        // Check array sets are correct
        auto array1 = instance.getVariable<int, 4>("array_var");
        auto array2 = instance.getVariable<int, 4>("array_var2");
        for (unsigned int k = 0; k < 4; ++k) {
            array1[k] -= j;
            array2[k] -= j;
        }
        EXPECT_EQ(array1, TEST_REFERENCE);
        EXPECT_EQ(array2, TEST_REFERENCE2);
        EXPECT_EQ(instance.getVariable<float>("y"), 14 + j);
    }
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayDefaultWorks) {
    const std::array<int, 4> TEST_REFERENCE = { 2, 4, 8, 16 };
    const std::array<int, 4> TEST_REFERENCE2 = { 3, 5, 9, 17 };
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<unsigned int>("id", UINT_MAX);
    agent.newVariable<int, 4>("array_var", TEST_REFERENCE);
    agent.newVariable<int, 4>("array_var2", TEST_REFERENCE2);
    agent.newVariable<float>("y", 13.0f);
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_DefaultWorks);
    CUDAAgentModel sim(model);
    sim.step();
    AgentPopulation population(agent);
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        const unsigned int j = instance.getVariable<unsigned int>("id");
        // Check array sets are correct
        auto array1 = instance.getVariable<int, 4>("array_var");
        auto array2 = instance.getVariable<int, 4>("array_var2");
        EXPECT_EQ(instance.getVariable<unsigned int>("id"), UINT_MAX);
        EXPECT_EQ(array1, TEST_REFERENCE);
        EXPECT_EQ(array2, TEST_REFERENCE2);
        EXPECT_EQ(instance.getVariable<float>("y"), 13.0f);
    }
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_LenWrong);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidVarArrayLen);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_LenWrong2);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), OutOfRangeVarArray);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong3) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_LenWrong);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidVarArrayLen);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong4) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_LenWrong2);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), OutOfRangeVarArray);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayTypeWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_TypeWrong);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidVarType);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayTypeWrong2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_TypeWrong2);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidVarType);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNameWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_NameWrong);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNameWrong2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_NameWrong);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNotSuitableSet) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_ArrayNotSuitableSet);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNotSuitableGet) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_ArrayNotSuitableGet);
    CUDAAgentModel sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
}  // namespace test_host_agent_creation
