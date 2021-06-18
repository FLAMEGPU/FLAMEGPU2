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
#include <set>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_host_agent_creation {
const unsigned int INIT_AGENT_COUNT = 512;
const unsigned int NEW_AGENT_COUNT = 512;
FLAMEGPU_STEP_FUNCTION(BasicOutput) {
    auto t = FLAMEGPU->agent("agent");
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i)
        t.newAgent().setVariable<float>("x", 1.0f);
}
FLAMEGPU_EXIT_CONDITION(BasicOutputCdn) {
    auto t = FLAMEGPU->agent("agent");
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i)
        t.newAgent().setVariable<float>("x", 1.0f);
    return CONTINUE;  // New agents wont be created if EXIT is passed
}
FLAMEGPU_STEP_FUNCTION(OutputState) {
    auto t = FLAMEGPU->agent("agent", "b");
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i)
        t.newAgent().setVariable<float>("x", 1.0f);
}
FLAMEGPU_STEP_FUNCTION(OutputMultiAgent) {
    auto t = FLAMEGPU->agent("agent", "b");
    auto t2 = FLAMEGPU->agent("agent2");
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i) {
        t.newAgent().setVariable<float>("x", 1.0f);
        t2.newAgent().setVariable<float>("y", 2.0f);
    }
}
FLAMEGPU_STEP_FUNCTION(BadVarName) {
    FLAMEGPU->agent("agent").newAgent().setVariable<float>("nope", 1.0f);
}
FLAMEGPU_STEP_FUNCTION(BadVarType) {
    FLAMEGPU->agent("agent").newAgent().setVariable<int64_t>("x", static_cast<int64_t>(1.0f));
}
FLAMEGPU_STEP_FUNCTION(Getter) {
    for (unsigned int i = 0; i < NEW_AGENT_COUNT; ++i) {
        auto newAgt = FLAMEGPU->agent("agent").newAgent();
        newAgt.setVariable<float>("x", newAgt.getVariable<float>("default"));
    }
}
FLAMEGPU_STEP_FUNCTION(GetBadVarName) {
    FLAMEGPU->agent("agent").newAgent().getVariable<float>("nope");
}
FLAMEGPU_STEP_FUNCTION(GetBadVarType) {
    FLAMEGPU->agent("agent").newAgent().getVariable<int64_t>("x");
}
TEST(HostAgentCreationTest, FromInit) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addInitFunction(BasicOutput);
    // Init agent pop
    CUDASimulation cudaSimulation(model);
    AgentVector population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (AgentVector::Agent instance : population) {
        instance.setVariable<float>("x", 12.0f);
    }
    cudaSimulation.setPopulationData(population);
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (AgentVector::Agent ai : population) {
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
    CUDASimulation cudaSimulation(model);
    AgentVector population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (AgentVector::Agent instance : population) {
        instance.setVariable<float>("x", 12.0f);
    }
    cudaSimulation.setPopulationData(population);
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (AgentVector::Agent ai : population) {
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
    CUDASimulation cudaSimulation(model);
    AgentVector population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (AgentVector::Agent instance : population) {
        instance.setVariable<float>("x", 12.0f);
    }
    cudaSimulation.setPopulationData(population);
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (AgentVector::Agent ai : population) {
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
    CUDASimulation cudaSimulation(model);
    AgentVector population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (AgentVector::Agent instance : population) {
        instance.setVariable<float>("x", 12.0f);
    }
    cudaSimulation.setPopulationData(population);
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(), INIT_AGENT_COUNT + NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (AgentVector::Agent ai : population) {
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
    CUDASimulation cudaSimulation(model);
    AgentVector population(model.Agent("agent"));
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(), NEW_AGENT_COUNT);
    unsigned int is_1 = 0;
    for (AgentVector::Agent ai : population) {
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
    CUDASimulation cudaSimulation(model);
    AgentVector population(model.Agent("agent"), INIT_AGENT_COUNT);
    // Initialise agents
    for (AgentVector::Agent instance : population) {
        instance.setVariable<float>("x", 12.0f);
    }
    cudaSimulation.setPopulationData(population, "a");
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    AgentVector population_a(model.Agent("agent"));
    AgentVector population_b(model.Agent("agent"));
    cudaSimulation.getPopulationData(population_a, "a");
    cudaSimulation.getPopulationData(population_b, "b");
    // Validate each agent has same result
    EXPECT_EQ(population_a.size(), INIT_AGENT_COUNT);
    EXPECT_EQ(population_b.size(), NEW_AGENT_COUNT);
    for (AgentVector::Agent ai : population_a) {
        EXPECT_EQ(12.0f, ai.getVariable<float>("x"));
    }
    for (AgentVector::Agent ai : population_b) {
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
    CUDASimulation cudaSimulation(model);
    AgentVector population(agent, INIT_AGENT_COUNT);
    // Initialise agents
    for (AgentVector::Agent instance : population) {
        instance.setVariable<float>("x", 12.0f);
    }
    cudaSimulation.setPopulationData(population, "a");
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    AgentVector population_a(model.Agent("agent"));
    AgentVector population_b(model.Agent("agent"));
    AgentVector population_2(model.Agent("agent2"));
    cudaSimulation.getPopulationData(population_a, "a");
    cudaSimulation.getPopulationData(population_b, "b");
    cudaSimulation.getPopulationData(population_2);
    // Validate each agent has same result
    EXPECT_EQ(population_a.size(), INIT_AGENT_COUNT);
    EXPECT_EQ(population_b.size(), NEW_AGENT_COUNT);
    EXPECT_EQ(population_2.size(), NEW_AGENT_COUNT);
    for (AgentVector::Agent ai : population_a) {
        EXPECT_EQ(12.0f, ai.getVariable<float>("x"));
    }
    for (AgentVector::Agent ai : population_b) {
        EXPECT_EQ(1.0f, ai.getVariable<float>("x"));
    }
    for (AgentVector::Agent ai : population_2) {
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
    CUDASimulation cudaSimulation(model);
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    AgentVector population(model.Agent("agent"), NEW_AGENT_COUNT);
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(),  NEW_AGENT_COUNT);
    unsigned int is_15 = 0;
    for (AgentVector::Agent ai : population) {
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
    CUDASimulation cudaSimulation(model);
    // Execute model
    EXPECT_THROW(cudaSimulation.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, BadVarType) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(BadVarType);
    // Init agent pop
    CUDASimulation cudaSimulation(model);
    // Execute model
    EXPECT_THROW(cudaSimulation.step(), InvalidVarType);
}
TEST(HostAgentCreationTest, GetterWorks) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    model.addStepFunction(Getter);
    // Init agent pop
    CUDASimulation cudaSimulation(model);
    // Execute model
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.applyConfig();
    cudaSimulation.simulate();
    // Test output
    AgentVector population(model.Agent("agent"), NEW_AGENT_COUNT);
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.size(), NEW_AGENT_COUNT);
    unsigned int is_15 = 0;
    for (AgentVector::Agent ai : population) {
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
    CUDASimulation cudaSimulation(model);
    // Execute model
    EXPECT_THROW(cudaSimulation.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, GetterBadVarType) {
    // Define model
    ModelDescription model("TestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    model.addStepFunction(GetBadVarType);
    // Init agent pop
    CUDASimulation cudaSimulation(model);
    // Execute model
    EXPECT_THROW(cudaSimulation.step(), InvalidVarType);
}

// array variable stuff
const unsigned int AGENT_COUNT = 1024;
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth) {
    auto t = FLAMEGPU->agent("agent_name");
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = t.newAgent();
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
    auto t = FLAMEGPU->agent("agent_name");
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = t.newAgent();
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
    auto t = FLAMEGPU->agent("agent_name");
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        t.newAgent();
    }
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_LenWrong) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<int, 5>("array_var", {});
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_LenWrong2) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<int>("array_var", 5, 0);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_TypeWrong) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<float, 4>("array_var", {});
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_TypeWrong2) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<float>("array_var", 4, 0.0F);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_NameWrong) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<int, 4>("array_varAAAAAA", {});
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_NameWrong2) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<int>("array_varAAAAAA", 4, 0);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_ArrayNotSuitableSet) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<int>("array_var", 12);
}
FLAMEGPU_STEP_FUNCTION(ArrayVarHostBirth_ArrayNotSuitableGet) {
    FLAMEGPU->agent("agent_name").newAgent().getVariable<int>("array_var");
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
    CUDASimulation sim(model);
    sim.step();
    AgentVector population(agent);
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (AgentVector::Agent instance : population) {
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
    CUDASimulation sim(model);
    sim.step();
    AgentVector population(agent);
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (AgentVector::Agent instance : population) {
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
    CUDASimulation sim(model);
    sim.step();
    AgentVector population(agent);
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.size(), AGENT_COUNT);
    for (AgentVector::Agent instance : population) {
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
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidVarArrayLen);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_LenWrong2);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), OutOfRangeVarArray);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong3) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_LenWrong);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidVarArrayLen);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayLenWrong4) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_LenWrong2);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), OutOfRangeVarArray);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayTypeWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_TypeWrong);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidVarType);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayTypeWrong2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_TypeWrong2);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidVarType);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNameWrong) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_NameWrong);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNameWrong2) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_NameWrong);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNotSuitableSet) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_ArrayNotSuitableSet);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
TEST(HostAgentCreationTest, HostAgentBirth_ArrayNotSuitableGet) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<int, 4>("array_var");
    // Run the init function
    model.addStepFunction(ArrayVarHostBirth_ArrayNotSuitableGet);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), InvalidAgentVar);
}
FLAMEGPU_STEP_FUNCTION(reserved_name_step) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<int>("_", 0);
}
FLAMEGPU_STEP_FUNCTION(reserved_name_step_array) {
    FLAMEGPU->agent("agent_name").newAgent().setVariable<int, 3>("_", {});
}
TEST(HostAgentCreationTest, reserved_name) {
    ModelDescription model("model");
    model.newAgent("agent_name");
    // Run the init function
    model.addStepFunction(reserved_name_step);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), ReservedName);
}
TEST(HostAgentCreationTest, reserved_name_array) {
    ModelDescription model("model");
    model.newAgent("agent_name");
    model.addStepFunction(reserved_name_step_array);
    CUDASimulation sim(model);
    EXPECT_THROW(sim.step(), ReservedName);
}
FLAMEGPU_HOST_FUNCTION(AgentID_HostNewAgentBirth) {
    const uint32_t birth_ct_a = FLAMEGPU->agent("agent", "a").count();
    const uint32_t birth_ct_b = FLAMEGPU->agent("agent", "b").count();

    for (uint32_t i = 0; i < birth_ct_a; ++i) {
        auto t = FLAMEGPU->agent("agent", "a").newAgent();
        t.setVariable<id_t>("id_copy", t.getID());
    }
    for (uint32_t i = 0; i < birth_ct_b; ++i) {
        auto t = FLAMEGPU->agent("agent", "b").newAgent();
        t.setVariable<id_t>("id_copy", t.getID());
    }
}
TEST(HostAgentCreationTest, AgentID_HostNewAgent_MultipleStatesUniqueIDs) {
    const uint32_t POP_SIZE = 100;
    // Create agents via AgentVector to two agent states
    // HostAgent Birth creates new agent in both states
    // Store agent IDs to an agent variable inside model
    // Export agents and check their IDs are unique
    // Also check that the id's copied during model match those at export

    ModelDescription model("test_agentid");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_copy", ID_NOT_SET);
    agent.newState("a");
    agent.newState("b");

    auto& layer_a = model.newLayer();
    layer_a.addHostFunction(AgentID_HostNewAgentBirth);

    AgentVector pop_in(agent, POP_SIZE);

    CUDASimulation sim(model);
    sim.setPopulationData(pop_in, "a");
    sim.setPopulationData(pop_in, "b");

    sim.step();

    AgentVector pop_out_a(agent);
    AgentVector pop_out_b(agent);

    sim.getPopulationData(pop_out_a, "a");
    sim.getPopulationData(pop_out_b, "b");

    std::set<id_t> ids;
    // Validate that there are no ID collisions
    for (auto a : pop_out_a) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    for (auto a : pop_out_b) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids.size(), 4 * POP_SIZE);  // No collisions
}
FLAMEGPU_HOST_FUNCTION(AgentID_HostNewAgentBirth2) {
    const uint32_t birth_ct_a = FLAMEGPU->agent("agent").count();
    const uint32_t birth_ct_b = FLAMEGPU->agent("agent2").count();

    for (uint32_t i = 0; i < birth_ct_a; ++i) {
        auto t = FLAMEGPU->agent("agent").newAgent();
        t.setVariable<id_t>("id_copy", t.getID());
    }
    for (uint32_t i = 0; i < birth_ct_b; ++i) {
        auto t = FLAMEGPU->agent("agent2").newAgent();
        t.setVariable<id_t>("id_copy", t.getID());
    }
}
TEST(HostAgentCreationTest, AgentID_MultipleAgents) {
    const uint32_t POP_SIZE = 100;
    // Create agents via AgentVector to two agent types
    // HostAgent Birth creates new agent in both types
    // Store agent IDs to an agent variable inside model
    // Export agents and check their IDs are unique
    // Also check that the id's copied during model match those at export

    ModelDescription model("test_agentid");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_copy", ID_NOT_SET);
    AgentDescription& agent2 = model.newAgent("agent2");
    agent2.newVariable<id_t>("id_copy", ID_NOT_SET);

    auto& layer_a = model.newLayer();
    layer_a.addHostFunction(AgentID_HostNewAgentBirth2);

    AgentVector pop_in_a(agent, POP_SIZE);
    AgentVector pop_in_b(agent2, POP_SIZE);

    CUDASimulation sim(model);
    sim.setPopulationData(pop_in_a);
    sim.setPopulationData(pop_in_b);

    sim.step();

    AgentVector pop_out_a(agent);
    AgentVector pop_out_b(agent);

    sim.getPopulationData(pop_out_a);
    sim.getPopulationData(pop_out_b);

    std::set<id_t> ids_a, ids_b;
    // Validate that there are no ID collisions
    for (auto a : pop_out_a) {
        ids_a.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids_a.size(), 2 * POP_SIZE);  // No collisions
    for (auto a : pop_out_b) {
        ids_b.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids_b.size(), 2 * POP_SIZE);  // No collisions
}
}  // namespace test_host_agent_creation
}  // namespace flamegpu
