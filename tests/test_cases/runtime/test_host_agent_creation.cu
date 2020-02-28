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
}  // namespace test_host_agent_creation
