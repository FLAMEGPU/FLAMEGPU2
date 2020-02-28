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
}  // namespace test_agent_population
