/**
* Tests of features of Agent Population
*
* Tests cover:
* > does agent pop create new agents with init values set
*/
#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_agent_population {
const unsigned int INIT_AGENT_COUNT = 10;

TEST(HostAgentCreationTest, DefaultVariableValue) {
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
}  // namespace test_agent_population
