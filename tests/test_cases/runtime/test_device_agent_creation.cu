/**
* Tests of features of device agent creation
*
* Tests cover:
* > agent output same/different agent, mandatory/optional, same/different state, with/without death
* > Device birthed agents have default values set
* Todo:
* > With birthing agent transitioning state
* > With birthing agent conditional state change
*/

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


namespace test_device_agent_creation {
FLAMEGPU_AGENT_FUNCTION(MandatoryOutput, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<float>("x", 12.0f);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OptionalOutput, MsgNone, MsgNone) {
    if (threadIdx.x % 2 == 0)
        FLAMEGPU->agent_out.setVariable<float>("x", 12.0f);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MandatoryOutputWithDeath, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<float>("x", 12.0f);
    return DEAD;
}
FLAMEGPU_AGENT_FUNCTION(OptionalOutputWithDeath, MsgNone, MsgNone) {
    if (threadIdx.x % 2 == 0)
        FLAMEGPU->agent_out.setVariable<float>("x", 12.0f);
    else
        return DEAD;
    return ALIVE;
}
TEST(DeviceAgentCreationTest, Mandatory_Output_SameState) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutput);
    function.setAgentOutput(agent);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT * 2);
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
    EXPECT_EQ(is_1, AGENT_COUNT);
    EXPECT_EQ(is_12, AGENT_COUNT);
}
TEST(DeviceAgentCreationTest, Optional_Output_SameState) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutput);
    function.setAgentOutput(agent);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), (unsigned int)(AGENT_COUNT * 1.5));
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
    EXPECT_EQ(is_1, AGENT_COUNT);
    EXPECT_EQ(is_12, AGENT_COUNT/2);
}
TEST(DeviceAgentCreationTest, Mandatory_Output_DifferentState) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutput);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(ai.getVariable<float>("x"), 1.0f);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentState) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutput);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(ai.getVariable<float>("x"), 1.0f);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Mandatory_Output_SameState_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutputWithDeath);
    function.setAgentOutput(agent);
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
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
    EXPECT_EQ(is_1, 0u);
    EXPECT_EQ(is_12, AGENT_COUNT);
}
TEST(DeviceAgentCreationTest, Optional_Output_SameState_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutputWithDeath);
    function.setAgentOutput(agent);
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
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
    EXPECT_EQ(is_1, AGENT_COUNT / 2);
    EXPECT_EQ(is_12, AGENT_COUNT / 2);
}
TEST(DeviceAgentCreationTest, Mandatory_Output_DifferentState_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutputWithDeath);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), 0u);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentState_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutputWithDeath);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(ai.getVariable<float>("x"), 1.0f);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Mandatory_Output_DifferentAgent) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent2.newVariable<float>("x");
    AgentFunctionDescription &function = agent2.newFunction("output", MandatoryOutput);
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    AgentPopulation newPopulation(model.Agent("agent"));
    cuda_model.getPopulationData(population);
    cuda_model.getPopulationData(newPopulation);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    EXPECT_EQ(newPopulation.getCurrentListSize("b"), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<float>("x"), 1.0f);
    }
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentAgent) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent2.newVariable<float>("x");
    AgentFunctionDescription &function = agent2.newFunction("output", OptionalOutput);
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    AgentPopulation newPopulation(model.Agent("agent"));
    cuda_model.getPopulationData(population);
    cuda_model.getPopulationData(newPopulation);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    EXPECT_EQ(newPopulation.getCurrentListSize("b"), AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<float>("x"), 1.0f);
    }
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Mandatory_Output_DifferentAgent_WithDeath) {
    // 1024 initial agents (type 'agent2') with value 1
    // every agent outputs a new agent  (type 'agent') with value 12, and then dies

    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent2.newVariable<float>("x");
    AgentFunctionDescription &function = agent2.newFunction("output", MandatoryOutputWithDeath);
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    AgentPopulation newPopulation(model.Agent("agent"));
    cuda_model.getPopulationData(population);
    cuda_model.getPopulationData(newPopulation);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), 0u);
    EXPECT_EQ(newPopulation.getCurrentListSize("b"), AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        float val = ai.getVariable<float>("x");
        if (val == 1.0f)
            is_1++;
        else if (val == 12.0f)
            is_12++;
        // EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
    EXPECT_EQ(is_1, 0u);
    EXPECT_EQ(is_12, AGENT_COUNT);
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentAgent_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent2.newVariable<float>("x");
    AgentFunctionDescription &function = agent2.newFunction("output", OptionalOutputWithDeath);
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    AgentPopulation newPopulation(model.Agent("agent"));
    cuda_model.getPopulationData(population);
    cuda_model.getPopulationData(newPopulation);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT / 2);
    EXPECT_EQ(newPopulation.getCurrentListSize("b"), AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<float>("x"), 1.0f);
    }
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, DefaultVariableValue) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    agent2.newVariable<float>("x");
    AgentFunctionDescription &function = agent2.newFunction("output", MandatoryOutput);
    function.setAgentOutput(agent);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", 1.0f);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    AgentPopulation newPopulation(model.Agent("agent"));
    cuda_model.getPopulationData(population);
    cuda_model.getPopulationData(newPopulation);
    // Validate each new agent has default value
    EXPECT_EQ(newPopulation.getCurrentListSize(), AGENT_COUNT);
    unsigned int is_15 = 0;
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize(); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i);
        if (ai.getVariable<float>("default") == 15.0f)
            is_15++;
    }
    EXPECT_EQ(is_15, AGENT_COUNT);
}
}  // namespace test_device_agent_creation
