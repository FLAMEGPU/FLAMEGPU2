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

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"


namespace test_device_agent_creation {
FLAMEGPU_AGENT_FUNCTION(MandatoryOutput, MsgNone, MsgNone) {
    unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
    FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0f);
    FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(OptionalOutput, MsgNone, MsgNone) {
    unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
    if (threadIdx.x % 2 == 0) {
        FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0f);
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MandatoryOutputWithDeath, MsgNone, MsgNone) {
    unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
    FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0f);
    FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
    return DEAD;
}
FLAMEGPU_AGENT_FUNCTION(OptionalOutputWithDeath, MsgNone, MsgNone) {
    unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
    if (threadIdx.x % 2 == 0) {
        FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0f);
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
    } else {
        return DEAD;
    }
    return ALIVE;
}
TEST(DeviceAgentCreationTest, Mandatory_Output_SameState) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutput);
    function.setAgentOutput(agent);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        const unsigned int id = ai.getVariable<unsigned int>("id");
        const float val = ai.getVariable<float>("x") - id;
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
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutput);
    function.setAgentOutput(agent);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        const unsigned int id = ai.getVariable<unsigned int>("id");
        const float val = ai.getVariable<float>("x") - id;
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
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutput);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentState) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutput);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Mandatory_Output_SameState_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutputWithDeath);
    function.setAgentOutput(agent);
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        const unsigned int id = ai.getVariable<unsigned int>("id");
        const float val = ai.getVariable<float>("x") - id;
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
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutputWithDeath);
    function.setAgentOutput(agent);
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        const unsigned int id = ai.getVariable<unsigned int>("id");
        const float val = ai.getVariable<float>("x") - id;
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
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutputWithDeath);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentState_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutputWithDeath);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
    }
}
    // Tests beyond here all also check id % 2 or id % 4
TEST(DeviceAgentCreationTest, Mandatory_Output_DifferentAgent) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    agent2.newVariable<float>("x");
    agent2.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent2.newFunction("output", MandatoryOutput);
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
    unsigned int is_1_mod2_0 = 0;
    unsigned int is_1_mod2_1 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        if (ai.getVariable<unsigned int>("id") % 2 == 0) {
            is_1_mod2_0++;
        } else {
            is_1_mod2_1++;
        }
    }
    EXPECT_EQ(is_1_mod2_0, AGENT_COUNT / 2);
    EXPECT_EQ(is_1_mod2_1, AGENT_COUNT / 2);
    unsigned int is_12_mod2_0 = 0;
    unsigned int is_12_mod2_1 = 0;
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
        if (ai.getVariable<unsigned int>("id") % 2 == 0) {
            is_12_mod2_0++;
        } else {
            is_12_mod2_1++;
        }
    }
    EXPECT_EQ(is_12_mod2_0, AGENT_COUNT / 2);
    EXPECT_EQ(is_12_mod2_1, AGENT_COUNT / 2);
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentAgent) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    agent2.newVariable<float>("x");
    agent2.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent2.newFunction("output", OptionalOutput);
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
    unsigned int is_1_mod2_0 = 0;
    unsigned int is_1_mod2_1 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        if (ai.getVariable<unsigned int>("id") % 2 == 0) {
            is_1_mod2_0++;
        } else {
            is_1_mod2_1++;
        }
    }
    EXPECT_EQ(is_1_mod2_0, AGENT_COUNT / 2);
    EXPECT_EQ(is_1_mod2_1, AGENT_COUNT / 2);
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
        EXPECT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
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
    agent.newVariable<unsigned int>("id");
    agent2.newVariable<float>("x");
    agent2.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent2.newFunction("output", MandatoryOutputWithDeath);
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
    unsigned int is_12_mod2_0 = 0;
    unsigned int is_12_mod2_1 = 0;
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        const unsigned int id = ai.getVariable<unsigned int>("id");
        const float val = ai.getVariable<float>("x") - id;
        if (val == 1.0f) {
            is_1++;
       } else if (val == 12.0f) {
            is_12++;
            if (ai.getVariable<unsigned int>("id") % 2 == 0) {
                is_12_mod2_0++;
            } else {
                is_12_mod2_1++;
            }
       }
    }
    EXPECT_EQ(is_1, 0u);
    EXPECT_EQ(is_12, AGENT_COUNT);
    EXPECT_EQ(is_12_mod2_0, AGENT_COUNT / 2);
    EXPECT_EQ(is_12_mod2_1, AGENT_COUNT / 2);
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentAgent_WithDeath) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    agent2.newVariable<float>("x");
    agent2.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent2.newFunction("output", OptionalOutputWithDeath);
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        EXPECT_EQ(ai.getVariable<unsigned int>("id") % 2, 0u);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
    }
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, DefaultVariableValue) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newVariable<float>("x");
    agent.newVariable<float>("default", 15.0f);
    agent.newVariable<unsigned int>("id");
    agent2.newVariable<float>("x");
    agent2.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent2.newFunction("output", MandatoryOutput);
    function.setAgentOutput(agent);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
    unsigned int is_12_mod2_1 = 0;
    unsigned int is_12_mod2_3 = 0;
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize(); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i);
        if (ai.getVariable<float>("default") == 15.0f) {
            is_15++;
            if (ai.getVariable<unsigned int>("id") % 4 == 1) {
                is_12_mod2_1++;
            } else if (ai.getVariable<unsigned int>("id") % 4 == 3) {
                is_12_mod2_3++;
            }
        }
    }
    EXPECT_EQ(is_15, AGENT_COUNT);
    EXPECT_EQ(is_12_mod2_1, AGENT_COUNT / 4);
    EXPECT_EQ(is_12_mod2_3, AGENT_COUNT / 4);
}
FLAMEGPU_AGENT_FUNCTION_CONDITION(EvenThreadsOnlyCdn) {
    return threadIdx.x % 2 == 0;
}
TEST(DeviceAgentCreationTest, Mandatory_Output_SameState_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutput);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setInitialState("a");
    function.setEndState("b");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
    }
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    unsigned int is_12_mod2_1 = 0;
    unsigned int is_12_mod2_3 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        if (ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id") == 12.0f) {
            is_12++;
            if (ai.getVariable<unsigned int>("id") % 4 == 1) {
                is_12_mod2_1++;
            } else if (ai.getVariable<unsigned int>("id") % 4 == 3) {
                is_12_mod2_3++;
            }
        } else if (ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id") == 1.0f) {
            is_1++;
            EXPECT_EQ(ai.getVariable<unsigned int>("id") % 2, 0u);
        } else {
            ASSERT_TRUE(false);  // This should never happen
        }
    }
    EXPECT_EQ(is_1, AGENT_COUNT / 2);  // Agent is from init
    EXPECT_EQ(is_12, AGENT_COUNT / 2);  // Agent is from device birth
    EXPECT_EQ(is_12_mod2_1, AGENT_COUNT / 4);
    EXPECT_EQ(is_12_mod2_3, AGENT_COUNT / 4);
}
TEST(DeviceAgentCreationTest, Optional_Output_SameState_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutput);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setInitialState("a");
    function.setEndState("b");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT / 2 + AGENT_COUNT / 4);
    unsigned int is_1 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        if (ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f) {
            is_1++;
            ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
        }
    }
    EXPECT_EQ(is_1, population.getCurrentListSize("a"));
    is_1 = 0;
    unsigned int is_12 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        if (ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id") == 12.0f) {
            is_12++;
            ASSERT_EQ(ai.getVariable<unsigned int>("id") % 4, 1u);
        } else if (ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id") == 1.0f) {
            is_1++;
            ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 0u);
        }
    }
    EXPECT_EQ(is_12, AGENT_COUNT / 4);
    EXPECT_EQ(is_1, AGENT_COUNT / 2);
}
TEST(DeviceAgentCreationTest, Mandatory_Output_DifferentState_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newState("c");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutput);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setInitialState("a");
    function.setEndState("c");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("c"), AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
    }
    unsigned int is_12_mod2_1 = 0;
    unsigned int is_12_mod2_3 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
        if (ai.getVariable<unsigned int>("id") % 4 == 1) {
            is_12_mod2_1++;
        } else if (ai.getVariable<unsigned int>("id") % 4 == 3) {
            is_12_mod2_3++;
        }
    }
    EXPECT_EQ(is_12_mod2_1, AGENT_COUNT / 4);
    EXPECT_EQ(is_12_mod2_3, AGENT_COUNT / 4);
    for (unsigned int i = 0; i < population.getCurrentListSize("c"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "c");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 0u);
    }
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentState_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newState("c");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutput);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setInitialState("a");
    function.setEndState("c");
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT / 2);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT / 4);
    EXPECT_EQ(population.getCurrentListSize("c"), AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 4, 1u);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("c"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "c");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 0u);
    }
}
TEST(DeviceAgentCreationTest, Mandatory_Output_SameState_WithDeath_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutputWithDeath);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setAgentOutput(agent);
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    // 50% original agents output new agent and died, 50% original agents lived on disabled
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    unsigned int is_12_mod2_1 = 0;
    unsigned int is_12_mod2_3 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id");
        if (val == 1.0f) {
            is_1++;
            ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
        } else if (val == 12.0f) {
            is_12++;
            if (ai.getVariable<unsigned int>("id") % 4 == 1) {
                is_12_mod2_1++;
            } else if (ai.getVariable<unsigned int>("id") % 4 == 3) {
                is_12_mod2_3++;
            }
        }
    }
    EXPECT_EQ(is_1, AGENT_COUNT / 2);
    EXPECT_EQ(is_12, AGENT_COUNT / 2);
    EXPECT_EQ(is_12_mod2_1, AGENT_COUNT / 4);
    EXPECT_EQ(is_12_mod2_3, AGENT_COUNT / 4);
}
TEST(DeviceAgentCreationTest, Optional_Output_SameState_WithDeath_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutputWithDeath);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setAgentOutput(agent);
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    // 50 % original agents did not execute so lived on = AGENT_COUNT / 2
    // 25 % original agents executed, output new agent and died
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    unsigned int is_1 = 0;
    unsigned int is_12 = 0;
    unsigned int is_1_mod2_0 = 0;
    unsigned int is_1_mod2_1 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        float val = ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id");
        if (val == 1.0f) {
            is_1++;
            if (ai.getVariable<unsigned int>("id") % 2 == 0) {
                is_1_mod2_0++;
            } else {
                is_1_mod2_1++;
            }
        } else if (val == 12.0f) {
            is_12++;
            ASSERT_EQ(ai.getVariable<unsigned int>("id") % 4, 1u);
        } else {
            printf("i:%u, x:%f, id:%u\n", i, ai.getVariable<float>("x"), ai.getVariable<unsigned int>("id"));
        }
    }
    EXPECT_EQ(is_1, AGENT_COUNT / 2 + AGENT_COUNT / 4);
    EXPECT_EQ(is_1_mod2_0, AGENT_COUNT / 4);
    EXPECT_EQ(is_1_mod2_1, AGENT_COUNT / 2);
    EXPECT_EQ(is_12, AGENT_COUNT / 4);
}
TEST(DeviceAgentCreationTest, Mandatory_Output_DifferentState_WithDeath_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", MandatoryOutputWithDeath);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
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
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
    }
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        ASSERT_EQ(ai.getVariable<unsigned int>("id") % 2, 1u);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
    }
}
TEST(DeviceAgentCreationTest, Optional_Output_DifferentState_WithDeath_WithAgentFunctionCondition) {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent.newFunction("output", OptionalOutputWithDeath);
    function.setFunctionCondition(EvenThreadsOnlyCdn);
    function.setInitialState("a");
    function.setEndState("a");
    function.setAgentOutput(agent, "b");
    function.setAllowAgentDeath(true);
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDASimulation cuda_model(model);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance("a");
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize("a"), AGENT_COUNT / 2 + AGENT_COUNT / 4);
    EXPECT_EQ(population.getCurrentListSize("b"), AGENT_COUNT / 4);
    unsigned int is_1_mod2_0 = 0;
    unsigned int is_1_mod2_1 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize("a"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "a");
        if (ai.getVariable<unsigned int>("id") % 2 == 0) {
            is_1_mod2_0++;
        } else {
            is_1_mod2_1++;
        }
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        if (ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id") != 1.0f)
            printf("i:%u, x:%f, id:%u\n", i, ai.getVariable<float>("x"), ai.getVariable<unsigned int>("id"));
    }
    EXPECT_EQ(is_1_mod2_0, AGENT_COUNT / 4);
    EXPECT_EQ(is_1_mod2_1, AGENT_COUNT / 2);
    for (unsigned int i = 0; i < population.getCurrentListSize("b"); ++i) {
        AgentInstance ai = population.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<unsigned int>("id") % 4, 1u);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
    }
}

// Agent arrays
const unsigned int AGENT_COUNT = 1024;
FLAMEGPU_AGENT_FUNCTION(ArrayVarDeviceBirth, MsgNone, MsgNone) {
    unsigned int i = FLAMEGPU->getVariable<unsigned int>("id") * 3;
    FLAMEGPU->agent_out.setVariable<unsigned int>("id", i);
    FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 0, 3 + i);
    FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 1, 5 + i);
    FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 2, 9 + i);
    FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 3, 17 + i);
    FLAMEGPU->agent_out.setVariable<float>("y", 14.0f + i);
    return DEAD;
}
FLAMEGPU_AGENT_FUNCTION(ArrayVarDeviceBirth_DefaultWorks, MsgNone, MsgNone) {
    unsigned int i = FLAMEGPU->getVariable<unsigned int>("id") * 3;
    FLAMEGPU->agent_out.setVariable<unsigned int>("id", i);
    return DEAD;
}
FLAMEGPU_AGENT_FUNCTION(ArrayVarDeviceBirth_ArrayUnsuitable, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<int>("array_var", 0);
    return DEAD;
}

TEST(DeviceAgentCreationTest, DeviceAgentBirth_ArraySet) {
    const std::array<int, 4> TEST_REFERENCE = { 3, 5, 9, 17 };
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<unsigned int>("id", UINT_MAX);
    agent.newVariable<int, 4>("array_var");
    agent.newVariable<float>("y");
    auto &fn = agent.newFunction("out", ArrayVarDeviceBirth);
    fn.setAllowAgentDeath(true);
    fn.setAgentOutput(agent);
    model.newLayer().addAgentFunction(fn);
    // Run the init function
    AgentPopulation population(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        auto in = population.getNextInstance();
        in.setVariable<unsigned int>("id", i);
    }
    CUDASimulation sim(model);
    sim.setPopulationData(population);
    sim.step();
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        const unsigned int j = instance.getVariable<unsigned int>("id");
        // Check array sets are correct
        auto array1 = instance.getVariable<int, 4>("array_var");
        for (unsigned int k = 0; k < 4; ++k) {
            array1[k] -= j;
        }
        EXPECT_EQ(j % 3, 0u);
        EXPECT_EQ(array1, TEST_REFERENCE);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f + j);
    }
}
TEST(DeviceAgentCreationTest, DeviceAgentBirth_DefaultWorks) {
    const std::array<int, 4> TEST_REFERENCE = { 3, 5, 9, 17 };
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent_name");
    agent.newVariable<unsigned int>("id", UINT_MAX);
    agent.newVariable<int, 4>("array_var", TEST_REFERENCE);
    agent.newVariable<float>("y", 14.0f);
    auto &fn = agent.newFunction("out", ArrayVarDeviceBirth_DefaultWorks);
    fn.setAllowAgentDeath(true);
    fn.setAgentOutput(agent);
    model.newLayer().addAgentFunction(fn);
    // Run the init function
    AgentPopulation population(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        auto in = population.getNextInstance();
        in.setVariable<unsigned int>("id", i);
    }
    CUDASimulation sim(model);
    sim.setPopulationData(population);
    sim.step();
    sim.getPopulationData(population);
    // Check data is correct
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    for (unsigned int i = 0; i < population.getCurrentListSize(); i++) {
        AgentInstance instance = population.getInstanceAt(i);
        const unsigned int j = instance.getVariable<unsigned int>("id");
        // Check array sets are correct
        auto array1 = instance.getVariable<int, 4>("array_var");
        EXPECT_EQ(j % 3, 0u);
        EXPECT_EQ(array1, TEST_REFERENCE);
        EXPECT_EQ(instance.getVariable<float>("y"), 14.0f);
    }
}
}  // namespace test_device_agent_creation
