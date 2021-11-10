#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_spatial_agent_sort {

const unsigned int AGENT_COUNT = 4;

// Function doesn't need to do anything, just needs to use spatial messaging
FLAMEGPU_AGENT_FUNCTION(dummySpatialFunc, MessageSpatial3D, MessageNone) {
    return ALIVE;
}

TEST(AutomaticSpatialAgentSort, WarningMessage) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("initial_order");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<float>("z");
    MessageSpatial3D::Description &locationMessage = model.newMessage<MessageSpatial3D>("location");
    locationMessage.setMin(-5, -5, -5);
    locationMessage.setMax(5, 5, 5);
    locationMessage.setRadius(0.2);
    AgentFunctionDescription& dummyFunc = agent.newFunction("dummySpatialFunc", dummySpatialFunc);
    dummyFunc.setMessageInput("location");
    LayerDescription& layer = model.newLayer();
    layer.addAgentFunction(dummyFunc);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        instance.setVariable<int>("initial_order", i);
        instance.setVariable<float>("x", i);
        instance.setVariable<float>("y", i);
        instance.setVariable<float>("z", i);
    }
    
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setSortAgentsEveryNSteps(1);
    cudaSimulation.determineAgentsToSort();
    cudaSimulation.setPopulationData(pop);
    
    // Intercept std::cout
    std::stringstream buffer;
    std::streambuf* prev = std::cout.rdbuf();
    std::cout.rdbuf(buffer.rdbuf());
    
    // Execute step fn
    cudaSimulation.step();
    
    // Reset cerr
    std::cout.rdbuf(prev);
    EXPECT_EQ(buffer.str(), "WARNING: Please set the INTERACTION_RADIUS, MIN_POSITION and MAX_POSITION environment properties to enable spatial sorting\n");
}

TEST(AutomaticSpatialAgentSort, SortingDisabled) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("initial_order");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<float>("z");
    EnvironmentDescription &env = model.Environment();
    env.newProperty<float>("MIN_POSITION", -5);
    env.newProperty<float>("MAX_POSITION", 5);
    env.newProperty<float>("INTERACTION_RADIUS", 0.2);
    MessageSpatial3D::Description &locationMessage = model.newMessage<MessageSpatial3D>("location");
    locationMessage.setMin(-5, -5, -5);
    locationMessage.setMax(5, 5, 5);
    locationMessage.setRadius(0.2);
    AgentFunctionDescription& dummyFunc = agent.newFunction("dummySpatialFunc", dummySpatialFunc);
    dummyFunc.setMessageInput("location");
    LayerDescription& layer = model.newLayer();
    layer.addAgentFunction(dummyFunc);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        instance.setVariable<int>("initial_order", i);
        instance.setVariable<float>("x", -i);
        instance.setVariable<float>("y", -i);
        instance.setVariable<float>("z", -i);
    }
    
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setSortAgentsEveryNSteps(0);
    cudaSimulation.determineAgentsToSort();
    cudaSimulation.setPopulationData(pop);
    
    // Execute step fn
    cudaSimulation.step();
    
    // Check results
    cudaSimulation.getPopulationData(pop);
    std::vector<int> finalOrder;
    for (AgentVector::Agent instance : pop) {
        finalOrder.push_back(instance.getVariable<int>("initial_order"));
    }
    std::vector<int> expectedResult {0, 1, 2, 3};
    EXPECT_EQ(expectedResult, finalOrder);
}

TEST(AutomaticSpatialAgentSort, PeriodicSort) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("initial_order");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<float>("z");
    EnvironmentDescription &env = model.Environment();
    env.newProperty<float>("MIN_POSITION", -5);
    env.newProperty<float>("MAX_POSITION", 5);
    env.newProperty<float>("INTERACTION_RADIUS", 0.2);
    MessageSpatial3D::Description &locationMessage = model.newMessage<MessageSpatial3D>("location");
    locationMessage.setMin(-5, -5, -5);
    locationMessage.setMax(5, 5, 5);
    locationMessage.setRadius(0.2);
    AgentFunctionDescription& dummyFunc = agent.newFunction("dummySpatialFunc", dummySpatialFunc);
    dummyFunc.setMessageInput("location");
    LayerDescription& layer = model.newLayer();
    layer.addAgentFunction(dummyFunc);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        instance.setVariable<int>("initial_order", i);
        instance.setVariable<float>("x", -i);
        instance.setVariable<float>("y", -i);
        instance.setVariable<float>("z", -i);
    }
    
    // Setup Model
    CUDASimulation cudaSimulation(model);
    // Disable sorting so sort doesn't take place on first step
    cudaSimulation.setSortAgentsEveryNSteps(0);
    cudaSimulation.determineAgentsToSort();
    cudaSimulation.setPopulationData(pop);
    
    // Execute step fn
    cudaSimulation.step();
    // Should still be unsorted after this step
    cudaSimulation.getPopulationData(pop);
    std::vector<int> midOrder;
    for (AgentVector::Agent instance : pop) {
        midOrder.push_back(instance.getVariable<int>("initial_order"));
    }
    std::vector<int> expectedMidResult {0, 1, 2, 3};
    EXPECT_EQ(expectedMidResult, midOrder);
    
    cudaSimulation.setSortAgentsEveryNSteps(1);
    cudaSimulation.step();
    cudaSimulation.step();
    cudaSimulation.step();
    
    // Check results
    cudaSimulation.getPopulationData(pop);
    std::vector<int> finalOrder;
    for (AgentVector::Agent instance : pop) {
        finalOrder.push_back(instance.getVariable<int>("initial_order"));
    }
    std::vector<int> expectedResult {3, 2, 1, 0};
    EXPECT_EQ(expectedResult, finalOrder);
}

TEST(AutomaticSpatialAgentSort, SortEveryStep) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("initial_order");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<float>("z");
    EnvironmentDescription &env = model.Environment();
    env.newProperty<float>("MIN_POSITION", -5);
    env.newProperty<float>("MAX_POSITION", 5);
    env.newProperty<float>("INTERACTION_RADIUS", 0.2);
    MessageSpatial3D::Description &locationMessage = model.newMessage<MessageSpatial3D>("location");
    locationMessage.setMin(-5, -5, -5);
    locationMessage.setMax(5, 5, 5);
    locationMessage.setRadius(0.2);
    AgentFunctionDescription& dummyFunc = agent.newFunction("dummySpatialFunc", dummySpatialFunc);
    dummyFunc.setMessageInput("location");
    LayerDescription& layer = model.newLayer();
    layer.addAgentFunction(dummyFunc);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        instance.setVariable<int>("initial_order", i);
        instance.setVariable<float>("x", -i);
        instance.setVariable<float>("y", -i);
        instance.setVariable<float>("z", -i);
    }
    
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setSortAgentsEveryNSteps(1);
    cudaSimulation.determineAgentsToSort();
    cudaSimulation.setPopulationData(pop);
    
    // Execute step fn
    cudaSimulation.step();
    
    // Check results
    cudaSimulation.getPopulationData(pop);
    std::vector<int> finalOrder;
    for (AgentVector::Agent instance : pop) {
        finalOrder.push_back(instance.getVariable<int>("initial_order"));
    }
    std::vector<int> expectedResult {3, 2, 1, 0};
    EXPECT_EQ(expectedResult, finalOrder);
}
}  // namespace test_spatial_agent_sort
}  // namespace flamegpu
