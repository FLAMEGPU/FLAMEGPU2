#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_spatial_agent_sort {

const unsigned int AGENT_COUNT = 4;

// Function doesn't need to do anything, just needs to use spatial messaging
FLAMEGPU_AGENT_FUNCTION(dummySpatialFunc, MessageSpatial3D, MessageNone) {
    return ALIVE;
}

// Initialises a reverse-sorted population and checks that it remains unsorted after a step when sorting is disabled
TEST(AutomaticSpatialAgentSort, SortingDisabled) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("initial_order");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<float>("z");
    agent.setSortPeriod(0);
    MessageSpatial3D::Description &locationMessage = model.newMessage<MessageSpatial3D>("location");
    locationMessage.setMin(-5, -5, -5);
    locationMessage.setMax(5, 5, 5);
    locationMessage.setRadius(0.2);
    AgentFunctionDescription& dummyFunc = agent.newFunction("dummySpatialFunc", dummySpatialFunc);
    dummyFunc.setMessageInput("location");
    LayerDescription& layer = model.newLayer();
    layer.addAgentFunction(dummyFunc);

    // Init pop - arranged in reverse order
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

// Initialises a reverse-sorted population and checks that it is correctly sorted after one step
TEST(AutomaticSpatialAgentSort, SortEveryStep) {
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
        instance.setVariable<float>("x", -i);
        instance.setVariable<float>("y", -i);
        instance.setVariable<float>("z", -i);
    }

    // Setup Model
    CUDASimulation cudaSimulation(model);
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
