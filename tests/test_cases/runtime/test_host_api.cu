#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

namespace test_host_api {

// Test host_api::getStepCounter()

FLAMEGPU_AGENT_FUNCTION(agent_function_getStepCounter, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<unsigned int>("step", FLAMEGPU->getStepCounter());
    return ALIVE;
}

const unsigned int TOTAL_STEPS = 4;

// globally scoped variable to track the value in the test?
unsigned int expectedStepCounter = 0;

// Init should always be 0th iteration/step
FLAMEGPU_INIT_FUNCTION(init_testGetStepCounter) {
    EXPECT_EQ(FLAMEGPU->getStepCounter(), 0u);
}
// host is during, so 0? - @todo dynamic
FLAMEGPU_HOST_FUNCTION(host_testGetStepCounter) {
    EXPECT_EQ(FLAMEGPU->getStepCounter(), expectedStepCounter);
}
// Step functions are at the end of the step
FLAMEGPU_STEP_FUNCTION(step_testGetStepCounter) {
    EXPECT_EQ(FLAMEGPU->getStepCounter(), expectedStepCounter);
}

// Runs between steps - i.e. after step functions
FLAMEGPU_EXIT_CONDITION(exitCondition_testGetStepCounter) {
    EXPECT_EQ(FLAMEGPU->getStepCounter(), expectedStepCounter);
    // Increment the counter used for testing multiple steps.
    expectedStepCounter++;
    return CONTINUE;
}

// exit is after all iterations, so stepCounter is the total number executed.
FLAMEGPU_EXIT_FUNCTION(exit_testGetStepCounter) {
    EXPECT_EQ(FLAMEGPU->getStepCounter(), TOTAL_STEPS);
}

TEST(hostAPITest, getStepCounter) {
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");

    model.addInitFunction(init_testGetStepCounter);
    model.newLayer().addHostFunction(host_testGetStepCounter);
    model.addStepFunction(step_testGetStepCounter);
    model.addExitCondition(exitCondition_testGetStepCounter);
    model.addExitFunction(exit_testGetStepCounter);

    // Init pop
    const unsigned int agentCount = 1;
    AgentPopulation init_population(agent, agentCount);
    for (int i = 0; i< static_cast<int>(agentCount); i++) {
        AgentInstance instance = init_population.getNextInstance("default");
    }
    // Setup Model
    CUDAAgentModel cuda_model(model);
    cuda_model.setPopulationData(init_population);

    cuda_model.SimulationConfig().steps = TOTAL_STEPS;
    cuda_model.simulate();
}

}  // namespace test_host_api
