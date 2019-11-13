#ifndef TESTS_TEST_CASES_RUNTIME_TEST_HOST_REDUCTIONS_H_
#define TESTS_TEST_CASES_RUNTIME_TEST_HOST_REDUCTIONS_H_

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"

namespace {
class MiniSim {
 public:
    MiniSim() :
      model("model"),
      agent("agent"),
      population(agent, AGENT_COUNT),
      simulation(model) {
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentInstance instance = population.getNextInstance();
        }
        model.addAgent(agent);
        // Run until exit condition triggers
        simulation.setSimulationSteps(0);
    }
    void run() {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.addAgent(agent) first
        CUDAAgentModel cuda_model(model);
        // This fails as agentMap is empty
        cuda_model.setInitialPopulationData(population);
        ASSERT_NO_THROW(cuda_model.simulate(simulation));
        // The negative of this, is that cuda_model is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cuda_model.getPopulationData(population));
    }

    const unsigned int AGENT_COUNT = 5;
    ModelDescription model;
    AgentDescription agent;
    AgentPopulation population;
    Simulation simulation;
};
/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class HostReductionTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
    }

    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};
}  // namespace

#endif  // TESTS_TEST_CASES_RUNTIME_TEST_HOST_REDUCTIONS_H_
