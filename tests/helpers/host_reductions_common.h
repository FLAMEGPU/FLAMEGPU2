#ifndef TESTS_HELPERS_HOST_REDUCTIONS_COMMON_H_
#define TESTS_HELPERS_HOST_REDUCTIONS_COMMON_H_

#include <array>
#include <random>
#include <numeric>
#include <algorithm>
#include <vector>

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_host_reductions {
const unsigned int TEST_LEN = 256;
extern float float_out;
extern double double_out;
extern char char_out;
extern unsigned char uchar_out;
extern uint16_t uint16_t_out;
extern int16_t int16_t_out;
extern uint32_t uint32_t_out;
extern int32_t int32_t_out;
extern uint64_t uint64_t_out;
extern int64_t int64_t_out;
extern std::vector<unsigned int> uint_vec;
extern std::vector<int> int_vec;

class MiniSim {
 public:
    MiniSim() :
        model("model"),
        agent("agent"),
        simulation(model),
        population(nullptr) {
        agent.addAgentVariable<float>("float");
        agent.addAgentVariable<double>("double");
        agent.addAgentVariable<char>("char");
        agent.addAgentVariable<unsigned char>("uchar");
        agent.addAgentVariable<uint16_t>("uint16_t");
        agent.addAgentVariable<int16_t>("int16_t");
        agent.addAgentVariable<uint32_t>("uint32_t");
        agent.addAgentVariable<int32_t>("int32_t");
        agent.addAgentVariable<uint64_t>("uint64_t");
        agent.addAgentVariable<int64_t>("int64_t");
        population = new AgentPopulation(agent, TEST_LEN);
        simulation.setSimulationSteps(1);
    }
    ~MiniSim() { delete population; }
    void run() {
        model.addAgent(agent);
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.addAgent(agent) first
        CUDAAgentModel cuda_model(model);
        // This fails as agentMap is empty
        cuda_model.setInitialPopulationData(*population);
        ASSERT_NO_THROW(cuda_model.simulate(simulation));
        // The negative of this, is that cuda_model is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cuda_model.getPopulationData(*population));
    }
    ModelDescription model;
    AgentDescription agent;
    Simulation simulation;
    AgentPopulation *population;
};
/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class HostReductionTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
        uint_vec.clear();
        int_vec.clear();
    }

    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};
}  // namespace test_host_reductions
#endif  // TESTS_HELPERS_HOST_REDUCTIONS_COMMON_H_
