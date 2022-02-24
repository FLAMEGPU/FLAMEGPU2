#ifndef TESTS_HELPERS_HOST_REDUCTIONS_COMMON_H_
#define TESTS_HELPERS_HOST_REDUCTIONS_COMMON_H_

#ifdef _MSC_VER
#pragma warning(push)
// conversion warnings inside header (e.g. int vs std::_Array_iterator<int, 256>)
#pragma warning(disable:4244)
#pragma warning(disable:4389)
#include <numeric>
#include <algorithm>
#pragma warning(pop)
#else
#include <numeric>
#include <algorithm>
#endif
#include <array>
#include <random>
#include <vector>
#include <utility>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
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
#ifdef USE_GLM
extern glm::vec3 vec3_t_out;
#endif
extern std::pair<double, double> mean_sd_out;
extern std::vector<unsigned int> uint_vec;
extern std::vector<int> int_vec;

class MiniSim {
 public:
    MiniSim() :
        model("model"),
        agent(model.newAgent("agent")),
        population(nullptr) {
        agent.newVariable<float>("float");
        agent.newVariable<double>("double");
        agent.newVariable<char>("char");
        agent.newVariable<unsigned char>("uchar");
        agent.newVariable<uint16_t>("uint16_t");
        agent.newVariable<int16_t>("int16_t");
        agent.newVariable<uint32_t>("uint32_t");
        agent.newVariable<int32_t>("int32_t");
        agent.newVariable<uint64_t>("uint64_t");
        agent.newVariable<int64_t>("int64_t");
#ifdef USE_GLM
        agent.newVariable<glm::vec3>("vec3");
#endif
        population = new AgentVector(agent, TEST_LEN);
    }
    ~MiniSim() { delete population; }
    void run() {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        CUDASimulation cudaSimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        // This fails as agentMap is empty
        cudaSimulation.setPopulationData(*population);
        ASSERT_NO_THROW(cudaSimulation.simulate());
        // The negative of this, is that cudaSimulation is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cudaSimulation.getPopulationData(*population));
    }
    ModelDescription model;
    AgentDescription &agent;
    AgentVector*population;
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
}  // namespace flamegpu
#endif  // TESTS_HELPERS_HOST_REDUCTIONS_COMMON_H_
