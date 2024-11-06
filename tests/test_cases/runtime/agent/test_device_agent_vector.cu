#include <string>
#include <set>
#include <vector>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace DeviceAgentVectorTest {
    const unsigned int AGENT_COUNT = 10;
    const std::string MODEL_NAME = "model";
    const std::string SUBMODEL_NAME = "submodel";
    const std::string AGENT_NAME = "agent";

FLAMEGPU_STEP_FUNCTION(SetGet) {
    // Accessing DeviceAgentVector like this would previously lead to an access violation (Issue #522, PR #751)
    DeviceAgentVector av = FLAMEGPU->agent(AGENT_NAME).getPopulationData();
    for (AgentVector::Agent ai : av) {
        ai.setVariable<int>("int", ai.getVariable<int>("int") + 12);
    }
}
FLAMEGPU_STEP_FUNCTION(SetGetHalf) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    for (unsigned int i = av.size()/4; i < av.size() - av.size()/4; ++i) {
        av[i].setVariable<int>("int", av[i].getVariable<int>("int") + 12);
    }
    // agent.setPopulationData(av);
}

FLAMEGPU_STEP_FUNCTION(GetIndex) {
    flamegpu::HostAgentAPI agents = FLAMEGPU->agent(AGENT_NAME);
    // Get DeviceAgentVector to the population
    flamegpu::DeviceAgentVector agent_vector = agents.getPopulationData();
    // check all index values
    unsigned int counter = 0;
    for (auto a : agent_vector) {
        ASSERT_EQ(a.getIndex(), counter);
        counter++;
    }
}
TEST(DeviceAgentVectorTest, SetGet) {
    // Initialise an agent population with values in a variable [0,1,2..N]
    // Inside a step function, retrieve the agent population as a DeviceAgentVector
    // Update all agents by adding 12 to their value
    // After model completion, retrieve the agent population and check their values are [12,13,14..N+12]
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(SetGet);

    // Init agent pop
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i)
      av[i].setVariable<int>("int", static_cast<int>(i));

    // Create and step simulation
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 12);
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 24);
    }
}
TEST(DeviceAgentVectorTest, SetGetHalf) {
    // Initialise an agent population with values in a variable [0,1,2..N]
    // Inside a step function, retrieve the agent population as a DeviceAgentVector
    // Update half agents (contiguous block) by adding 12 to their value
    // After model completion, retrieve the agent population and check their values are [12,13,14..N+12]
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(SetGetHalf);

    // Init agent pop
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i)
        av[i].setVariable<int>("int", static_cast<int>(i));

    // Create and step simulation
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        if (i < AGENT_COUNT/4 || i >= AGENT_COUNT - AGENT_COUNT/4) {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        } else  {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 12);
        }
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        if (i < AGENT_COUNT / 4 || i >= AGENT_COUNT - AGENT_COUNT / 4) {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
        } else {
            ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i) + 24);
        }
    }
}
TEST(DeviceAgentVectorTest, GetIndex) {
    // Initialise an agent population with values in a variable [0,1,2..N]
    // Inside a step function, iterate the device agent vector
    // Assert that agent index matches the order in the vector.
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(GetIndex);

    // Init agent pop
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i)
        av[i].setVariable<int>("int", static_cast<int>(i));

    // Create and step simulation
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();  // agent step function involved
}
FLAMEGPU_AGENT_FUNCTION(MasterIncrement, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("uint", FLAMEGPU->getVariable<unsigned int>("uint") + 1);
    return ALIVE;
}

FLAMEGPU_STEP_FUNCTION(Resize) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    unsigned int av_size = av.size();
    av.resize(av_size + AGENT_COUNT);
    // Continue the existing variable pattern
    for (unsigned int i = av_size; i < av_size + AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", i);
    }
}
TEST(DeviceAgentVectorTest, Resize) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector works as expected (when CUDAAgent resizes)
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 0);
    model.addStepFunction(Resize);

    // Init agent pop
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i)
        av[i].setVariable<int>("int", static_cast<int>(i));

    // Create and step simulation
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 2);
    for (unsigned int i = 0; i < AGENT_COUNT * 2; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
    }

    // Step again
    sim.step();

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), AGENT_COUNT * 3);
    for (unsigned int i = 0; i < AGENT_COUNT * 3; ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(i));
    }
}
FLAMEGPU_STEP_FUNCTION(Insert) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    AgentInstance ai(av[0]);
    av.insert(av.size() - AGENT_COUNT/2, AGENT_COUNT, ai);
}
FLAMEGPU_STEP_FUNCTION(Erase) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    av.erase(AGENT_COUNT / 4, AGENT_COUNT / 2);
    av.push_back();
    av.back().setVariable<int>("int", -2);
}
FLAMEGPU_EXIT_CONDITION(AlwaysExit) {
    return EXIT;
}
TEST(DeviceAgentVectorTest, SubmodelResize) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector works as expected
    // Specifically, that when the agent population is resized, unbound variabled in the master-model are default init
    ModelDescription sub_model(SUBMODEL_NAME);
    AgentDescription sub_agent = sub_model.newAgent(AGENT_NAME);
    sub_agent.newVariable<int>("int", 0);
    sub_model.addStepFunction(Resize);
    sub_model.addExitCondition(AlwaysExit);


    ModelDescription master_model(MODEL_NAME);
    AgentDescription master_agent = master_model.newAgent(AGENT_NAME);
    master_agent.newVariable<int>("int", 0);
    master_agent.newVariable<unsigned int>("uint", 12u);
    master_agent.newFunction("MasterIncrement", MasterIncrement);
    SubModelDescription sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
    sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, true);
    master_model.newLayer().addAgentFunction(MasterIncrement);
    master_model.newLayer().addSubModel(sub_desc);

    // Init agent pop
    AgentVector av(master_agent, AGENT_COUNT);
    std::vector<int> vec_int;
    std::vector<unsigned int> vec_uint;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
        av[i].setVariable<unsigned int>("uint", static_cast<int>(i));
        vec_int.push_back(i);
        vec_uint.push_back(i);
    }

    // Create and step simulation
    CUDASimulation sim(master_model);
    sim.setPopulationData(av);
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.resize(vec_int.size() + AGENT_COUNT, 0);
    vec_uint.resize(vec_uint.size() + AGENT_COUNT, 12u);
    for (unsigned int i = AGENT_COUNT; i < 2 * AGENT_COUNT; ++i)
      vec_int[i] = static_cast<int>(i);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
    }

    // Step again
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.resize(vec_int.size() + AGENT_COUNT, 0);
    vec_uint.resize(vec_uint.size() + AGENT_COUNT, 12u);
    for (unsigned int i = 2 * AGENT_COUNT; i < 3 *AGENT_COUNT; ++i)
        vec_int[i] = static_cast<int>(i);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
    }
}
TEST(DeviceAgentVectorTest, SubmodelInsert) {
    // In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
    // The minimum buffer len is 1024 and resize grows by 25%
    // So to trigger resize, we can grow from 1024->2048

    // The intention of this test is to check that agent birth via DeviceAgentVector::insert works as expected
    // Specifically, that when the agent population is resized, unbound variabled in the master-model are default init
    ModelDescription sub_model(SUBMODEL_NAME);
    AgentDescription sub_agent = sub_model.newAgent(AGENT_NAME);
    sub_agent.newVariable<int>("int", 0);
    sub_model.addStepFunction(Insert);
    sub_model.addExitCondition(AlwaysExit);


    ModelDescription master_model(MODEL_NAME);
    AgentDescription master_agent = master_model.newAgent(AGENT_NAME);
    master_agent.newVariable<int>("int", 0);
    master_agent.newVariable<unsigned int>("uint", 12u);
    master_agent.newFunction("MasterIncrement", MasterIncrement);
    SubModelDescription sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
    sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, true);
    master_model.newLayer().addAgentFunction(MasterIncrement);
    master_model.newLayer().addSubModel(sub_desc);

    // Init agent pop
    AgentVector av(master_agent, AGENT_COUNT);
    std::vector<int> vec_int;
    std::vector<unsigned int> vec_uint;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
        av[i].setVariable<unsigned int>("uint", static_cast<int>(i));
        vec_int.push_back(i);
        vec_uint.push_back(i);
    }

    // Create and step simulation
    CUDASimulation sim(master_model);
    sim.setPopulationData(av);
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.insert(vec_int.begin() + (vec_int.size() - AGENT_COUNT / 2), AGENT_COUNT, vec_int[0]);
    vec_uint.insert(vec_uint.begin() + (vec_uint.size() - AGENT_COUNT / 2), AGENT_COUNT, 12u);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
    }

    // Step again
    sim.step();
    // Update vectors to match
    for (unsigned int i = 0; i < vec_uint.size(); ++i)
        vec_uint[i]++;
    vec_int.insert(vec_int.begin() + (vec_int.size() - AGENT_COUNT / 2), AGENT_COUNT, vec_int[0]);
    vec_uint.insert(vec_uint.begin() + (vec_uint.size() - AGENT_COUNT / 2), AGENT_COUNT, 12u);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < av.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<unsigned int>("uint"), vec_uint[i]);
    }
}
TEST(DeviceAgentVectorTest, SubmodelErase) {
    // The intention of this test is to check that agent death via DeviceAgentVector::erase works as expected
    ModelDescription sub_model(SUBMODEL_NAME);
    AgentDescription sub_agent = sub_model.newAgent(AGENT_NAME);
    sub_agent.newVariable<int>("int", 0);
    sub_model.addStepFunction(Erase);
    sub_model.addExitCondition(AlwaysExit);


    ModelDescription master_model(MODEL_NAME);
    AgentDescription master_agent = master_model.newAgent(AGENT_NAME);
    master_agent.newVariable<int>("int", -1);
    master_agent.newVariable<float>("float", 12.0f);
    SubModelDescription sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
    sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, true);
    master_model.newLayer().addSubModel(sub_desc);

    // Init agent pop, and test vectors
    AgentVector av(master_agent, AGENT_COUNT);
    std::vector<int> vec_int;
    std::vector<float> vec_flt;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
        vec_int.push_back(static_cast<int>(i));
        vec_flt.push_back(12.0f);
    }
    // Create and step simulation
    CUDASimulation sim(master_model);
    sim.setPopulationData(av);
    sim.step();
    // Update vectors to match
    vec_int.erase(vec_int.begin() + (AGENT_COUNT / 4), vec_int.begin() + (AGENT_COUNT / 2));
    vec_flt.erase(vec_flt.begin() + (AGENT_COUNT /4), vec_flt.begin() + (AGENT_COUNT / 2));
    vec_int.push_back(-2);
    vec_flt.push_back(12.0f);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < vec_int.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<float>("float"), vec_flt[i]);
    }

    // Step again
    sim.step();
    // Update vectors to match
    vec_int.erase(vec_int.begin() + (AGENT_COUNT / 4), vec_int.begin() + (AGENT_COUNT / 2));
    vec_flt.erase(vec_flt.begin() + (AGENT_COUNT / 4), vec_flt.begin() + (AGENT_COUNT / 2));
    vec_int.push_back(-2);
    vec_flt.push_back(12.0f);

    // Retrieve and validate agents match
    sim.getPopulationData(av);
    ASSERT_EQ(av.size(), vec_int.size());
    for (unsigned int i = 0; i < vec_int.size(); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), vec_int[i]);
        ASSERT_EQ(av[i].getVariable<float>("float"), vec_flt[i]);
    }
}
FLAMEGPU_CUSTOM_REDUCTION(bespoke_sum, a, b) {
    return a + b;
}
FLAMEGPU_CUSTOM_TRANSFORM(bespoke_triple, a) {
    return a * 3;
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_count) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(agent.count(), AGENT_COUNT);
    // Update the agent pop (erase first 4 items)
    DeviceAgentVector av = agent.getPopulationData();
    av.erase(0, 4);
    // Test reduce again (need to test each with it's own update)
    ASSERT_EQ(agent.count(), AGENT_COUNT - 4);
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_sum) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    int sum_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        sum_result += i;
    }
    ASSERT_EQ(agent.sum<int>("int"), sum_result);
    // Update the agent pop (erase first 4 items)
    DeviceAgentVector av = agent.getPopulationData();
    sum_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        int j = av[i].getVariable<int>("int");
        j *= 2;
        sum_result += j;
        av[i].setVariable<int>("int", j);
    }
    // Test reduce again (need to test each with it's own update)
    ASSERT_EQ(agent.sum<int>("int"), sum_result);
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_min) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    int min_result = INT_MAX;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        min_result = min_result < i ? min_result : i;
    }
    ASSERT_EQ(agent.min<int>("int"), min_result);
    // Update the agent pop (erase first 4 items)
    DeviceAgentVector av = agent.getPopulationData();
    min_result = INT_MAX;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        int j = av[i].getVariable<int>("int");
        j *= 2;
        min_result = min_result < j ? min_result : j;
        av[i].setVariable<int>("int", j);
    }
    // Test reduce again (need to test each with it's own update)
    ASSERT_EQ(agent.min<int>("int"), min_result);
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_max) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    int max_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        max_result = max_result > i ? max_result : i;
    }
    ASSERT_EQ(agent.max<int>("int"), max_result);
    // Update the agent pop (erase first 4 items)
    DeviceAgentVector av = agent.getPopulationData();
    max_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        int j = av[i].getVariable<int>("int");
        j *= 2;
        max_result = max_result > j ? max_result : j;
        av[i].setVariable<int>("int", j);
    }
    // Test reduce again (need to test each with it's own update)
    ASSERT_EQ(agent.max<int>("int"), max_result);
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_countif) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    unsigned int count_if_result = 0;
    const int count_if_test = 3;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        count_if_result += i == count_if_test ? 1 : 0;
    }
    ASSERT_EQ(agent.count<int>("int", count_if_test), count_if_result);
    // Update the agent pop (erase first 4 items)
    DeviceAgentVector av = agent.getPopulationData();
    count_if_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        int j = av[i].getVariable<int>("int");
        j *= 2;
        count_if_result += j == count_if_test ? 1 : 0;
        av[i].setVariable<int>("int", j);
    }
    // Test reduce again (need to test each with it's own update)
    ASSERT_EQ(agent.count<int>("int", count_if_test), count_if_result);
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_histogrameven) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    const int histogram_even_bins = 5;
    std::array<unsigned int, histogram_even_bins> histogram_even_result = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        histogram_even_result[i / (AGENT_COUNT / 5)] += 1;
    }
    std::vector<unsigned int> histogram_even_test = agent.histogramEven<int>("int", histogram_even_bins, 0, AGENT_COUNT);
    for (size_t i = 0; i < histogram_even_result.size(); ++i) {
        ASSERT_EQ(histogram_even_test[i], histogram_even_result[i]);
    }
    // Update the agent pop
    DeviceAgentVector av = agent.getPopulationData();
    histogram_even_result = { 0, 0, 0, 0, 0 };
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        int j = av[i].getVariable<int>("int");
        j /= 2;
        histogram_even_result[j / (AGENT_COUNT / 5)] += 1;
        av[i].setVariable<int>("int", j);
    }
    // Test reduce again (need to test each with it's own update)
    histogram_even_test = agent.histogramEven<int>("int", histogram_even_bins, 0, AGENT_COUNT);
    for (size_t i = 0; i < histogram_even_result.size(); ++i) {
        ASSERT_EQ(histogram_even_test[i], histogram_even_result[i]);
    }
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_reduce) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    int sum_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        sum_result += i;
    }
    ASSERT_EQ(agent.reduce<int>("int", bespoke_sum, 0), sum_result);
    // Update the agent pop (erase first 4 items)
    DeviceAgentVector av = agent.getPopulationData();
    sum_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        int j = av[i].getVariable<int>("int");
        j *= 2;
        sum_result += j;
        av[i].setVariable<int>("int", j);
    }
    // Test reduce again (need to test each with it's own update)
    ASSERT_EQ(agent.reduce<int>("int", bespoke_sum, 0), sum_result);
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_transformreduce) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    int sum_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        sum_result += i;
    }
    ASSERT_EQ(agent.transformReduce<int>("int", bespoke_triple, bespoke_sum, 0), sum_result * 3);
    // Update the agent pop (erase first 4 items)
    DeviceAgentVector av = agent.getPopulationData();
    sum_result = 0;
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        int j = av[i].getVariable<int>("int");
        j *= 2;
        sum_result += j;
        av[i].setVariable<int>("int", j);
    }
    // Test reduce again (need to test each with it's own update)
    ASSERT_EQ(agent.transformReduce<int>("int", bespoke_triple, bespoke_sum, 0), sum_result * 3);
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_sort1) {
    // Sort1, sort first
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    // Reverse sort order
    agent.sort<int>("int", HostAgentAPI::Desc);
    // Check the device agent vector sees the result correctly
    DeviceAgentVector av = agent.getPopulationData();
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), static_cast<int>(AGENT_COUNT - 1 - i));
    }
}
FLAMEGPU_STEP_FUNCTION(HostReduceAutoSync_step_sort2) {
    // Sort2, transform via device agent vector first
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    // Reverse sort order
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        av[i].setVariable<int>("int", static_cast<int>(AGENT_COUNT - 1 - i));
    }
    // Reverse sort order again
    agent.sort<int>("int", HostAgentAPI::Asc);
    // Check the device agent vector sees the result correctly
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), i);
    }
}
// This macro acts as a test fixture for executing a test in the named step function
#define AUTOSYNCTEST(name)\
TEST(DeviceAgentVectorTest, HostReduceAutoSync_ ## name) {\
    ModelDescription model(MODEL_NAME);\
    AgentDescription agent = model.newAgent(AGENT_NAME);\
    agent.newVariable<int>("int", 10);\
    model.addStepFunction(HostReduceAutoSync_step_ ## name);\
    AgentVector av(agent, AGENT_COUNT);\
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {\
        av[i].setVariable<int>("int", static_cast<int>(i));\
    }\
    CUDASimulation sim(model);\
    sim.setPopulationData(av);\
    sim.step();\
}
AUTOSYNCTEST(count)
AUTOSYNCTEST(sum)
AUTOSYNCTEST(min)
AUTOSYNCTEST(max)
AUTOSYNCTEST(countif)
AUTOSYNCTEST(histogrameven)
AUTOSYNCTEST(reduce)
AUTOSYNCTEST(transformreduce)
AUTOSYNCTEST(sort1)
AUTOSYNCTEST(sort2)

/**
 * The following tests all test the interaction between host agent birth and DeviceAgentVector
 * All DeviceAgentVector methods are tested individually, to confirm they do apply host agent births before
 * performing their actions.
 */
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_at) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Test again (need to test each with it's own update)
    for (int i = 0; i < 4; ++i) {
        ASSERT_EQ(av.at(AGENT_COUNT + i).getVariable<int>("int"), -i);
    }
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        ASSERT_EQ(av.at(i).getVariable<int>("int"), i);
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_at) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_at);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + 4);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_front) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), 0u);
    // Host agent birth 1 agent
    agent.newAgent().setVariable<int>("int", -12);
    // Test again (need to test each with it's own update)
    ASSERT_EQ(av.front().getVariable<int>("int"), -12);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_front) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_front);
    AgentVector av(agent, AGENT_COUNT);
    CUDASimulation sim(model);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), 1u);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_back) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Test again (need to test each with it's own update)
    ASSERT_EQ(av.back().getVariable<int>("int"), -3);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_back) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_back);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + 4);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_begin) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Test again (need to test each with it's own update)
    int i = 0;
    for (auto a = av.begin(); a != av.end(); ++a) {
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ((*a).getVariable<int>("int"), i);
        } else {
            ASSERT_EQ((*a).getVariable<int>("int"), - static_cast<int>(i- AGENT_COUNT));
        }
        ++i;
    }
    // Check we iterated the expected amount
    ASSERT_EQ(static_cast<unsigned int>(i), AGENT_COUNT * 2u);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_begin) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_begin);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + AGENT_COUNT);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_begin2) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Test again (need to test each with it's own update)
    int i = 0;
    for (auto a : av) {
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ(a.getVariable<int>("int"), i);
        } else {
            ASSERT_EQ(a.getVariable<int>("int"), -static_cast<int>(i - AGENT_COUNT));
        }
        ++i;
    }
    // Check we iterated the expected amount
    ASSERT_EQ(static_cast<unsigned int>(i), AGENT_COUNT * 2u);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_begin2) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_begin2);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + AGENT_COUNT);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_cbegin) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Test again (need to test each with it's own update)
    int i = 0;
    for (auto a = av.cbegin(); a != av.cend(); ++a) {
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ((*a).getVariable<int>("int"), i);
        } else {
            ASSERT_EQ((*a).getVariable<int>("int"), -static_cast<int>(i - AGENT_COUNT));
        }
        ++i;
    }
    // Check we iterated the expected amount
    ASSERT_EQ(static_cast<unsigned int>(i), AGENT_COUNT * 2u);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_cbegin) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_cbegin);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + AGENT_COUNT);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_rbegin) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Test again (need to test each with it's own update)
    int i = AGENT_COUNT * 2;
    for (auto a = av.rbegin(); a != av.rend(); ++a) {
        --i;
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ((*a).getVariable<int>("int"), i);
        } else {
            ASSERT_EQ((*a).getVariable<int>("int"), -static_cast<int>(i - AGENT_COUNT));
        }
    }
    // Check we iterated the expected amount
    ASSERT_EQ(i, 0);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_rbegin) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_rbegin);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + AGENT_COUNT);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_crbegin) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Test again (need to test each with it's own update)
    int i = AGENT_COUNT * 2;
    for (auto a = av.crbegin(); a != av.crend(); ++a) {
        --i;
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ((*a).getVariable<int>("int"), i);
        } else {
            ASSERT_EQ((*a).getVariable<int>("int"), -static_cast<int>(i - AGENT_COUNT));
        }
    }
    // Check we iterated the expected amount
    ASSERT_EQ(i, 0);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_crbegin) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_crbegin);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + AGENT_COUNT);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_empty) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), 0u);
    ASSERT_EQ(av.empty(), true);
    // Host agent birth 1 agent
    agent.newAgent().setVariable<int>("int", -12);
    // Test again (need to test each with it's own update)
    ASSERT_EQ(av.empty(), false);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_empty) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_empty);
    AgentVector av(agent, AGENT_COUNT);
    CUDASimulation sim(model);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), 1u);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_size) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i)
      agent.newAgent();  // This creates the agent, we don't actually care about it's values at this point
    // Test again (need to test each with it's own update)
    ASSERT_EQ(av.size(), AGENT_COUNT + 4u);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_size) {
    ModelDescription model(MODEL_NAME);
        AgentDescription agent = model.newAgent(AGENT_NAME);
        agent.newVariable<int>("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_size);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_capacity) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i)
        agent.newAgent();  // This creates the agent, we don't actually care about it's values at this point
    // Force auto resize
    EXPECT_EQ(av.size(), AGENT_COUNT + 4u);  // Don't need to test this here, but lint doesn't like us ignoring the value it returns
    ASSERT_GE(av.capacity(), AGENT_COUNT + 4u);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_capacity) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_capacity);
    AgentVector av(agent, AGENT_COUNT);
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_shrink_to_fit) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Add agents until we reach a state where av.capacity() > av.size()
    unsigned int total_size = AGENT_COUNT;
    int ct = 0;
    while (av.capacity() == av.size()) {
        // Exit the test early if capacity always equals size.
        if (ct >= 10)
            return;
        for (int i = 0; i < 4; ++i) {
            agent.newAgent();  // This creates the agent, we don't actually care about it's values at this point
            ++total_size;
        }
        // Force auto resize
        EXPECT_EQ(av.size(), total_size);  // Don't need to test this here, but lint doesn't like us ignoring the value it returns
        ++ct;
    }
    ASSERT_GT(av.capacity(), av.size());
    // Add 1 more agent and shrink to fit
    agent.newAgent();  // This creates the agent, we don't actually care about it's values at this point
    ++total_size;
    av.shrink_to_fit();
    // Check capacity now equals total_count
    ASSERT_GE(av.capacity(), total_size);
    ASSERT_GE(av.size(), total_size);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_step_shrink_to_fit) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_shrink_to_fit);
    AgentVector av(agent, AGENT_COUNT);
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_clear) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i)
        agent.newAgent();  // This creates the agent, we don't actually care about it's values at this point
    av.clear();
    // Test again after clear, to ensure it doesn't miss the host agent birth'd agents
    ASSERT_EQ(av.size(), 0u);
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_clear) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_clear);
    AgentVector av(agent, AGENT_COUNT);
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_insert1) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Insert 4 agents at the end of the initial list
    av.insert(AGENT_COUNT, 4, av[1]);
    // Check the size has changed correctly
    ASSERT_EQ(av.size(), AGENT_COUNT  + 8u);
    // Test again (need to test each with it's own update)
    for (int i = 0; i < static_cast<int>(AGENT_COUNT + 8); ++i) {
        auto a = av[i];
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ(a.getVariable<int>("int"), i);
        } else if (static_cast<unsigned int>(i) < AGENT_COUNT + 4) {
            ASSERT_EQ(a.getVariable<int>("int"), 1);  // We inserted 4 copies of i
        } else  {
            ASSERT_EQ(a.getVariable<int>("int"), -static_cast<int>(i - (AGENT_COUNT + 4)));  // Host new agents
        }
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_insert1) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_insert1);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + 8);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_insert2) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Insert 4 agents at the end of the initial list
    auto it1 = av.begin();
    for (int i = 0; i < 1; ++i)
        ++it1;
    auto it5 = av.begin();
    for (int i = 0; i < 5; ++i)
        ++it5;
    av.insert(AGENT_COUNT, it1, it5);
    // Check the size has changed correctly
    ASSERT_EQ(av.size(), AGENT_COUNT + 8u);
    // Test again (need to test each with it's own update)
    for (int i = 0; i < static_cast<int>(AGENT_COUNT + 8); ++i) {
        auto a = av[i];
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ(a.getVariable<int>("int"), i);
        } else if (static_cast<unsigned int>(i) < AGENT_COUNT + 4) {
            ASSERT_EQ(a.getVariable<int>("int"), 1 + static_cast<int>(i - AGENT_COUNT));  // We inserted copies of first items [1,4]
        } else {
            ASSERT_EQ(a.getVariable<int>("int"), -static_cast<int>(i - (AGENT_COUNT + 4)));  // Host new agents
        }
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_insert2) {
    // Test the templated insert method, it doesn't use the common insert
    // This is kind of redundant, as begin() will be called before insert()
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_insert2);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + 8);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_erase) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Remove 4 agents, 2 from end of initial list, 2 from start of new list
    av.erase(AGENT_COUNT - 2, AGENT_COUNT + 2);
    // Check the size has changed correctly
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Test again (need to test each with it's own update)
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); ++i) {
        auto a = av[i];
        if (static_cast<unsigned int>(i) < AGENT_COUNT - 2) {
            ASSERT_EQ(a.getVariable<int>("int"), i);
        } else {
            ASSERT_EQ(a.getVariable<int>("int"), -static_cast<int>(i + 4 - AGENT_COUNT));  // Host new agents
        }
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_erase) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", 10);
    model.addStepFunction(HostAgentBirthAutoSync_step_erase);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_push_back) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Insert 4 agents at the end of the initial list
    av.push_back();
    // Check the size has changed correctly
    ASSERT_EQ(av.size(), AGENT_COUNT + 5u);
    // Test again (need to test each with it's own update)
    for (int i = 0; i < static_cast<int>(AGENT_COUNT + 5); ++i) {
        auto a = av[i];
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ(a.getVariable<int>("int"), i);
        } else if (static_cast<unsigned int>(i) < AGENT_COUNT + 4u) {
            ASSERT_EQ(a.getVariable<int>("int"), -static_cast<int>(i - AGENT_COUNT));  // Host new agents
        } else {
            ASSERT_EQ(a.getVariable<int>("int"), -12);  // push_back
        }
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_puck_back) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", -12);
    model.addStepFunction(HostAgentBirthAutoSync_step_push_back);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + 5u);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_pop_back) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Insert 4 agents at the end of the initial list
    av.pop_back();
    // Check the size has changed correctly
    ASSERT_EQ(av.size(), AGENT_COUNT + 3u);
    // Test again (need to test each with it's own update)
    for (int i = 0; i < static_cast<int>(AGENT_COUNT + 3); ++i) {
        auto a = av[i];
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ(a.getVariable<int>("int"), i);
        } else {
            ASSERT_EQ(a.getVariable<int>("int"), -static_cast<int>(i - AGENT_COUNT));  // Host new agents
        }
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_pop_back) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", -12);
    model.addStepFunction(HostAgentBirthAutoSync_step_pop_back);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + 3u);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_resize_up) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Insert 4 agents at the end of the initial list
    av.resize(AGENT_COUNT + 8u);
    // Check the size has changed correctly
    ASSERT_EQ(av.size(), AGENT_COUNT + 8u);
    // Test again (need to test each with it's own update)
    for (int i = 0; i < static_cast<int>(AGENT_COUNT + 8); ++i) {
        if (static_cast<unsigned int>(i) < AGENT_COUNT) {
            ASSERT_EQ(av[i].getVariable<int>("int"), i);
        } else if (static_cast<unsigned int>(i) < AGENT_COUNT + 4u) {
            ASSERT_EQ(av[i].getVariable<int>("int"), -static_cast<int>(i - AGENT_COUNT));  // Host new agents
        } else {
            ASSERT_EQ(av[i].getVariable<int>("int"), -12);  // resize added agents should be default value
        }
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_resize_up) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", -12);
    model.addStepFunction(HostAgentBirthAutoSync_step_resize_up);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT + 8u);
}
FLAMEGPU_STEP_FUNCTION(HostAgentBirthAutoSync_step_resize_down) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // It shouldn't matter whether this is called before/after
    DeviceAgentVector av = agent.getPopulationData();
    // Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    ASSERT_EQ(av.size(), AGENT_COUNT);
    // Host agent birth 4 agents
    for (int i = 0; i < 4; ++i) {
        auto a = agent.newAgent();
        a.setVariable<int>("int", -i);
    }
    // Insert 4 agents at the end of the initial list
    av.resize(AGENT_COUNT - 4u);
    // Check the size has changed correctly
    ASSERT_EQ(av.size(), AGENT_COUNT - 4u);
    // Test again (need to test each with it's own update)
    for (int i = 0; i < static_cast<int>(AGENT_COUNT - 4); ++i) {
        ASSERT_EQ(av[i].getVariable<int>("int"), i);
    }
}
TEST(DeviceAgentVectorTest, HostAgentBirthAutoSync_resize_down) {
    ModelDescription model(MODEL_NAME);
    AgentDescription agent = model.newAgent(AGENT_NAME);
    agent.newVariable<int>("int", -12);
    model.addStepFunction(HostAgentBirthAutoSync_step_resize_down);
    AgentVector av(agent, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        av[i].setVariable<int>("int", static_cast<int>(i));
    }
    CUDASimulation sim(model);
    sim.setPopulationData(av);
    sim.step();
    sim.getPopulationData(av);
    // Also confirm the agents weren't added twice
    ASSERT_EQ(av.size(), AGENT_COUNT - 4u);
}
FLAMEGPU_HOST_FUNCTION(AgentID_DeviceAgentVectorBirth) {
    auto agt_a = FLAMEGPU->agent("agent", "a");
    auto agt_b = FLAMEGPU->agent("agent", "b");
    DeviceAgentVector vec_a = agt_a.getPopulationData();
    DeviceAgentVector vec_b = agt_b.getPopulationData();
    const uint32_t birth_ct_a = vec_a.size();
    const uint32_t birth_ct_b = vec_b.size();
    vec_a.resize(birth_ct_a * 2);
    vec_b.resize(birth_ct_b * 2);
    for (uint32_t i = birth_ct_a; i < 2 * birth_ct_a; ++i) {
        auto t = vec_a[i];
        t.setVariable<id_t>("id_copy", t.getID());
    }
    for (uint32_t i = birth_ct_b; i < 2 * birth_ct_b; ++i) {
        auto t = vec_b[i];
        t.setVariable<id_t>("id_copy", t.getID());
    }
}
TEST(DeviceAgentVectorTest, AgentID_MultipleStatesUniqueIDs) {
    const uint32_t POP_SIZE = 100;
    // Create agents via AgentVector to two agent states
    // DeviceAgentVector Birth creates new agent in both states (at the end of the current agents)
    // Store agent IDs to an agent variable inside model
    // Export agents and check their IDs are unique
    // Also check that the id's copied during model match those at export

    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_copy", ID_NOT_SET);
    agent.newState("a");
    agent.newState("b");

    auto layer_a = model.newLayer();
    layer_a.addHostFunction(AgentID_DeviceAgentVectorBirth);

    AgentVector pop_in(agent, POP_SIZE);

    CUDASimulation sim(model);
    sim.setPopulationData(pop_in, "a");
    sim.setPopulationData(pop_in, "b");

    sim.step();

    AgentVector pop_out_a(agent);
    AgentVector pop_out_b(agent);

    sim.getPopulationData(pop_out_a, "a");
    sim.getPopulationData(pop_out_b, "b");

    std::set<id_t> ids;
    // Validate that there are no ID collisions
    for (auto a : pop_out_a) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    for (auto a : pop_out_b) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids.size(), 4 * POP_SIZE);  // No collisions
}
FLAMEGPU_HOST_FUNCTION(AgentID_DeviceAgentVectorBirthMultiAgent) {
    auto agt_a = FLAMEGPU->agent("agent");
    auto agt_b = FLAMEGPU->agent("agent2");
    DeviceAgentVector vec_a = agt_a.getPopulationData();
    DeviceAgentVector vec_b = agt_b.getPopulationData();
    const uint32_t birth_ct_a = vec_a.size();
    const uint32_t birth_ct_b = vec_b.size();
    vec_a.resize(birth_ct_a * 2);
    vec_b.resize(birth_ct_b * 2);
    for (uint32_t i = birth_ct_a; i < 2 * birth_ct_a; ++i) {
        auto t = vec_a[i];
        t.setVariable<id_t>("id_copy", t.getID());
    }
    for (uint32_t i = birth_ct_b; i < 2 * birth_ct_b; ++i) {
        auto t = vec_b[i];
        t.setVariable<id_t>("id_copy", t.getID());
    }
}
TEST(DeviceAgentVectorTest, AgentID_MultipleAgents) {
    const uint32_t POP_SIZE = 100;
    // Create agents via AgentVector to two agent types
    // DeviceAgentVector Birth creates new agent in both types
    // Store agent IDs to an agent variable inside model
    // Export agents and check their IDs are unique
    // Also check that the id's copied during model match those at export

    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_copy", ID_NOT_SET);
    AgentDescription agent2 = model.newAgent("agent2");
    agent2.newVariable<id_t>("id_copy", ID_NOT_SET);

    auto layer_a = model.newLayer();
    layer_a.addHostFunction(AgentID_DeviceAgentVectorBirthMultiAgent);

    AgentVector pop_in_a(agent, POP_SIZE);
    AgentVector pop_in_b(agent2, POP_SIZE);

    CUDASimulation sim(model);
    sim.setPopulationData(pop_in_a);
    sim.setPopulationData(pop_in_b);

    sim.step();

    AgentVector pop_out_a(agent);
    AgentVector pop_out_b(agent);

    sim.getPopulationData(pop_out_a);
    sim.getPopulationData(pop_out_b);

    std::set<id_t> ids_a, ids_b;
    // Validate that there are no ID collisions
    for (auto a : pop_out_a) {
        ids_a.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids_a.size(), 2 * POP_SIZE);  // No collisions
    for (auto a : pop_out_b) {
        ids_b.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids_b.size(), 2 * POP_SIZE);  // No collisions
}
FLAMEGPU_HOST_FUNCTION(AgentID_DeviceAgentVectorBirth2) {
    auto agt_a = FLAMEGPU->agent("agent", "a");
    auto agt_b = FLAMEGPU->agent("agent", "b");
    DeviceAgentVector vec_a = agt_a.getPopulationData();
    DeviceAgentVector vec_b = agt_b.getPopulationData();
    const uint32_t birth_ct_a = vec_a.size();
    const uint32_t birth_ct_b = vec_b.size();
    AgentInstance copy_a(vec_a[0]);
    AgentInstance copy_b(vec_a[0]);
    auto it_a = vec_a.begin();
    for (unsigned int i = 0; i < birth_ct_a / 2; ++i)
        ++it_a;
    auto it_b = vec_a.begin();
    for (unsigned int i = 0; i < birth_ct_b / 2; ++i)
        ++it_b;
    vec_a.insert(it_a, birth_ct_a, copy_a);
    vec_b.insert(it_b, birth_ct_b, copy_b);
    for (uint32_t i = birth_ct_a; i < 2 * birth_ct_a; ++i) {
        auto t = vec_a[i];
        t.setVariable<id_t>("id_copy", t.getID());
    }
    for (uint32_t i = birth_ct_b; i < 2 * birth_ct_b; ++i) {
        auto t = vec_b[i];
        t.setVariable<id_t>("id_copy", t.getID());
    }
}
TEST(DeviceAgentVectorTest, AgentID_MultipleStatesUniqueIDs2) {
    const uint32_t POP_SIZE = 100;
    // Create agents via AgentVector to two agent states
    // DeviceAgentVector Birth creates new agent in both states (in the middle of the current agents)
    // Store agent IDs to an agent variable inside model
    // Export agents and check their IDs are unique
    // Also check that the id's copied during model match those at export

    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_copy", ID_NOT_SET);
    agent.newState("a");
    agent.newState("b");

    auto layer_a = model.newLayer();
    layer_a.addHostFunction(AgentID_DeviceAgentVectorBirth2);

    AgentVector pop_in(agent, POP_SIZE);

    CUDASimulation sim(model);
    sim.setPopulationData(pop_in, "a");
    sim.setPopulationData(pop_in, "b");

    sim.step();

    AgentVector pop_out_a(agent);
    AgentVector pop_out_b(agent);

    sim.getPopulationData(pop_out_a, "a");
    sim.getPopulationData(pop_out_b, "b");

    std::set<id_t> ids;
    // Validate that there are no ID collisions
    for (auto a : pop_out_a) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    for (auto a : pop_out_b) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids.size(), 4 * POP_SIZE);  // No collisions
}
FLAMEGPU_HOST_FUNCTION(AgentID_DeviceAgentVectorBirth3) {
    auto agt_a = FLAMEGPU->agent("agent", "a");
    auto agt_b = FLAMEGPU->agent("agent", "b");
    DeviceAgentVector vec_a = agt_a.getPopulationData();
    DeviceAgentVector vec_b = agt_b.getPopulationData();
    const uint32_t birth_pos_a = vec_a.size();
    vec_a.insert(birth_pos_a / 2, vec_b.begin(), vec_b.end());
    const uint32_t birth_pos_b = vec_b.size();
    vec_b.insert(birth_pos_b / 2, vec_a.begin(), vec_a.end());
    for (auto t : vec_a) {
        t.setVariable<id_t>("id_copy", t.getID());
    }
    for (auto t : vec_b) {
        t.setVariable<id_t>("id_copy", t.getID());
    }
}
TEST(DeviceAgentVectorTest, AgentID_MultipleStatesUniqueIDs3) {
    const uint32_t POP_SIZE = 100;
    // Create agents via AgentVector to two agent states
    // DeviceAgentVector Birth creates new agent in both states (in the middle of the current agents)
    // Store agent IDs to an agent variable inside model
    // Export agents and check their IDs are unique
    // Also check that the id's copied during model match those at export

    ModelDescription model("test_agentid");
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<id_t>("id_copy", ID_NOT_SET);
    agent.newState("a");
    agent.newState("b");

    auto layer_a = model.newLayer();
    layer_a.addHostFunction(AgentID_DeviceAgentVectorBirth3);

    AgentVector pop_in(agent, POP_SIZE);

    CUDASimulation sim(model);
    sim.setPopulationData(pop_in, "a");
    sim.setPopulationData(pop_in, "b");

    sim.step();

    AgentVector pop_out_a(agent);
    AgentVector pop_out_b(agent);

    sim.getPopulationData(pop_out_a, "a");
    sim.getPopulationData(pop_out_b, "b");

    std::set<id_t> ids;
    // Validate that there are no ID collisions
    for (auto a : pop_out_a) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    for (auto a : pop_out_b) {
        ids.insert(a.getID());
        if (a.getVariable<id_t>("id_copy") != ID_NOT_SET) {
            ASSERT_EQ(a.getID(), a.getVariable<id_t>("id_copy"));  // ID is same as reported at birth
        }
    }
    ASSERT_EQ(ids.size(), 5 * POP_SIZE);  // No collisions
}

}  // namespace DeviceAgentVectorTest
}  // namespace flamegpu
