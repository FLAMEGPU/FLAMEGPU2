#include "flamegpu/flame_api.h"

#include "gtest/gtest.h"

namespace test_host_agent_sort {

const unsigned int AGENT_COUNT = 1024;
FLAMEGPU_STEP_FUNCTION(sort_ascending_float) {
    FLAMEGPU->agent("agent").sort<float>("float", HostAgentAPI::Asc);
}
FLAMEGPU_STEP_FUNCTION(sort_descending_float) {
    FLAMEGPU->agent("agent").sort<float>("float", HostAgentAPI::Desc);
}
FLAMEGPU_STEP_FUNCTION(sort_ascending_int) {
  FLAMEGPU->agent("agent").sort<int>("int", HostAgentAPI::Asc);
}
FLAMEGPU_STEP_FUNCTION(sort_descending_int) {
    FLAMEGPU->agent("agent").sort<int>("int", HostAgentAPI::Desc);
}

TEST(HostAgentSort, Ascending_float) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("float");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort_ascending_float);
    std::mt19937 rd(31313131);  // Fixed seed (at Pete's request)
    std::uniform_real_distribution <float> dist(1, 1000000);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const float t = dist(rd);
        instance.setVariable<float>("float", t);
        instance.setVariable<int>("spare", static_cast<int>(t+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    float prev = 1;
    for (AgentVector::Agent instance : pop) {
        const float f = instance.getVariable<float>("float");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(static_cast<int>(f+12), s);
        // Agent variables are ordered
        EXPECT_GE(f, prev);
        // Store prev
        prev = f;
    }
}
TEST(HostAgentSort, Descending_float) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("float");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort_descending_float);
    std::mt19937 rd(888);  // Fixed seed (at Pete's request)
    std::uniform_real_distribution <float> dist(1, 1000000);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const float t = dist(rd);
        instance.setVariable<float>("float", t);
        instance.setVariable<int>("spare", static_cast<int>(t+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    float prev = 1000000;
    for (AgentVector::Agent instance : pop) {
        const float f = instance.getVariable<float>("float");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(static_cast<int>(f+12), s);
        // Agent variables are ordered
        EXPECT_LE(f, prev);
        // Store prev
        prev = f;
    }
}
TEST(HostAgentSort, Ascending_int) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("int");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort_ascending_int);
    std::mt19937 rd(77777);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist(0, 1000000);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const int t = i == AGENT_COUNT/2 ? 0 : dist(rd);  // Ensure zero is output atleast once
        instance.setVariable<int>("int", t);
        instance.setVariable<int>("spare", t+12);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    int prev = 0;
    for (AgentVector::Agent instance : pop) {
        const int f = instance.getVariable<int>("int");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(s-f, 12);
        // Agent variables are ordered
        EXPECT_GE(f, prev);
        // Store prev
        prev = f;
    }
}
TEST(HostAgentSort, Descending_int) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("int");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort_descending_int);
    std::mt19937 rd(12);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist(1, 1000000);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const int t = dist(rd);
        instance.setVariable<int>("int", t);
        instance.setVariable<int>("spare", t+12);
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    int prev = 1000000;
    for (AgentVector::Agent instance : pop) {
        const int f = instance.getVariable<int>("int");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(s-f, 12);
        // Agent variables are ordered
        EXPECT_LE(f, prev);
        // Store prev
        prev = f;
    }
}

FLAMEGPU_STEP_FUNCTION(sort2x_ascending_float) {
    FLAMEGPU->agent("agent").sort<float, float>("float1", HostAgentAPI::Asc, "float2", HostAgentAPI::Asc);
}
FLAMEGPU_STEP_FUNCTION(sort2x_descending_float) {
    FLAMEGPU->agent("agent").sort<float, float>("float1", HostAgentAPI::Desc, "float2", HostAgentAPI::Desc);
}
FLAMEGPU_STEP_FUNCTION(sort2x_ascending_int) {
    FLAMEGPU->agent("agent").sort<int, int>("int1", HostAgentAPI::Asc, "int2", HostAgentAPI::Asc);
}
FLAMEGPU_STEP_FUNCTION(sort2x_descending_int) {
    FLAMEGPU->agent("agent").sort<int, int>("int1", HostAgentAPI::Desc, "int2", HostAgentAPI::Desc);
}
FLAMEGPU_STEP_FUNCTION(sort2x_ascdesc_int) {
    FLAMEGPU->agent("agent").sort<int, int>("int1", HostAgentAPI::Asc, "int2", HostAgentAPI::Desc);
}
FLAMEGPU_STEP_FUNCTION(sort2x_descasc_int) {
    FLAMEGPU->agent("agent").sort<int, int>("int1", HostAgentAPI::Desc, "int2", HostAgentAPI::Asc);
}
TEST(HostAgentSort, 2x_Ascending_float) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("float1");
    agent.newVariable<float>("float2");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort2x_ascending_float);
    std::mt19937 rd(54323);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist1(0, 9);
    std::uniform_real_distribution <float> dist2(0, 999);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const float t1 = static_cast<float>(dist1(rd)*1000);
        const float t2 = dist2(rd);
        instance.setVariable<float>("float1", t1);
        instance.setVariable<float>("float2", t2);
        instance.setVariable<int>("spare", static_cast<int>(t2+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    float prev = 1;
    for (AgentVector::Agent instance : pop) {
        const float f1 = instance.getVariable<float>("float1");
        const float f2 = instance.getVariable<float>("float2");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(static_cast<int>(f2+12), s);
        // Agent variables are ordered
        EXPECT_GE(f1 + f2, prev);
        // Store prev
        prev = f1 + f2;
    }
}
TEST(HostAgentSort, 2x_Descending_float) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float>("float1");
    agent.newVariable<float>("float2");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort2x_descending_float);
    std::mt19937 rd(5422323);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist1(0, 9);
    std::uniform_real_distribution <float> dist2(0, 999);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const float t1 = static_cast<float>(dist1(rd)*1000);
        const float t2 = dist2(rd);
        instance.setVariable<float>("float1", t1);
        instance.setVariable<float>("float2", t2);
        instance.setVariable<int>("spare", static_cast<int>(t2+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    float prev = 1000000;
    for (AgentVector::Agent instance : pop) {
        const float f1 = instance.getVariable<float>("float1");
        const float f2 = instance.getVariable<float>("float2");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(static_cast<int>(f2+12), s);
        // Agent variables are ordered
        EXPECT_LE(f1 + f2, prev);
        // Store prev
        prev = f1 + f2;
    }
}
TEST(HostAgentSort, 2x_Ascending_int) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("int1");
    agent.newVariable<int>("int2");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort2x_ascending_int);
    std::mt19937 rd(123123);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist1(0, 9);
    std::uniform_int_distribution <int> dist2(0, 999);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const int t1 = static_cast<int>(dist1(rd)*1000);
        const int t2 = dist2(rd);
        instance.setVariable<int>("int1", t1);
        instance.setVariable<int>("int2", t2);
        instance.setVariable<int>("spare", static_cast<int>(t2+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    int prev = 0;
    for (AgentVector::Agent instance : pop) {
        const int f1 = instance.getVariable<int>("int1");
        const int f2 = instance.getVariable<int>("int2");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(f2+12, s);
        // Agent variables are ordered
        EXPECT_GE(f1 + f2, prev);
        // Store prev
        prev = f1 + f2;
    }
}
TEST(HostAgentSort, 2x_Descending_int) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("int1");
    agent.newVariable<int>("int2");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort2x_descending_int);
    std::mt19937 rd(12333123);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist1(0, 9);
    std::uniform_int_distribution <int> dist2(0, 999);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const int t1 = static_cast<int>(dist1(rd)*1000);
        const int t2 = dist2(rd);
        instance.setVariable<int>("int1", t1);
        instance.setVariable<int>("int2", t2);
        instance.setVariable<int>("spare", static_cast<int>(t2+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    int prev = 1000000;
    for (AgentVector::Agent instance : pop) {
        const int f1 = instance.getVariable<int>("int1");
        const int f2 = instance.getVariable<int>("int2");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(f2+12, s);
        // Agent variables are ordered
        EXPECT_LE(f1 + f2, prev);
        // Store prev
        prev = f1 + f2;
    }
}
TEST(HostAgentSort, 2x_AscDesc_int) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("int1");
    agent.newVariable<int>("int2");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort2x_ascdesc_int);
    std::mt19937 rd(123123);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist1(0, 9);
    std::uniform_int_distribution <int> dist2(0, 999);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const int t1 = static_cast<int>(dist1(rd)*1000);
        const int t2 = dist2(rd);
        instance.setVariable<int>("int1", t1);
        instance.setVariable<int>("int2", t2);
        instance.setVariable<int>("spare", static_cast<int>(t2+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    int prev = 0;
    for (AgentVector::Agent instance : pop) {
        const int f1 = instance.getVariable<int>("int1");
        const int f2 = instance.getVariable<int>("int2");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(f2+12, s);
        // Agent variables are ordered
        EXPECT_GE(f1 + (999-f2), prev);
        // Store prev
        prev = f1 + (999-f2);
    }
}
TEST(HostAgentSort, 2x_DescAsc_int) {
    // Define model
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("int1");
    agent.newVariable<int>("int2");
    agent.newVariable<int>("spare");
    model.newLayer().addHostFunction(sort2x_descasc_int);
    std::mt19937 rd(12333123);  // Fixed seed (at Pete's request)
    std::uniform_int_distribution <int> dist1(0, 9);
    std::uniform_int_distribution <int> dist2(0, 999);

    // Init pop
    AgentVector pop(agent, AGENT_COUNT);
    for (int i = 0; i< static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = pop[i];
        const int t1 = static_cast<int>(dist1(rd)*1000);
        const int t2 = dist2(rd);
        instance.setVariable<int>("int1", t1);
        instance.setVariable<int>("int2", t2);
        instance.setVariable<int>("spare", static_cast<int>(t2+12));
    }
    // Setup Model
    CUDASimulation cuda_model(model);
    cuda_model.setPopulationData(pop);
    // Execute step fn
    cuda_model.step();
    // Check results
    cuda_model.getPopulationData(pop);
    EXPECT_EQ(AGENT_COUNT, pop.size());
    int prev = 1000000;
    for (AgentVector::Agent instance : pop) {
        const int f1 = instance.getVariable<int>("int1");
        const int f2 = instance.getVariable<int>("int2");
        const int s = instance.getVariable<int>("spare");
        // Agent variables are still aligned
        EXPECT_EQ(f2+12, s);
        // Agent variables are ordered
        EXPECT_LE(f1 + (999-f2), prev);
        // Store prev
        prev = f1 + (999-f2);
    }
}
}  // namespace test_host_agent_sort
