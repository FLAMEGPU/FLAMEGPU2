/**
 * Tests of submodel functionality of class: CUDAMacroEnvironment
 * ReadTest: Test that HostMacroProperty can read data (written via agent fn's)
 * WriteTest: Test that HostMacroProperty can write data (read via agent fn's)
 * ZeroTest: Test that HostMacroProperty can zero data (read via agent fn's)
 */

#include <array>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace {
FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}
FLAMEGPU_STEP_FUNCTION(Host_Write_5) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("a") = 5;
    FLAMEGPU->environment.getMacroProperty<unsigned int>("b") = 21;
}
FLAMEGPU_HOST_FUNCTION(Host_Read_5) {
    const unsigned int result = FLAMEGPU->environment.getMacroProperty<unsigned int>("a");
    EXPECT_EQ(result, 5u);
    const unsigned int result2 = FLAMEGPU->environment.getMacroProperty<unsigned int>("b");
    EXPECT_EQ(result2, 21u);
}
FLAMEGPU_AGENT_FUNCTION(Agent_Write_5, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("a").exchange(5u);
    FLAMEGPU->environment.getMacroProperty<unsigned int>("b").exchange(21u);
    return flamegpu::ALIVE;
}
TEST(SubCUDAMacroEnvironmentTest, SubWriteHostMasterRead) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<unsigned int>("a");
        m2.Environment().newMacroProperty<unsigned int>("b");
        m2.addStepFunction(Host_Write_5);
        m2.newAgent("test");
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<unsigned int>("a");
        m.Environment().newMacroProperty<unsigned int>("b");
    }
    auto& sm = m.newSubModel("sub", m2);
    auto& senv = sm.SubEnvironment();
    senv.autoMap();
    m.newLayer().addSubModel(sm);
    m.newLayer().addHostFunction(Host_Read_5);
    AgentVector pop(m.newAgent("test"), 1);

    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 1;
    sim.setPopulationData(pop);
    EXPECT_NO_THROW(sim.simulate());
}
TEST(SubCUDAMacroEnvironmentTest, SubWriteAgentMasterRead) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<unsigned int>("a");
        m2.Environment().newMacroProperty<unsigned int>("b");
        m2.newAgent("test").newFunction("t", Agent_Write_5);
        m2.newLayer().addAgentFunction(Agent_Write_5);
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<unsigned int>("a");
        m.Environment().newMacroProperty<unsigned int>("b");
    }
    auto &agt = m.newAgent("test");
    auto& sm = m.newSubModel("sub", m2);
    auto& senv = sm.SubEnvironment();
    senv.autoMapMacroProperties();
    sm.bindAgent("test", "test");
    m.newLayer().addSubModel(sm);
    m.newLayer().addHostFunction(Host_Read_5);
    AgentVector pop(agt, 1);

    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 1;
    sim.setPopulationData(pop);
    EXPECT_NO_THROW(sim.simulate());
}
TEST(SubCUDAMacroEnvironmentTest, MasterWriteSubReadHost) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<unsigned int>("a");
        m2.Environment().newMacroProperty<unsigned int>("b");
        m2.addStepFunction(Host_Read_5);
        m2.newAgent("test");
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<unsigned int>("a");
        m.Environment().newMacroProperty<unsigned int>("b");
    }
    auto& agt = m.newAgent("test");
    auto& sm = m.newSubModel("sub", m2);
    sm.SubEnvironment(true);
    m.newLayer().addHostFunction(Host_Write_5);
    m.newLayer().addSubModel(sm);
    AgentVector pop(agt, 1);

    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 1;
    sim.setPopulationData(pop);
    EXPECT_NO_THROW(sim.simulate());
}
FLAMEGPU_AGENT_FUNCTION(Agent_Read_Write_5, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("a", FLAMEGPU->environment.getMacroProperty<unsigned int>("a").exchange(0u));
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("b").exchange(0u));
    return flamegpu::ALIVE;
}
FLAMEGPU_HOST_FUNCTION(Host_Agent_Read_5) {
    auto agt = FLAMEGPU->agent("test");
    DeviceAgentVector pop = agt.getPopulationData();
    const unsigned int result = pop[0].getVariable<unsigned int>("a");
    EXPECT_EQ(result, 5u);
    const unsigned int result2 = pop[0].getVariable<unsigned int>("b");
    EXPECT_EQ(result2, 21u);
}
TEST(SubCUDAMacroEnvironmentTest, MasterWriteSubReadAgent) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        m2.Environment().newMacroProperty<unsigned int>("a");
        m2.Environment().newMacroProperty<unsigned int>("b");
        auto &agt = m2.newAgent("test");
        agt.newVariable<unsigned int>("a");
        agt.newVariable<unsigned int>("b");
        agt.newFunction("arw", Agent_Read_Write_5);
        m2.newLayer().addAgentFunction(Agent_Read_Write_5);
        m2.addStepFunction(Host_Agent_Read_5);
    }
    ModelDescription m("host");
    {
        // Define Model
        m.Environment().newMacroProperty<unsigned int>("a");
        m.Environment().newMacroProperty<unsigned int>("b");
    }
    auto& agt = m.newAgent("test");
    auto& sm = m.newSubModel("sub", m2);
    auto& senv = sm.SubEnvironment();
    senv.autoMap();
    sm.bindAgent("test", "test");
    m.newLayer().addHostFunction(Host_Write_5);
    m.newLayer().addSubModel(sm);
    AgentVector pop(agt, 1);

    CUDASimulation sim(m);
    sim.SimulationConfig().steps = 1;
    sim.setPopulationData(pop);
    EXPECT_NO_THROW(sim.simulate());
}
}  // namespace
}  // namespace flamegpu
