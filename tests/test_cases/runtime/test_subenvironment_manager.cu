#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_sub_environment_manager {

const int PROPERTY_READ_TEST = 12;
const std::array<int, 2> PROPERTY_READ2_TEST = {14, 150};
FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}

FLAMEGPU_AGENT_FUNCTION(DeviceAPIGetFn, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<int>("property_read", FLAMEGPU->environment.getProperty<int>("property_read"));
    FLAMEGPU->setVariable<int, 2>("property_read2", 0, FLAMEGPU->environment.getProperty<int>("property_read2", 0));
    FLAMEGPU->setVariable<int, 2>("property_read2", 1, FLAMEGPU->environment.getProperty<int>("property_read2", 1));
    return ALIVE;
}
FLAMEGPU_HOST_FUNCTION(HostAPIGetFn) {
    std::array<int, 2> property_read2 = FLAMEGPU->environment.getProperty<int, 2>("property_read2");
    EXPECT_EQ(FLAMEGPU->environment.getProperty<int>("property_read"), PROPERTY_READ_TEST);  // Value read in sub was default from master
    EXPECT_EQ(property_read2, PROPERTY_READ2_TEST);  // Value read in sub was default from master
}
FLAMEGPU_HOST_FUNCTION(HostAPISetFn) {
    FLAMEGPU->environment.setProperty<int>("property_read", PROPERTY_READ_TEST);
    FLAMEGPU->environment.setProperty<int, 2>("property_read2", PROPERTY_READ2_TEST);
}
FLAMEGPU_HOST_FUNCTION(HostAPISetIsConstFn) {
    auto array_set_fn = &HostEnvironment::setProperty<int, 2>;
    EXPECT_THROW(FLAMEGPU->environment.setProperty<int>("property_read", 0), exception::ReadOnlyEnvProperty);
    EXPECT_THROW((FLAMEGPU->environment.*array_set_fn)("property_read2", {0, 0}), exception::ReadOnlyEnvProperty);
}
TEST(SubEnvironmentManagerTest, SubDeviceAPIGetDefault) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = m2.newAgent("agent");
        a.newFunction("", DeviceAPIGetFn);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addAgentFunction(DeviceAPIGetFn);
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", PROPERTY_READ_TEST);
        m.Environment().newProperty<int, 2>("property_read2", PROPERTY_READ2_TEST);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", m2);
    sm.bindAgent("agent", "agent", true, true);
    sm.SubEnvironment(true);
    // Construct model layers
    m.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    cm.step();
    // Test result
    cm.getPopulationData(pop);
    EXPECT_EQ(pop.size(), 1u);  // Pop size is unchanged
    AgentVector::Agent ai = pop.front();
    std::array<int, 2> property_read2 = ai.getVariable<int, 2>("property_read2");
    EXPECT_EQ(ai.getVariable<int>("property_read"), PROPERTY_READ_TEST);  // Value read in sub was default from master
    EXPECT_EQ(property_read2, PROPERTY_READ2_TEST);  // Value read in sub was default from master
}
TEST(SubEnvironmentManagerTest, SubDeviceAPIGetMasterChange) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = m2.newAgent("agent");
        a.newFunction("", DeviceAPIGetFn);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addAgentFunction(DeviceAPIGetFn);
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", 0);
        m.Environment().newProperty<int, 2>("property_read2", {0, 0});
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", m2);
    sm.bindAgent("agent", "agent", true, true);
    sm.SubEnvironment(true);
    // Construct model layers
    m.newLayer().addHostFunction(HostAPISetFn);
    m.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    cm.step();
    // Test result
    cm.getPopulationData(pop);
    EXPECT_EQ(pop.size(), 1u);  // Pop size is unchanged
    AgentVector::Agent ai = pop.front();
    std::array<int, 2> property_read2 = ai.getVariable<int, 2>("property_read2");
    EXPECT_EQ(ai.getVariable<int>("property_read"), PROPERTY_READ_TEST);  // Value read in sub was default from master
    EXPECT_EQ(property_read2, PROPERTY_READ2_TEST);  // Value read in sub was default from master
}
TEST(SubEnvironmentManagerTest, SubHostAPIGetMasterChange) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        m2.newAgent("agent");
        m2.newLayer().addHostFunction(HostAPIGetFn);
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", 0);
        m.Environment().newProperty<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", m2);
    sm.SubEnvironment(true);
    // Construct model layers
    m.newLayer().addHostFunction(HostAPISetFn);
    m.newLayer().addSubModel(sm);  // HostAPIGetFn
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    // Stepping the model performs the test (the assert is in the step fn)
    cm.step();
}
TEST(SubEnvironmentManagerTest, SubHostAPISetSub) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = m2.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addHostFunction(HostAPISetFn);
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", 0);
        m.Environment().newProperty<int, 2>("property_read2", {0, 0});
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", m2);
    sm.SubEnvironment().autoMap();
    // Construct model layers
    m.newLayer().addSubModel(sm);  // HostAPISetFn
    m.newLayer().addHostFunction(HostAPIGetFn);
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    // Stepping the model performs the test (the assert is in the step fn)
    cm.step();
}
TEST(SubEnvironmentManagerTest, SubHostAPISetConstSub) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0,  true);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0},  true);
        auto &a = m2.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addHostFunction(HostAPISetIsConstFn);
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", PROPERTY_READ_TEST);
        m.Environment().newProperty<int, 2>("property_read2", PROPERTY_READ2_TEST);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", m2);
    sm.bindAgent("agent", "agent", true, true);
    sm.SubEnvironment().autoMapProperties();
    // Construct model layers
    m.newLayer().addSubModel(sm);  // HostAPISetIsConstFn
    m.newLayer().addHostFunction(HostAPIGetFn);
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    // Stepping the model performs the test (the assert is in the step fn)
    cm.step();
}
/**
 * These subsub variants use a nested model to ensure we can nest through multiple layers of sub models
 */
TEST(SubEnvironmentManagerTest, SubSubDeviceAPIGetDefault) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = m2.newAgent("agent");
        a.newFunction("", DeviceAPIGetFn);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addAgentFunction(DeviceAPIGetFn);
    }
    ModelDescription proxy("proxy");
    {
        // Define SubModel
        proxy.addExitCondition(ExitAlways);
        // These defaults won't be used
        proxy.Environment().newProperty<int>("property_read", 0);
        proxy.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = proxy.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        auto &sm = proxy.newSubModel("proxysub", m2);
        sm.SubEnvironment(true);
        sm.bindAgent("agent", "agent", true, true);
        proxy.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", PROPERTY_READ_TEST);
        m.Environment().newProperty<int, 2>("property_read2", PROPERTY_READ2_TEST);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", proxy);
    sm.bindAgent("agent", "agent", true, true);
    sm.SubEnvironment(true);
    // Construct model layers
    m.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    cm.step();
    // Test result
    cm.getPopulationData(pop);
    EXPECT_EQ(pop.size(), 1u);  // Pop size is unchanged
    AgentVector::Agent ai = pop.front();
    std::array<int, 2> property_read2 = ai.getVariable<int, 2>("property_read2");
    EXPECT_EQ(ai.getVariable<int>("property_read"), PROPERTY_READ_TEST);  // Value read in sub was default from master
    EXPECT_EQ(property_read2, PROPERTY_READ2_TEST);  // Value read in sub was default from master
}
TEST(SubEnvironmentManagerTest, SubSubDeviceAPIGetMasterChange) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = m2.newAgent("agent");
        a.newFunction("", DeviceAPIGetFn);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addAgentFunction(DeviceAPIGetFn);
    }
    ModelDescription proxy("proxy");
    {
        // Define SubModel
        proxy.addExitCondition(ExitAlways);
        // These defaults won't be used
        proxy.Environment().newProperty<int>("property_read", 0);
        proxy.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = proxy.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        auto &sm = proxy.newSubModel("proxysub", m2);
        sm.SubEnvironment(true);
        sm.bindAgent("agent", "agent", true, true);
        proxy.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", 0);
        m.Environment().newProperty<int, 2>("property_read2", {0, 0});
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", proxy);
    sm.SubEnvironment(true);
    sm.bindAgent("agent", "agent", true, true);
    // Construct model layers
    m.newLayer().addHostFunction(HostAPISetFn);
    m.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    cm.step();
    // Test result
    cm.getPopulationData(pop);
    EXPECT_EQ(pop.size(), 1u);  // Pop size is unchanged
    AgentVector::Agent ai = pop.front();
    std::array<int, 2> property_read2 = ai.getVariable<int, 2>("property_read2");
    EXPECT_EQ(ai.getVariable<int>("property_read"), PROPERTY_READ_TEST);  // Value read in sub was default from master
    EXPECT_EQ(property_read2, PROPERTY_READ2_TEST);  // Value read in sub was default from master
}
TEST(SubEnvironmentManagerTest, SubSubHostAPIGetMasterChange) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        m2.newAgent("agent");
        m2.newLayer().addHostFunction(HostAPIGetFn);
    }
    ModelDescription proxy("proxy");
    {
        // Define SubModel
        proxy.addExitCondition(ExitAlways);
        // These defaults won't be used
        proxy.Environment().newProperty<int>("property_read", 0);
        proxy.Environment().newProperty<int, 2>("property_read2", {0, 0});
        proxy.newAgent("agent");
        auto &sm = proxy.newSubModel("proxysub", m2);
        sm.SubEnvironment(true);
        proxy.newLayer().addSubModel(sm);  // HostAPIGetFn
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", 0);
        m.Environment().newProperty<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", proxy);
    sm.SubEnvironment(true);
    // Construct model layers
    m.newLayer().addHostFunction(HostAPISetFn);
    m.newLayer().addSubModel(sm);  // HostAPIGetFn
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    // Stepping the model performs the test (the assert is in the step fn)
    cm.step();
}
TEST(SubEnvironmentManagerTest, SubSubHostAPISetSub) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = m2.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addHostFunction(HostAPISetFn);
    }
    ModelDescription proxy("proxy");
    {
        // Define SubModel
        proxy.addExitCondition(ExitAlways);
        // These defaults won't be used
        proxy.Environment().newProperty<int>("property_read", 0);
        proxy.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = proxy.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        auto &sm = proxy.newSubModel("proxysub", m2);
        sm.SubEnvironment(true);
        sm.bindAgent("agent", "agent", true, true);
        proxy.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", 0);
        m.Environment().newProperty<int, 2>("property_read2", {0, 0});
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", proxy);
    sm.bindAgent("agent", "agent", true, true);
    sm.SubEnvironment(true);
    // Construct model layers
    m.newLayer().addSubModel(sm);  // HostAPISetFn
    m.newLayer().addHostFunction(HostAPIGetFn);
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    // Stepping the model performs the test (the assert is in the step fn)
    cm.step();
}
TEST(SubEnvironmentManagerTest, SubSubHostAPISetConstSub) {
    ModelDescription m2("sub");
    {
        // Define SubModel
        m2.addExitCondition(ExitAlways);
        // These defaults won't be used
        m2.Environment().newProperty<int>("property_read", 0,  true);
        m2.Environment().newProperty<int, 2>("property_read2", {0, 0},  true);
        auto &a = m2.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        m2.newLayer().addHostFunction(HostAPISetIsConstFn);
    }
    ModelDescription proxy("proxy");
    {
        // Define SubModel
        proxy.addExitCondition(ExitAlways);
        // These defaults won't be used
        proxy.Environment().newProperty<int>("property_read", 0);
        proxy.Environment().newProperty<int, 2>("property_read2", {0, 0});
        auto &a = proxy.newAgent("agent");
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
        auto &sm = proxy.newSubModel("proxysub", m2);
        sm.SubEnvironment(true);
        proxy.newLayer().addSubModel(sm);  // DeviceAPIGetFn
    }
    ModelDescription m("host");
    auto &a = m.newAgent("agent");
    {
        // Define Model
        m.Environment().newProperty<int>("property_read", PROPERTY_READ_TEST);
        m.Environment().newProperty<int, 2>("property_read2", PROPERTY_READ2_TEST);
        a.newVariable<int>("property_read", 0);
        a.newVariable<int, 2>("property_read2", {0, 0});
    }
    // Setup submodel bindings
    auto &sm = m.newSubModel("sub", proxy);
    sm.SubEnvironment(true);
    // Construct model layers
    m.newLayer().addSubModel(sm);  // HostAPISetIsConstFn
    m.newLayer().addHostFunction(HostAPIGetFn);
    // Init agent population (we only need 1 agent, default value is fine)
    AgentVector pop(a, 1);
    // Init and step model
    CUDASimulation cm(m);
    cm.setPopulationData(pop);
    // Stepping the model performs the test (the assert is in the step fn)
    cm.step();
}
}  // namespace test_sub_environment_manager
}  // namespace flamegpu
