#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace test_layer {
FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn2, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn3, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn4, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn5, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_HOST_FUNCTION(host_fn) {
    // do nothing
}
FLAMEGPU_EXIT_CONDITION(exit_cdn) {
    return EXIT;
}
const char *MODEL_NAME = "Model";
const char *AGENT_NAME = "Agent1";
const char *LAYER_NAME = "Layer1";
const char *FUNCTION_NAME1 = "Function1";
const char *FUNCTION_NAME2 = "Function2";
const char *FUNCTION_NAME3 = "Function3";
const char *FUNCTION_NAME4 = "Function4";
const char *WRONG_MODEL_NAME = "Model2";
const char *STATE_NAME = "State1";
const char *NEW_STATE_NAME = "State2";
const char *WRONG_STATE_NAME = "State3";
const char *OTHER_STATE_NAME = "State4";

TEST(LayerDescriptionTest, AgentFunction) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    a.newState(STATE_NAME);
    a.newState(NEW_STATE_NAME);
    a.newState(WRONG_STATE_NAME);
    a.newState(OTHER_STATE_NAME);
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    f1.setInitialState(STATE_NAME);
    f1.setEndState(STATE_NAME);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    f2.setInitialState(NEW_STATE_NAME);
    f2.setEndState(NEW_STATE_NAME);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f3.setInitialState(WRONG_STATE_NAME);
    f3.setEndState(WRONG_STATE_NAME);
    AgentFunctionDescription &f4 = a.newFunction(FUNCTION_NAME4, agent_fn4);
    f4.setInitialState(OTHER_STATE_NAME);
    f4.setEndState(OTHER_STATE_NAME);
    LayerDescription &l = _m.newLayer(LAYER_NAME);
    EXPECT_EQ(l.getAgentFunctionsCount(), 0u);
    // Add by fn
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn1));
    EXPECT_EQ(l.getAgentFunctionsCount(), 1u);
    // Add by fn description
    EXPECT_NO_THROW(l.addAgentFunction(f2));
    EXPECT_EQ(l.getAgentFunctionsCount(), 2u);
    // Add by string
    EXPECT_NO_THROW(l.addAgentFunction(std::string(FUNCTION_NAME3)));
    EXPECT_EQ(l.getAgentFunctionsCount(), 3u);
    // Add by string literal (char*)
    EXPECT_NO_THROW(l.addAgentFunction(FUNCTION_NAME4));
    EXPECT_EQ(l.getAgentFunctionsCount(), 4u);

    // Cannot add function not attached to an agent
    EXPECT_THROW(l.addAgentFunction(f1), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f4), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(agent_fn5), InvalidAgentFunc);
    EXPECT_EQ(l.getAgentFunctionsCount(), 4u);

    // Cannot add duplicate function variable with same name
    EXPECT_THROW(l.addAgentFunction(agent_fn2), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f3), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(std::string(FUNCTION_NAME4)), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(FUNCTION_NAME1), InvalidAgentFunc);
    EXPECT_EQ(l.getAgentFunctionsCount(), 4u);

    // Function have the right name (not implemented
    // EXPECT_TRUE(m.hasAgentFunction(FUNCTION_NAME1));
    // EXPECT_TRUE(m.hasAgentFunction(FUNCTION_NAME2));
    // EXPECT_TRUE(m.hasAgentFunction(FUNCTION_NAME3));
    // EXPECT_TRUE(m.hasAgentFunction(FUNCTION_NAME4));
}

TEST(LayerDescriptionTest, HostFunction) {
    ModelDescription _m(MODEL_NAME);
    LayerDescription &l = _m.newLayer(LAYER_NAME);
    EXPECT_EQ(l.getHostFunctionsCount(), 0u);
    l.addHostFunction(host_fn);
    EXPECT_EQ(l.getHostFunctionsCount(), 1u);
    // Cannot create function with same name
    EXPECT_THROW(l.addHostFunction(host_fn), InvalidHostFunc);
}

TEST(LayerDescriptionTest, AgentFunction_WrongModel) {
    ModelDescription _m(MODEL_NAME);
    ModelDescription _m2(WRONG_MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentDescription &a2 = _m2.newAgent(AGENT_NAME);
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a2.newFunction(FUNCTION_NAME1, agent_fn1);
    LayerDescription &l = _m.newLayer(LAYER_NAME);

    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addAgentFunction(f2), DifferentModel);
}

TEST(LayerDescriptionTest, SameAgentAndState1) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    a.newFunction(FUNCTION_NAME1, agent_fn2);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn3);
    LayerDescription &l = _m.newLayer();
    // Both have agent in default state
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn2));
    EXPECT_THROW(l.addAgentFunction(agent_fn3), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(FUNCTION_NAME2), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState2) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    a.newState(STATE_NAME);
    a.newState(NEW_STATE_NAME);
    a.newState(WRONG_STATE_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn2);
    f.setInitialState(STATE_NAME);
    f.setEndState(NEW_STATE_NAME);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn3);
    f2.setInitialState(STATE_NAME);
    f2.setEndState(NEW_STATE_NAME);
    LayerDescription &l = _m.newLayer();
    // Both have STATE_NAME:NEW_STATE_NAME state
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn2));
    EXPECT_THROW(l.addAgentFunction(agent_fn3), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(FUNCTION_NAME2), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState3) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    a.newState(STATE_NAME);
    a.newState(NEW_STATE_NAME);
    a.newState(WRONG_STATE_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn2);
    f.setInitialState(STATE_NAME);
    f.setEndState(NEW_STATE_NAME);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn3);
    f2.setInitialState(STATE_NAME);
    f2.setEndState(WRONG_STATE_NAME);
    LayerDescription &l = _m.newLayer();
    // Both share initial state
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn2));
    EXPECT_THROW(l.addAgentFunction(agent_fn3), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(FUNCTION_NAME2), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState4) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    a.newState(STATE_NAME);
    a.newState(NEW_STATE_NAME);
    a.newState(WRONG_STATE_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn2);
    f.setInitialState(STATE_NAME);
    f.setEndState(NEW_STATE_NAME);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn3);
    f2.setInitialState(WRONG_STATE_NAME);
    f2.setEndState(NEW_STATE_NAME);
    LayerDescription &l = _m.newLayer();
    // start matches end and vice versa
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn2));
    EXPECT_THROW(l.addAgentFunction(agent_fn3), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(FUNCTION_NAME2), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState5) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    a.newState(STATE_NAME);
    a.newState(NEW_STATE_NAME);
    a.newState(WRONG_STATE_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn2);
    f.setInitialState(STATE_NAME);
    f.setEndState(NEW_STATE_NAME);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn3);
    f2.setInitialState(NEW_STATE_NAME);
    f2.setEndState(WRONG_STATE_NAME);
    LayerDescription &l = _m.newLayer();
    // end matches start state
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn2));
    EXPECT_THROW(l.addAgentFunction(agent_fn3), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(FUNCTION_NAME2), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState6) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    a.newState(STATE_NAME);
    a.newState(NEW_STATE_NAME);
    a.newState(WRONG_STATE_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn2);
    f.setInitialState(STATE_NAME);
    f.setEndState(NEW_STATE_NAME);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn3);
    f2.setInitialState(WRONG_STATE_NAME);
    f2.setEndState(NEW_STATE_NAME);
    LayerDescription &l = _m.newLayer();
    // start matches end state
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn2));
    EXPECT_THROW(l.addAgentFunction(agent_fn3), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(FUNCTION_NAME2), InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), InvalidAgentFunc);
}

TEST(LayerDescriptionTest, SubModelAndHostOrAgentFunction) {
    ModelDescription m2("model2");
    m2.addExitCondition(exit_cdn);
    ModelDescription m3("model2");
    m3.addExitCondition(exit_cdn);
    ModelDescription m(MODEL_NAME);
    auto &sm = m.newSubModel("sm", m2);
    auto &sm2 = m.newSubModel("sm2", m3);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    auto &af1 = a.newFunction("", agent_fn1);
    // Submodel can't go in layer with agent function and host fn
    auto &layer1 = m.newLayer();
    EXPECT_NO_THROW(layer1.addAgentFunction(af1));
    EXPECT_NO_THROW(layer1.addHostFunction(host_fn));
    EXPECT_THROW(layer1.addSubModel(sm), InvalidLayerMember);
    // Submodel can't go in layer with agent function
    auto &layer2 = m.newLayer();
    EXPECT_NO_THROW(layer2.addAgentFunction(af1));
    EXPECT_THROW(layer2.addSubModel(sm), InvalidLayerMember);
    EXPECT_NO_THROW(layer2.addHostFunction(host_fn));
    // Submodel can't go in layer with host function
    auto &layer3 = m.newLayer();
    EXPECT_NO_THROW(layer3.addHostFunction(host_fn));
    EXPECT_THROW(layer3.addSubModel(sm), InvalidLayerMember);
    EXPECT_NO_THROW(layer3.addAgentFunction(af1));
    // Host and agent functions can't go in layer with submodel
    // Submodel can't go in layer with submodel
    auto &layer4 = m.newLayer();
    EXPECT_NO_THROW(layer4.addSubModel(sm));
    EXPECT_THROW(layer4.addSubModel(sm2), InvalidSubModel);
    EXPECT_THROW(layer4.addAgentFunction(af1), InvalidLayerMember);
    EXPECT_THROW(layer4.addHostFunction(host_fn), InvalidLayerMember);
}
}  // namespace test_layer
