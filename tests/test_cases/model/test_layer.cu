#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_layer {
FLAMEGPU_AGENT_FUNCTION(agent_fn1, MessageNone, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn2, MessageNone, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn3, MessageNone, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn4, MessageNone, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn5, MessageNone, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_messageout1, MessageNone, MessageSpatial3D) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_messageout2, MessageNone, MessageSpatial3D) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_messagein1, MessageSpatial3D, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(agent_fn_messagein2, MessageSpatial3D, MessageNone) {
    // do nothing
    return ALIVE;
}
FLAMEGPU_HOST_FUNCTION(host_fn) {
    // do nothing
}
FLAMEGPU_HOST_FUNCTION(host_fn2) {
    // do nothing
}
FLAMEGPU_EXIT_CONDITION(exit_cdn) {
    return EXIT;
}
const char *MODEL_NAME = "Model";
const char* AGENT_NAME = "Agent1";
const char* AGENT_NAME2 = "Agent2";
const char *MESSAGE_NAME = "Message";
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
    AgentDescription a = _m.newAgent(AGENT_NAME);
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
    EXPECT_NO_THROW(l.addAgentFunction(AGENT_NAME, std::string(FUNCTION_NAME3)));
    EXPECT_EQ(l.getAgentFunctionsCount(), 3u);
    // Add by string literal (char*)
    EXPECT_NO_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME4));
    EXPECT_EQ(l.getAgentFunctionsCount(), 4u);

    // Cannot add function not attached to an agent
    EXPECT_THROW(l.addAgentFunction(f1), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f4), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(agent_fn5), exception::InvalidAgentFunc);
    EXPECT_EQ(l.getAgentFunctionsCount(), 4u);

    // Cannot add duplicate function variable with same name
    EXPECT_THROW(l.addAgentFunction(agent_fn2), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f3), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, std::string(FUNCTION_NAME4)), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME1), exception::InvalidAgentFunc);
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
    EXPECT_THROW(l.addHostFunction(host_fn), exception::InvalidLayerMember);
}

TEST(LayerDescriptionTest, AgentFunction_WrongModel) {
    ModelDescription _m(MODEL_NAME);
    ModelDescription _m2(WRONG_MODEL_NAME);
    AgentDescription a = _m.newAgent(AGENT_NAME);
    AgentDescription a2 = _m2.newAgent(AGENT_NAME);
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a2.newFunction(FUNCTION_NAME1, agent_fn1);
    LayerDescription &l = _m.newLayer(LAYER_NAME);

    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addAgentFunction(f2), exception::DifferentModel);
}

TEST(LayerDescriptionTest, SameAgentAndState1) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a = _m.newAgent(AGENT_NAME);
    a.newFunction(FUNCTION_NAME1, agent_fn2);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn3);
    LayerDescription &l = _m.newLayer();
    // Both have agent in default state
    EXPECT_NO_THROW(l.addAgentFunction(agent_fn2));
    EXPECT_THROW(l.addAgentFunction(agent_fn3), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState2) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a = _m.newAgent(AGENT_NAME);
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
    EXPECT_THROW(l.addAgentFunction(agent_fn3), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState3) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a = _m.newAgent(AGENT_NAME);
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
    EXPECT_THROW(l.addAgentFunction(agent_fn3), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState4) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a = _m.newAgent(AGENT_NAME);
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
    EXPECT_THROW(l.addAgentFunction(agent_fn3), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState5) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a = _m.newAgent(AGENT_NAME);
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
    EXPECT_THROW(l.addAgentFunction(agent_fn3), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameAgentAndState6) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a = _m.newAgent(AGENT_NAME);
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
    EXPECT_THROW(l.addAgentFunction(agent_fn3), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidAgentFunc);
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, SameMessageListOutOut) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    AgentDescription a2 = _m.newAgent(AGENT_NAME2);
    MessageSpatial3D::Description& message = _m.newMessage<MessageSpatial3D>(MESSAGE_NAME);
    AgentFunctionDescription& f1 = a1.newFunction(FUNCTION_NAME1, agent_fn_messageout1);
    AgentFunctionDescription& f2 = a2.newFunction(FUNCTION_NAME1, agent_fn_messageout2);
    f1.setMessageOutput(message);
    f2.setMessageOutput(message);
    LayerDescription& l = _m.newLayer();
    // Both agent functions output to same message list
    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME2, FUNCTION_NAME1), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(agent_fn_messageout2), exception::InvalidLayerMember);
}
TEST(LayerDescriptionTest, SameMessageListOutIn) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    AgentDescription a2 = _m.newAgent(AGENT_NAME2);
    MessageSpatial3D::Description& message = _m.newMessage<MessageSpatial3D>(MESSAGE_NAME);
    AgentFunctionDescription& f1 = a1.newFunction(FUNCTION_NAME1, agent_fn_messageout1);
    AgentFunctionDescription& f2 = a2.newFunction(FUNCTION_NAME1, agent_fn_messagein2);
    f1.setMessageOutput(message);
    f2.setMessageInput(message);
    LayerDescription& l = _m.newLayer();
    // Both agent functions output to same message list
    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME2, FUNCTION_NAME1), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(agent_fn_messagein2), exception::InvalidLayerMember);
}
TEST(LayerDescriptionTest, SameMessageListInOut) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    AgentDescription a2 = _m.newAgent(AGENT_NAME2);
    MessageSpatial3D::Description& message = _m.newMessage<MessageSpatial3D>(MESSAGE_NAME);
    AgentFunctionDescription& f1 = a1.newFunction(FUNCTION_NAME1, agent_fn_messagein1);
    AgentFunctionDescription& f2 = a2.newFunction(FUNCTION_NAME1, agent_fn_messageout2);
    f1.setMessageInput(message);
    f2.setMessageOutput(message);
    LayerDescription& l = _m.newLayer();
    // Both agent functions output to same message list
    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME2, FUNCTION_NAME1), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(agent_fn_messageout2), exception::InvalidLayerMember);
}
TEST(LayerDescriptionTest, AgentOutMatchesInputState1) {
    // Can't add an agent function which outputs to the same agent state that is an input state for another agent function
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    a1.newState("a");
    a1.newState("b");
    AgentFunctionDescription& f1 = a1.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription& f2 = a1.newFunction(FUNCTION_NAME2, agent_fn2);
    f1.setInitialState("a");
    f1.setEndState("a");  // Redundant for the test?
    f2.setInitialState("b");
    f2.setEndState("b");  // Redundant for the test?
    f2.setAgentOutput(a1, "a");
    LayerDescription& l = _m.newLayer();
    // Both agent functions output to same message list
    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(agent_fn2), exception::InvalidLayerMember);
}
TEST(LayerDescriptionTest, AgentOutMatchesInputState2) {
    // Can't add an agent function which outputs to the same agent state that is an input state for another agent function
    // This tests the inverse order (agent output fn added first)
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    a1.newState("a");
    a1.newState("b");
    AgentFunctionDescription& f1 = a1.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription& f2 = a1.newFunction(FUNCTION_NAME2, agent_fn2);
    f1.setInitialState("a");
    f1.setEndState("a");  // Redundant for the test?
    f2.setInitialState("b");
    f2.setEndState("b");  // Redundant for the test?
    f1.setAgentOutput(a1, "b");
    LayerDescription& l = _m.newLayer();
    // Both agent functions output to same message list
    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addAgentFunction(f2), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME2), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(agent_fn2), exception::InvalidLayerMember);
}
TEST(LayerDescriptionTest, NoSuitableAgentFunctions) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    AgentDescription a2 = _m.newAgent(AGENT_NAME2);
    a1.newFunction(FUNCTION_NAME1, agent_fn2);
    a2.newFunction(FUNCTION_NAME1, agent_fn2);
    LayerDescription& l = _m.newLayer();
    // No agent functions within the model use this agent function body
    EXPECT_THROW(l.addAgentFunction(agent_fn_messageout1), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, MultipleSuitableAgentFunctions) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    AgentDescription a2 = _m.newAgent(AGENT_NAME2);
    a1.newFunction(FUNCTION_NAME1, agent_fn2);
    a2.newFunction(FUNCTION_NAME1, agent_fn2);
    LayerDescription& l = _m.newLayer();
    // Multiple agent functions within the model use this agent function body
    EXPECT_THROW(l.addAgentFunction(agent_fn2), exception::InvalidAgentFunc);
}
TEST(LayerDescriptionTest, AgentFnHostFnSameLayer) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    auto &f1 = a1.newFunction(FUNCTION_NAME1, agent_fn1);
    LayerDescription& l = _m.newLayer();
    // Multiple agent functions within the model use this agent function body
    EXPECT_NO_THROW(l.addAgentFunction(f1));
    EXPECT_THROW(l.addHostFunction(host_fn), exception::InvalidLayerMember);
}
TEST(LayerDescriptionTest, HostFnAgentFnSameLayer) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription a1 = _m.newAgent(AGENT_NAME);
    auto& f1 = a1.newFunction(FUNCTION_NAME1, agent_fn1);
    LayerDescription& l = _m.newLayer();
    // Multiple agent functions within the model use this agent function body
    EXPECT_NO_THROW(l.addHostFunction(host_fn));
    EXPECT_THROW(l.addAgentFunction(f1), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(AGENT_NAME, FUNCTION_NAME1), exception::InvalidLayerMember);
    EXPECT_THROW(l.addAgentFunction(agent_fn1), exception::InvalidLayerMember);
}

TEST(LayerDescriptionTest, SubModelAndHostOrAgentFunction) {
    ModelDescription m2("model2");
    m2.addExitCondition(exit_cdn);
    ModelDescription m3("model2");
    m3.addExitCondition(exit_cdn);
    ModelDescription m(MODEL_NAME);
    auto &sm = m.newSubModel("sm", m2);
    auto &sm2 = m.newSubModel("sm2", m3);
    AgentDescription a = m.newAgent(AGENT_NAME);
    AgentDescription a2 = m.newAgent(AGENT_NAME2);

    auto &af1 = a.newFunction("a", agent_fn1);
    auto &af2 = a2.newFunction("b", agent_fn2);
    // Submodel can't go in layer with agent function
    auto &layer2 = m.newLayer();
    EXPECT_NO_THROW(layer2.addAgentFunction(af1));
    EXPECT_THROW(layer2.addSubModel(sm), exception::InvalidLayerMember);
    EXPECT_NO_THROW(layer2.addAgentFunction(af2));
    // Submodel can't go in layer with host function
    auto &layer3 = m.newLayer();
    EXPECT_NO_THROW(layer3.addHostFunction(host_fn));
    EXPECT_THROW(layer3.addSubModel(sm), exception::InvalidLayerMember);
    EXPECT_THROW(layer3.addHostFunction(host_fn2), exception::InvalidLayerMember);
    // Host and agent functions can't go in layer with submodel
    // Submodel can't go in layer with submodel
    auto &layer4 = m.newLayer();
    EXPECT_NO_THROW(layer4.addSubModel(sm));
    EXPECT_THROW(layer4.addSubModel(sm2), exception::InvalidSubModel);
    EXPECT_THROW(layer4.addAgentFunction(af1), exception::InvalidLayerMember);
    EXPECT_THROW(layer4.addAgentFunction(AGENT_NAME, ""), exception::InvalidLayerMember);
    EXPECT_THROW(layer4.addAgentFunction(agent_fn1), exception::InvalidLayerMember);
    EXPECT_THROW(layer4.addHostFunction(host_fn), exception::InvalidLayerMember);
}
}  // namespace test_layer
}  // namespace flamegpu
