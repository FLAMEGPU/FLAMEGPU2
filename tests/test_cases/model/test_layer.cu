#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

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
const char *MODEL_NAME = "Model";
const char *AGENT_NAME = "Agent1";
const char *LAYER_NAME = "Layer1";
const char *FUNCTION_NAME1 = "Function1";
const char *FUNCTION_NAME2 = "Function2";
const char *FUNCTION_NAME3 = "Function3";
const char *FUNCTION_NAME4 = "Function4";
const char *WRONG_MODEL_NAME = "Model2";

TEST(LayerDescriptionTest, AgentFunction) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f1 = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    AgentFunctionDescription &f4 = a.newFunction(FUNCTION_NAME4, agent_fn4);
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
}  // namespace test_layer
