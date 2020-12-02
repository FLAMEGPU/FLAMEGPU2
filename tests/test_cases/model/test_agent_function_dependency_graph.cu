#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace test_agent_function_dependency_graph {
FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgBruteForce, MsgBruteForce) {
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

const char *MODEL_NAME = "Model";
const char *WRONG_MODEL_NAME = "Model2";
const char *AGENT_NAME = "Agent1";
const char *AGENT_NAME2 = "Agent2";
const char *AGENT_NAME3 = "Agent3";
const char *MESSAGE_NAME1 = "Message1";
const char *MESSAGE_NAME2 = "Message2";
const char *VARIABLE_NAME1 = "Var1";
const char *VARIABLE_NAME2 = "Var2";
const char *VARIABLE_NAME3 = "Var3";
const char *FUNCTION_NAME1 = "Function1";
const char *FUNCTION_NAME2 = "Function2";
const char *FUNCTION_NAME3 = "Function3";
const char *STATE_NAME = "State1";
const char *NEW_STATE_NAME = "State2";
const char *WRONG_STATE_NAME = "State3";
const char *OTHER_STATE_NAME = "State4";

TEST(AgentFunctionDependencyGraphTest, ValidateEmptyGraph) {
    AgentFunctionDependencyGraph graph;
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph); 
}

TEST(AgentFunctionDependencyGraphTest, ValidateSingleNode) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_TRUE(graph.validateDependencyGraph()); 
}

TEST(AgentFunctionDependencyGraphTest, ValidateSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f2);
    AgentFunctionDependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_TRUE(graph.validateDependencyGraph()); 
}

TEST(AgentFunctionDependencyGraphTest, ValidateBranch) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f);
    AgentFunctionDependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_TRUE(graph.validateDependencyGraph()); 
}

TEST(AgentFunctionDependencyGraphTest, ValidateCycle) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f2.dependsOn(&f3);
    f3.dependsOn(&f2);
    AgentFunctionDependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph); 
}

TEST(AgentFunctionDependencyGraphTest, ValidateRootWithDependencies) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f2);
    AgentFunctionDependencyGraph graph;
    graph.addRoot(&f2);
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph); 
}
TEST(AgentFunctionDependencyGraphTest, ConstructLayersSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f2);
    AgentFunctionDependencyGraph graph;
    graph.addRoot(&f);
    graph.generateLayers(_m); 
}
TEST(AgentFunctionDependencyGraphTest, ConstructLayersRootTwoChildrenConflict) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f);
    AgentFunctionDependencyGraph graph;
    graph.addRoot(&f);
    graph.generateLayers(_m); 
}
}  // namespace test_agent_function
