#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace test_dependency_graph {
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
FLAMEGPU_AGENT_FUNCTION(agent_fn4, MsgNone, MsgNone) {
    // do nothing
    return ALIVE;
}

FLAMEGPU_HOST_FUNCTION(host_fn1) {
    // do nothing
    return;
}

FLAMEGPU_HOST_FUNCTION(host_fn2) {
    // do nothing
    return;
}

FLAMEGPU_HOST_FUNCTION(host_fn3) {
    // do nothing
    return;
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
const char *FUNCTION_NAME4 = "Function4";
const char *HOST_FN_NAME1 = "HostFn1";
const char *HOST_FN_NAME2 = "HostFn2";
const char *HOST_FN_NAME3 = "HostFn3";
const char *STATE_NAME = "State1";
const char *NEW_STATE_NAME = "State2";
const char *WRONG_STATE_NAME = "State3";
const char *OTHER_STATE_NAME = "State4";

TEST(DependencyGraphTest, ValidateEmptyGraph) {
    DependencyGraph graph;
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph); 
}

TEST(DependencyGraphTest, ValidateSingleNode) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    DependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_TRUE(graph.validateDependencyGraph()); 
}

TEST(DependencyGraphTest, ValidateSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f2);
    DependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_TRUE(graph.validateDependencyGraph()); 
}

TEST(DependencyGraphTest, ValidateBranch) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f);
    DependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_TRUE(graph.validateDependencyGraph()); 
}

TEST(DependencyGraphTest, ValidateCycle) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f2.dependsOn(&f3);
    f3.dependsOn(&f2);
    DependencyGraph graph;
    graph.addRoot(&f);
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph); 
}

TEST(DependencyGraphTest, ValidateRootWithDependencies) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f2);
    DependencyGraph graph;
    graph.addRoot(&f2);
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph); 
}

TEST(DependencyGraphTest, ConstructLayersSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f2);
    DependencyGraph graph;
    graph.addRoot(&f);
    graph.generateLayers(_m); 
}

TEST(DependencyGraphTest, ConstructLayersRootTwoChildrenConflict) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f);
    DependencyGraph graph;
    graph.addRoot(&f);
    graph.generateLayers(_m); 
}

TEST(DependencyGraphTest, AddHostFunctionAsDependent) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    hf.dependsOn(&f);
    DependencyGraph graph;
    graph.addRoot(&f);
    graph.generateLayers(_m); 
}

TEST(DependencyGraphTest, AddHostFunctionAsDependency) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    f.dependsOn(&hf);
    DependencyGraph graph;
    graph.addRoot(&hf);
    graph.generateLayers(_m); 
}

TEST(DependencyGraphTest, DOTDiagramSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f2);
    DependencyGraph graph;
    graph.addRoot(&f);
    graph.generateLayers(_m);
    graph.generateDOTDiagram("singlechain.gv");
}

TEST(DependencyGraphTest, DOTDiagramTwoDependencies) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(&f);
    f3.dependsOn(&f);
    DependencyGraph graph;
    graph.addRoot(&f);
    graph.generateDOTDiagram("twodeps.gv");
}

TEST(DependencyGraphTest, DOTDiagramDiamond) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    AgentFunctionDescription &f4 = a.newFunction(FUNCTION_NAME4, agent_fn4);
    f2.dependsOn(&f);
    f3.dependsOn(&f);
    f4.dependsOn(&f2);
    f4.dependsOn(&f3);
    DependencyGraph graph;
    graph.addRoot(&f);
    graph.generateDOTDiagram("diamond.gv");
}

TEST(DependencyGraphTest, DOTDiagramHostFunctions) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    AgentFunctionDescription &f4 = a.newFunction(FUNCTION_NAME4, agent_fn4);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    HostFunctionDescription hf2(HOST_FN_NAME2, host_fn2);
    f2.dependsOn(&f);
    f3.dependsOn(&hf);
    f4.dependsOn(&f2);
    f4.dependsOn(&hf);
    hf2.dependsOn(&f3);
    DependencyGraph graph;
    graph.addRoot(&f);
    graph.addRoot(&hf);
    graph.generateDOTDiagram("host_functions.gv");
}
}  // namespace test_agent_function_dependency_graph
