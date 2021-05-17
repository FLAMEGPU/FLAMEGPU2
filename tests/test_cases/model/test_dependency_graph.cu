#include <cstdio>

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

#include "gtest/gtest.h"

namespace flamegpu {


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

FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}

const char *MODEL_NAME = "Model";
const char *MODEL_NAME2 = "Model2";
const char *WRONG_MODEL_NAME = "Model2";
const char *SUBMODEL_NAME = "SubModel1";
const char *SUBAGENT_NAME = "SubAgent1";
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
const char *LAYER_NAME = "Layer1";

TEST(DependencyGraphTest, ValidateEmptyGraph) {
    DependencyGraph graph;
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph);
}

TEST(DependencyGraphTest, ValidateSingleNode) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    EXPECT_TRUE(graph.validateDependencyGraph());
}

TEST(DependencyGraphTest, ValidateSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f3.dependsOn(f2);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    EXPECT_TRUE(graph.validateDependencyGraph());
}

TEST(DependencyGraphTest, ValidateBranch) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f3.dependsOn(f);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    EXPECT_TRUE(graph.validateDependencyGraph());
}

TEST(DependencyGraphTest, ValidateCycle) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f2.dependsOn(f3);
    f3.dependsOn(f2);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph);
}

TEST(DependencyGraphTest, ValidateRootWithDependencies) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f3.dependsOn(f2);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f2);
    EXPECT_THROW(graph.validateDependencyGraph(), InvalidDependencyGraph);
}

TEST(DependencyGraphTest, ConstructLayersSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f3.dependsOn(f2);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    _m.generateLayers();
}

TEST(DependencyGraphTest, ConstructLayersRootTwoChildrenConflict) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f3.dependsOn(f);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    _m.generateLayers();
}

TEST(DependencyGraphTest, AddHostFunctionAsDependent) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    hf.dependsOn(f);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    _m.generateLayers();
}

TEST(DependencyGraphTest, AddHostFunctionAsDependency) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    f.dependsOn(hf);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(hf);
    _m.generateLayers();
}

TEST(DependencyGraphTest, AddSubmodelAsDependent) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);

    ModelDescription _sm(SUBMODEL_NAME);
    _sm.newAgent(SUBAGENT_NAME);
    _sm.addExitCondition(ExitAlways);
    SubModelDescription& _smd = _m.newSubModel("sub", _sm);

    _smd.dependsOn(f);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    _m.generateLayers();
}

TEST(DependencyGraphTest, AddSubmodelAsDependency) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);

    ModelDescription _sm(SUBMODEL_NAME);
    _sm.newAgent(SUBAGENT_NAME);
    _sm.addExitCondition(ExitAlways);
    SubModelDescription& _smd = _m.newSubModel("sub", _sm);

    f.dependsOn(_smd);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(_smd);
    _m.generateLayers();
}

TEST(DependencyGraphTest, DOTDiagramSingleChain) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f3.dependsOn(f2);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    _m.generateLayers();
    graph.generateDOTDiagram("singlechain.gv");

    // Check file contents
    std::ifstream dot("singlechain.gv");
    std::stringstream dotBuffer;
    if (dot) {
        dotBuffer << dot.rdbuf();
    }
    std::string expectedDot = R"###(digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function3[style = filled, color = red];
    Function1 -> Function2;
    Function2 -> Function3;
})###";
    EXPECT_EQ(expectedDot, dotBuffer.str());
    dot.close();
    // Test passed, remove file
    std::remove("singlechain.gv");
}

TEST(DependencyGraphTest, DOTDiagramTwoDependencies) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    f2.dependsOn(f);
    f3.dependsOn(f);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    graph.generateDOTDiagram("twodeps.gv");

    // Check file contents
    std::ifstream dot("twodeps.gv");
    std::stringstream dotBuffer;
    if (dot) {
        dotBuffer << dot.rdbuf();
    }
    std::string expectedDot = R"###(digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function3[style = filled, color = red];
    Function1 -> Function2;
    Function1 -> Function3;
})###";
    EXPECT_EQ(expectedDot, dotBuffer.str());
    dot.close();
    // Test passed, remove file
    std::remove("twodeps.gv");
}

TEST(DependencyGraphTest, DOTDiagramDiamond) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    AgentFunctionDescription &f4 = a.newFunction(FUNCTION_NAME4, agent_fn4);
    f2.dependsOn(f);
    f3.dependsOn(f);
    f4.dependsOn(f2);
    f4.dependsOn(f3);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    graph.generateDOTDiagram("diamond.gv");

    // Check file contents
    std::ifstream dot("diamond.gv");
    std::stringstream dotBuffer;
    if (dot) {
        dotBuffer << dot.rdbuf();
    }
    std::string expectedDot = R"###(digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function4[style = filled, color = red];
    Function3[style = filled, color = red];
    Function4[style = filled, color = red];
    Function1 -> Function2;
    Function2 -> Function4;
    Function1 -> Function3;
    Function3 -> Function4;
})###";
    EXPECT_EQ(expectedDot, dotBuffer.str());
    dot.close();
    // Test passed, remove file
    std::remove("diamond.gv");
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
    f2.dependsOn(f);
    f3.dependsOn(hf);
    f4.dependsOn(f2);
    f4.dependsOn(hf);
    hf2.dependsOn(f3);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    graph.addRoot(hf);
    graph.generateDOTDiagram("host_functions.gv");

    // Check file contents
    std::ifstream dot("host_functions.gv");
    std::stringstream dotBuffer;
    if (dot) {
        dotBuffer << dot.rdbuf();
    }
    std::string expectedDot = R"###(digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function4[style = filled, color = red];
    HostFn1[style = filled, color = yellow];
    Function3[style = filled, color = red];
    HostFn2[style = filled, color = yellow];
    Function4[style = filled, color = red];
    Function1 -> Function2;
    Function2 -> Function4;
    HostFn1 -> Function3;
    Function3 -> HostFn2;
    HostFn1 -> Function4;
})###";
    EXPECT_EQ(expectedDot, dotBuffer.str());
    dot.close();
    // Test passed, remove file
    std::remove("host_functions.gv");
}

TEST(DependencyGraphTest, DOTDiagramAllDependencies) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    AgentFunctionDescription &f4 = a.newFunction(FUNCTION_NAME4, agent_fn4);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    HostFunctionDescription hf2(HOST_FN_NAME2, host_fn2);
    ModelDescription _sm(SUBMODEL_NAME);
    _sm.newAgent(SUBAGENT_NAME);
    _sm.addExitCondition(ExitAlways);
    SubModelDescription& _smd = _m.newSubModel("sub", _sm);
    f2.dependsOn(f);
    f3.dependsOn(hf);
    f4.dependsOn(f2);
    f4.dependsOn(hf);
    hf2.dependsOn(f3);
    _smd.dependsOn(hf2);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    graph.addRoot(hf);
    graph.generateDOTDiagram("all_dependencies.gv");

    // Check file contents
    std::ifstream dot("all_dependencies.gv");
    std::stringstream dotBuffer;
    if (dot) {
        dotBuffer << dot.rdbuf();
    }
    std::string expectedDot = R"###(digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function4[style = filled, color = red];
    HostFn1[style = filled, color = yellow];
    Function3[style = filled, color = red];
    HostFn2[style = filled, color = yellow];
    sub[style = filled, color = green];
    Function4[style = filled, color = red];
    Function1 -> Function2;
    Function2 -> Function4;
    HostFn1 -> Function3;
    Function3 -> HostFn2;
    HostFn2 -> sub;
    HostFn1 -> Function4;
})###";
    EXPECT_EQ(expectedDot, dotBuffer.str());
    dot.close();
    // Test passed, remove file
    std::remove("all_dependencies.gv");
}
TEST(DependencyGraphTest, CorrectLayersAllDependencies) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a.newFunction(FUNCTION_NAME3, agent_fn3);
    AgentFunctionDescription &f4 = a.newFunction(FUNCTION_NAME4, agent_fn4);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    HostFunctionDescription hf2(HOST_FN_NAME2, host_fn2);
    ModelDescription _sm(SUBMODEL_NAME);
    _sm.newAgent(SUBAGENT_NAME);
    _sm.addExitCondition(ExitAlways);
    SubModelDescription& _smd = _m.newSubModel("sub", _sm);
    f2.dependsOn(f);
    f3.dependsOn(hf);
    f4.dependsOn(f2, hf);
    hf2.dependsOn(f3);
    _smd.dependsOn(hf2);
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);
    graph.addRoot(hf);
    _m.generateLayers();
    std::string expectedLayers = R"###(--------------------
Layer 0
--------------------
Function1

--------------------
Layer 1
--------------------
HostFn1

--------------------
Layer 2
--------------------
Function2

--------------------
Layer 3
--------------------
Function3

--------------------
Layer 4
--------------------
Function4

--------------------
Layer 5
--------------------
HostFn2

--------------------
Layer 6
--------------------
sub

)###";
    EXPECT_TRUE(expectedLayers == graph.getConstructedLayersString());
}
TEST(DependencyGraphTest, CorrectLayersConcurrent) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentDescription &a2 = _m.newAgent(AGENT_NAME2);
    AgentDescription &a3 = _m.newAgent(AGENT_NAME3);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a2.newFunction(FUNCTION_NAME2, agent_fn2);
    AgentFunctionDescription &f3 = a3.newFunction(FUNCTION_NAME3, agent_fn3);
    HostFunctionDescription hf(HOST_FN_NAME1, host_fn1);
    HostFunctionDescription hf2(HOST_FN_NAME2, host_fn2);
    f.dependsOn(hf);
    f2.dependsOn(hf);
    hf2.dependsOn(f, f2, f3);

    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f3);
    graph.addRoot(hf);
    _m.generateLayers();
    std::string expectedLayers = R"###(--------------------
Layer 0
--------------------
Function3

--------------------
Layer 1
--------------------
HostFn1

--------------------
Layer 2
--------------------
Function1
Function2

--------------------
Layer 3
--------------------
HostFn2

)###";
    EXPECT_EQ(expectedLayers, graph.getConstructedLayersString());
}
TEST(DependencyGraphTest, InterModelDependency) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);

    ModelDescription _m2(MODEL_NAME2);
    AgentDescription &a2 = _m2.newAgent(AGENT_NAME2);
    AgentFunctionDescription &f2 = a2.newFunction(FUNCTION_NAME2, agent_fn2);

    EXPECT_THROW(f2.dependsOn(f), InvalidDependencyGraph);
}
TEST(DependencyGraphTest, UnattachedFunctionWarning) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);

    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);

    // Intercept std::cerr
    std::stringstream buffer;
    std::streambuf* prev = std::cerr.rdbuf();
    std::cerr.rdbuf(buffer.rdbuf());
    _m.generateLayers();
    // Reset cerr
    std::cerr.rdbuf(prev);
    EXPECT_EQ(buffer.str(), "WARNING: Not all agent functions are used in the dependency graph - have you forgotten to add one?");
}
TEST(DependencyGraphTest, ModelAlreadyHasLayers) {
    ModelDescription _m(MODEL_NAME);
    AgentDescription &a = _m.newAgent(AGENT_NAME);
    AgentFunctionDescription &f = a.newFunction(FUNCTION_NAME1, agent_fn1);
    AgentFunctionDescription &f2 = a.newFunction(FUNCTION_NAME2, agent_fn2);

    // Create manual layer
    LayerDescription &l = _m.newLayer(LAYER_NAME);
    l.addAgentFunction(f2);

    // Create DG
    DependencyGraph& graph = _m.getDependencyGraph();
    graph.addRoot(f);

    EXPECT_THROW(_m.generateLayers(), InvalidDependencyGraph);
}
}  // namespace test_dependency_graph
}  // namespace flamegpu
