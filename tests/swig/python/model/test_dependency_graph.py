import pytest
from unittest import TestCase
from pyflamegpu import *
import io
import sys
import os

MODEL_NAME = "Model"
MODEL_NAME2 = "Model2"
WRONG_MODEL_NAME = "Model2"
SUBMODEL_NAME = "SubModel1"
SUBAGENT_NAME = "SubAgent1"
AGENT_NAME = "Agent1"
AGENT_NAME2 = "Agent2"
AGENT_NAME3 = "Agent3"
LAYER_NAME = "Layer1"
MESSAGE_NAME1 = "Message1"
MESSAGE_NAME2 = "Message2"
VARIABLE_NAME1 = "Var1"
VARIABLE_NAME2 = "Var2"
VARIABLE_NAME3 = "Var3"
FUNCTION_NAME1 = "Function1"
FUNCTION_NAME2 = "Function2"
FUNCTION_NAME3 = "Function3"
FUNCTION_NAME4 = "Function4"
HOST_FN_NAME1 = "HostFn1"
HOST_FN_NAME2 = "HostFn2"
HOST_FN_NAME3 = "HostFn3"
STATE_NAME = "State1"
NEW_STATE_NAME = "State2"
WRONG_STATE_NAME = "State3"
OTHER_STATE_NAME = "State4"

class EmptyHostFunc(pyflamegpu.HostFunctionCallback):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        pass

class ExitAlways(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
      return pyflamegpu.EXIT;

class DependencyGraphTest(TestCase):

    agent_fn1 = """FLAMEGPU_AGENT_FUNCTION(agent_fn1, flamegpu::MsgBruteForce, flamegpu::MsgBruteForce) {
        // do nothing
        return flamegpu::ALIVE
    }"""
    agent_fn2 = """FLAMEGPU_AGENT_FUNCTION(agent_fn2, flamegpu::MsgNone, flamegpu::MsgNone) {
        // do nothing
        return flamegpu::ALIVE
    }"""
    agent_fn3 = """FLAMEGPU_AGENT_FUNCTION(agent_fn3, flamegpu::MsgNone, flamegpu::MsgNone) {
        // do nothing
        return flamegpu::ALIVE
    }"""
    agent_fn4 = """FLAMEGPU_AGENT_FUNCTION(agent_fn4, flamegpu::MsgNone, flamegpu::MsgNone) {
        // do nothing
        return flamegpu::ALIVE
    }"""
    
    host_fn1 = EmptyHostFunc()
    
    host_fn2 = EmptyHostFunc()
    
    host_fn3 = EmptyHostFunc()
    
    exit_always = ExitAlways()
    
    
    
    def test_ValidateEmptyGraph(self):
        graph = pyflamegpu.DependencyGraph()
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            graph.validateDependencyGraph()
        assert e.value.type() == "InvalidDependencyGraph"
    
    def test_ValidateSingleNode(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        assert graph.validateDependencyGraph() == True
    
    def test_ValidateSingleChain(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f2)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        assert graph.validateDependencyGraph() == True
    
    def test_ValidateBranch(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        assert graph.validateDependencyGraph() == True
    
    def test_ValidateCycle(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f2.dependsOn(f3)
        f3.dependsOn(f2)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            graph.validateDependencyGraph()
        assert e.value.type() == "InvalidDependencyGraph"
    
    def test_ValidateRootWithDependencies(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f2)
        graph = _m.getDependencyGraph()
        graph.addRoot(f2)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            graph.validateDependencyGraph()
        assert e.value.type() == "InvalidDependencyGraph"
    
    def test_ConstructLayersSingleChain(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f2)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m)
        assert graph.validateDependencyGraph() == True
        assert _m.getLayersCount() == 3
    
    def test_ConstructLayersRootTwoChildrenConflict(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m)
        assert _m.getLayersCount() == 3
    
    def test_AddHostFunctionAsDependent(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        hf.dependsOn(f)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m)
        assert graph.validateDependencyGraph() == True
    
    def test_AddHostFunctionAsDependency(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        f.dependsOn(hf)
        graph = _m.getDependencyGraph()
        graph.addRoot(hf)
        graph.generateLayers(_m)
        assert graph.validateDependencyGraph() == True
    
    def test_AddSubmodelAsDependent(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
    
        _sm = pyflamegpu.ModelDescription(SUBMODEL_NAME)
        _sm.newAgent(SUBAGENT_NAME)
        _sm.addExitConditionCallback(self.exit_always)
        _smd = _m.newSubModel("sub", _sm)
    
        _smd.dependsOn(f)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m)
        assert graph.validateDependencyGraph() == True
        assert _m.getLayersCount() == 2
    
    def test_AddSubmodelAsDependency(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
    
        _sm = pyflamegpu.ModelDescription(SUBMODEL_NAME)
        _sm.newAgent(SUBAGENT_NAME)
        _sm.addExitConditionCallback(self.exit_always)
        _smd = _m.newSubModel("sub", _sm)
    
        f.dependsOn(_smd)
        graph = _m.getDependencyGraph()
        graph.addRoot(_smd)
        graph.generateLayers(_m)
        assert graph.validateDependencyGraph() == True
    
    def test_DOTDiagramSingleChain(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f2)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m)
        assert graph.validateDependencyGraph() == True
        graph.generateDOTDiagram("singlechain.gv")

        # Check file contents
        dotFile = open('singlechain.gv', 'r')
        dotBuffer = dotFile.read()
        expectedDot = '''digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function3[style = filled, color = red];
    Function1 -> Function2;
    Function2 -> Function3;
}'''
        assert expectedDot == dotBuffer
        dotFile.close()
        # Remove file
        os.remove("singlechain.gv")

    def test_DOTDiagramTwoDependencies(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        assert graph.validateDependencyGraph() == True
        graph.generateDOTDiagram("twodeps.gv")

        # Check file contents
        dotFile = open('twodeps.gv', 'r')
        dotBuffer = dotFile.read()
        expectedDot = '''digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function3[style = filled, color = red];
    Function1 -> Function2;
    Function1 -> Function3;
}'''
        assert expectedDot == dotBuffer
        dotFile.close()
        # Remove file
        os.remove("twodeps.gv")
    
    def test_DOTDiagramDiamond(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f4 = a.newRTCFunction(FUNCTION_NAME4, self.agent_fn4)
        f2.dependsOn(f)
        f3.dependsOn(f)
        f4.dependsOn(f2)
        f4.dependsOn(f3)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        assert graph.validateDependencyGraph() == True
        graph.generateDOTDiagram("diamond.gv")

        # Check file contents
        dotFile = open('diamond.gv', 'r')
        dotBuffer = dotFile.read()
        expectedDot = '''digraph {
    Function1[style = filled, color = red];
    Function2[style = filled, color = red];
    Function4[style = filled, color = red];
    Function3[style = filled, color = red];
    Function4[style = filled, color = red];
    Function1 -> Function2;
    Function2 -> Function4;
    Function1 -> Function3;
    Function3 -> Function4;
}'''
        assert expectedDot == dotBuffer
        dotFile.close()
        # Remove file
        os.remove("diamond.gv")
    
    def test_DOTDiagramHostFunctions(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f4 = a.newRTCFunction(FUNCTION_NAME4, self.agent_fn4)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        hf2 = pyflamegpu.HostFunctionDescription(HOST_FN_NAME2, self.host_fn2)
        f2.dependsOn(f)
        f3.dependsOn(hf)
        f4.dependsOn(f2)
        f4.dependsOn(hf)
        hf2.dependsOn(f3)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.addRoot(hf)
        assert graph.validateDependencyGraph() == True
        graph.generateDOTDiagram("host_functions.gv")

        # Check file contents
        dotFile = open('host_functions.gv', 'r')
        dotBuffer = dotFile.read()
        expectedDot = '''digraph {
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
}'''
        assert expectedDot == dotBuffer
        dotFile.close()
        # Remove file
        os.remove("host_functions.gv")
    
    def test_DOTDiagramAllDependencies(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f4 = a.newRTCFunction(FUNCTION_NAME4, self.agent_fn4)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        hf2 = pyflamegpu.HostFunctionDescription(HOST_FN_NAME2, self.host_fn2)
        _sm = pyflamegpu.ModelDescription(SUBMODEL_NAME)
        _sm.newAgent(SUBAGENT_NAME)
        _sm.addExitConditionCallback(self.exit_always)
        _smd = _m.newSubModel("sub", _sm)
        f2.dependsOn(f)
        f3.dependsOn(hf)
        f4.dependsOn(f2)
        f4.dependsOn(hf)
        hf2.dependsOn(f3)
        _smd.dependsOn(hf2)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.addRoot(hf)
        assert graph.validateDependencyGraph() == True
        graph.generateDOTDiagram("all_dependencies.gv")

        # Check file contents
        dotFile = open('all_dependencies.gv', 'r')
        dotBuffer = dotFile.read()
        expectedDot = '''digraph {
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
}'''
        assert expectedDot == dotBuffer
        dotFile.close()
        # Remove file
        os.remove("all_dependencies.gv")
    
    def test_CorrectLayersAllDependencies(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f4 = a.newRTCFunction(FUNCTION_NAME4, self.agent_fn4)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        hf2 = pyflamegpu.HostFunctionDescription(HOST_FN_NAME2, self.host_fn2)
        _sm = pyflamegpu.ModelDescription(SUBMODEL_NAME)
        _sm.newAgent(SUBAGENT_NAME)
        _sm.addExitConditionCallback(self.exit_always)
        _smd = _m.newSubModel("sub", _sm)
        f2.dependsOn(f)
        f3.dependsOn(hf)
        f4.dependsOn(f2)
        f4.dependsOn(hf)
        hf2.dependsOn(f3)
        _smd.dependsOn(hf2)
        graph = _m.getDependencyGraph()
        graph.addRoot(f)
        graph.addRoot(hf)
        _m.generateLayers()
        expectedLayers = '''--------------------
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

'''
        print (expectedLayers)
        print (graph.getConstructedLayersString())
        assert expectedLayers == graph.getConstructedLayersString()
        assert _m.getLayersCount() == 7

    def test_CorrectLayersConcurrent(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a2 = _m.newAgent(AGENT_NAME2)
        a3 = _m.newAgent(AGENT_NAME3)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a2.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a3.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        hf2 = pyflamegpu.HostFunctionDescription(HOST_FN_NAME2, self.host_fn2)
        f.dependsOn(hf)
        f2.dependsOn(hf)
        hf2.dependsOn(f)
        hf2.dependsOn(f2)
        hf2.dependsOn(f3)

        graph = _m.getDependencyGraph()
        graph.addRoot(f3)
        graph.addRoot(hf)
        _m.generateLayers()
        expectedLayers = '''--------------------
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

'''
        assert expectedLayers == graph.getConstructedLayersString()
        assert _m.getLayersCount() == 4

    def test_InterModelDependency(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)

        _m2 = pyflamegpu.ModelDescription(MODEL_NAME2)
        a2 = _m2.newAgent(AGENT_NAME2)
        f2 = a2.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)

        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            f2.dependsOn(f)
        assert e.value.type() == "InvalidDependencyGraph"

    def test_ModelAlreadyHasLayers(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)

        # Create manual layer
        l = _m.newLayer(LAYER_NAME)
        l.addAgentFunction(f2)

        graph = _m.getDependencyGraph()
        graph.addRoot(f)

        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            _m.generateLayers()
        assert e.value.type() == "InvalidDependencyGraph"