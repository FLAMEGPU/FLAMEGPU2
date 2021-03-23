import pytest
from unittest import TestCase
from pyflamegpu import *

MODEL_NAME = "Model"
WRONG_MODEL_NAME = "Model2"
SUBMODEL_NAME = "SubModel1"
SUBAGENT_NAME = "SubAgent1"
AGENT_NAME = "Agent1"
AGENT_NAME2 = "Agent2"
AGENT_NAME3 = "Agent3"
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

    agent_fn1 = """FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgBruteForce, MsgBruteForce) {
        // do nothing
        return ALIVE
    }"""
    agent_fn2 = """FLAMEGPU_AGENT_FUNCTION(agent_fn2, MsgNone, MsgNone) {
        // do nothing
        return ALIVE
    }"""
    agent_fn3 = """FLAMEGPU_AGENT_FUNCTION(agent_fn3, MsgNone, MsgNone) {
        // do nothing
        return ALIVE
    }"""
    agent_fn4 = """FLAMEGPU_AGENT_FUNCTION(agent_fn4, MsgNone, MsgNone) {
        // do nothing
        return ALIVE
    }"""
    
    host_fn1 = EmptyHostFunc()
    
    host_fn2 = EmptyHostFunc()
    
    host_fn3 = EmptyHostFunc()
    
    exit_always = ExitAlways()
    
    
    
    def test_ValidateEmptyGraph(self):
        graph = pyflamegpu.DependencyGraph()
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            graph.validateDependencyGraph()
        assert e.value.type() == "InvalidDependencyGraph"
    
    def test_ValidateSingleNode(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        graph = pyflamegpu.DependencyGraph()
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
        graph = pyflamegpu.DependencyGraph()
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
        graph = pyflamegpu.DependencyGraph()
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
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
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
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f2)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
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
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m) 
    
    def test_ConstructLayersRootTwoChildrenConflict(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f)
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m) 
    
    def test_AddHostFunctionAsDependent(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        hf.dependsOn(f)
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m) 
    
    def test_AddHostFunctionAsDependency(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        hf = pyflamegpu.HostFunctionDescription(HOST_FN_NAME1, self.host_fn1)
        f.dependsOn(hf)
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(hf)
        graph.generateLayers(_m) 
    
    def test_AddSubmodelAsDependent(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
    
        _sm = pyflamegpu.ModelDescription(SUBMODEL_NAME)
        _sm.newAgent(SUBAGENT_NAME)
        _sm.addExitConditionCallback(self.exit_always)
        _smd = _m.newSubModel("sub", _sm)
    
        _smd.dependsOn(f)
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m) 
    
    def test_AddSubmodelAsDependency(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
    
        _sm = pyflamegpu.ModelDescription(SUBMODEL_NAME)
        _sm.newAgent(SUBAGENT_NAME)
        _sm.addExitConditionCallback(self.exit_always)
        _smd = _m.newSubModel("sub", _sm)
    
        f.dependsOn(_smd)
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(_smd)
        graph.generateLayers(_m) 
    
    def test_DOTDiagramSingleChain(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f2)
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.generateLayers(_m)
        graph.generateDOTDiagram("singlechain.gv")
    
    def test_DOTDiagramTwoDependencies(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f2.dependsOn(f)
        f3.dependsOn(f)
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.generateDOTDiagram("twodeps.gv")
    
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
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.generateDOTDiagram("diamond.gv")
    
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
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.addRoot(hf)
        graph.generateDOTDiagram("host_functions.gv")
    
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
        graph = pyflamegpu.DependencyGraph()
        graph.addRoot(f)
        graph.addRoot(hf)
        graph.generateDOTDiagram("all_dependencies.gv")
    