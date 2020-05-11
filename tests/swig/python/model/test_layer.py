import pytest
from unittest import TestCase
from pyflamegpu import *

MODEL_NAME = "Model"
AGENT_NAME = "Agent1"
LAYER_NAME = "Layer1"
FUNCTION_NAME1 = "Function1"
FUNCTION_NAME2 = "Function2"
FUNCTION_NAME3 = "Function3"
FUNCTION_NAME4 = "Function4"
FUNCTION_NAME5 = "Function5"
WRONG_MODEL_NAME = "Model2"
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


class LayerDescriptionTest(TestCase):
    
    agent_fn1 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgNone, MsgNone) {
        # do nothing
        return ALIVE
    }
    """
    
    agent_fn2 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn2, MsgNone, MsgNone) {
        # do nothing
        return ALIVE
    }
    """
    
    agent_fn3 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn3, MsgNone, MsgNone) {
        # do nothing
        return ALIVE
    }
    """
    
    agent_fn4 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn4, MsgNone, MsgNone) {
        # do nothing
        return ALIVE
    }
    """
    
    agent_fn5 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn5, MsgNone, MsgNone) {
        # do nothing
        return ALIVE
    }
    """
    
    # empty host function
    host_fn = EmptyHostFunc()


    def test_agent_function(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a.newState(STATE_NAME)
        a.newState(NEW_STATE_NAME)
        a.newState(WRONG_STATE_NAME)
        a.newState(OTHER_STATE_NAME)
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f1.setInitialState(STATE_NAME)
        f1.setEndState(STATE_NAME)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f2.setInitialState(NEW_STATE_NAME)
        f2.setEndState(NEW_STATE_NAME)
        f3 = a.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        f3.setInitialState(WRONG_STATE_NAME)
        f3.setEndState(WRONG_STATE_NAME)
        f4 = a.newRTCFunction(FUNCTION_NAME4, self.agent_fn4)
        f4.setInitialState(OTHER_STATE_NAME)
        f4.setEndState(OTHER_STATE_NAME)
        l = _m.newLayer(LAYER_NAME)
        
        assert l.getAgentFunctionsCount() == 0
        # Add by fn
        l.addAgentFunction(f1)
        assert l.getAgentFunctionsCount() == 1
        # Add by fn description
        l.addAgentFunction(f2)
        assert l.getAgentFunctionsCount() == 2
        # Add by string
        l.addAgentFunction(FUNCTION_NAME3)
        assert l.getAgentFunctionsCount() == 3
        # Add by string literal (char*)
        l.addAgentFunction(FUNCTION_NAME4)
        assert l.getAgentFunctionsCount() == 4

        # Cannot add function not attached to an agent
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f1)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f4)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME5)
        assert e.value.type() == "InvalidAgentFunc"
        assert l.getAgentFunctionsCount() == 4

        # Cannot add duplicate function variable with same name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)  # has to be added by function description
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f3)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME4)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME1)
        assert e.value.type() == "InvalidAgentFunc"
        assert l.getAgentFunctionsCount() == 4


    def test_host_function(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        l = _m.newLayer(LAYER_NAME)
        assert l.getHostFunctionCallbackCount() == 0
        l.addHostFunctionCallback(self.host_fn)
        assert l.getHostFunctionCallbackCount() == 1
        # Cannot create function with same name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addHostFunctionCallback(self.host_fn)
        assert e.value.type() == "InvalidHostFunc"

    def test_agent_function_wrong_model(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        _m2 = pyflamegpu.ModelDescription(WRONG_MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a2 = _m2.newAgent(AGENT_NAME)
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a2.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        l = _m.newLayer(LAYER_NAME)

        l.addAgentFunction(f1)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)
        assert e.value.type() == "DifferentModel"


    def test_same_agent_and_state1(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn3)
        l = _m.newLayer()
        # Both have agent in default state
        l.addAgentFunction(f1)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME2)
        assert e.value.type() == "InvalidAgentFunc"


    def test_same_agent_and_state2(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a.newState(STATE_NAME)
        a.newState(NEW_STATE_NAME)
        a.newState(WRONG_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        f.setInitialState(STATE_NAME)
        f.setEndState(NEW_STATE_NAME)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn3)
        f2.setInitialState(STATE_NAME)
        f2.setEndState(NEW_STATE_NAME)
        l = _m.newLayer()
        # Both have STATE_NAME:NEW_STATE_NAME state
        l.addAgentFunction(f)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME2)
        assert e.value.type() == "InvalidAgentFunc"

    def test_same_agent_and_state3(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a.newState(STATE_NAME)
        a.newState(NEW_STATE_NAME)
        a.newState(WRONG_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        f.setInitialState(STATE_NAME)
        f.setEndState(NEW_STATE_NAME)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn3)
        f2.setInitialState(STATE_NAME)
        f2.setEndState(WRONG_STATE_NAME)
        l = _m.newLayer()
        # Both share initial state
        l.addAgentFunction(f)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME2)
        assert e.value.type() == "InvalidAgentFunc"

    def test_same_agent_and_state4(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a.newState(STATE_NAME)
        a.newState(NEW_STATE_NAME)
        a.newState(WRONG_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        f.setInitialState(STATE_NAME)
        f.setEndState(NEW_STATE_NAME)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn3)
        f2.setInitialState(WRONG_STATE_NAME)
        f2.setEndState(NEW_STATE_NAME)
        l = _m.newLayer()
        # start matches end and vice versa
        l.addAgentFunction(f)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME2)
        assert e.value.type() == "InvalidAgentFunc"

    def test_same_agent_and_state5(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a.newState(STATE_NAME)
        a.newState(NEW_STATE_NAME)
        a.newState(WRONG_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        f.setInitialState(STATE_NAME)
        f.setEndState(NEW_STATE_NAME)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn3)
        f2.setInitialState(NEW_STATE_NAME)
        f2.setEndState(WRONG_STATE_NAME)
        l = _m.newLayer()
        # end matches start state
        l.addAgentFunction(f)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME2)
        assert e.value.type() == "InvalidAgentFunc"

    def test_same_agent_and_state6(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = _m.newAgent(AGENT_NAME)
        a.newState(STATE_NAME)
        a.newState(NEW_STATE_NAME)
        a.newState(WRONG_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        f.setInitialState(STATE_NAME)
        f.setEndState(NEW_STATE_NAME)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn3)
        f2.setInitialState(WRONG_STATE_NAME)
        f2.setEndState(NEW_STATE_NAME)
        l = _m.newLayer()
        # start matches end state
        l.addAgentFunction(f)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(f2)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            l.addAgentFunction(FUNCTION_NAME2)
        assert e.value.type() == "InvalidAgentFunc"
    
