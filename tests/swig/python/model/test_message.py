import pytest
from unittest import TestCase
from pyflamegpu import *

MODEL_NAME = "Model"
MESSAGE_NAME1 = "Message1"
VARIABLE_NAME1 = "Var1"
VARIABLE_NAME2 = "Var2"
VARIABLE_NAME3 = "Var3"
VARIABLE_NAME4 = "Var4"

class MessageDescriptionTest(TestCase):

    def test_variables(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        m = _m.newMessageBruteForce(MESSAGE_NAME1)
        assert m.hasVariable(VARIABLE_NAME1) == False
        assert m.hasVariable(VARIABLE_NAME2) == False
        assert m.getVariablesCount() == 0
        m.newVariableFloat(VARIABLE_NAME1)
        assert m.getVariablesCount() == 1
        m.newVariableInt16(VARIABLE_NAME2)
        assert m.getVariablesCount() == 2
        # Cannot create variable with same name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m.newVariableInt64(VARIABLE_NAME1)
        assert e.value.type() == "InvalidMessageVar"
        # Variable have the right name
        assert m.hasVariable(VARIABLE_NAME1)
        assert m.hasVariable(VARIABLE_NAME2)
        # Returned variable data is same
        assert m.getVariableSize(VARIABLE_NAME1) == pyflamegpu.FloatType.size()
        assert m.getVariableSize(VARIABLE_NAME2) == pyflamegpu.Int16Type.size()
        assert pyflamegpu.typeName(m.getVariableType(VARIABLE_NAME1)) == pyflamegpu.FloatType.typeName()
        assert pyflamegpu.typeName(m.getVariableType(VARIABLE_NAME2)) == pyflamegpu.Int16Type.typeName()

    def test_variables_array(self):
        _m = pyflamegpu.ModelDescription(MODEL_NAME)
        m = _m.newMessageBruteForce(MESSAGE_NAME1)
        assert m.hasVariable(VARIABLE_NAME1) == False
        assert m.hasVariable(VARIABLE_NAME2) == False
        assert m.getVariablesCount() == 0
        m.newVariableFloat(VARIABLE_NAME1)
        assert m.getVariablesCount() == 1
        m.newVariableInt16(VARIABLE_NAME3)
        assert m.getVariablesCount() == 2
        m.newVariableInt16(VARIABLE_NAME2)
        assert m.getVariablesCount() == 3
        # Cannot create variable with same name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m.newVariableInt16(VARIABLE_NAME1)
        assert e.value.type() == "InvalidMessageVar"
        assert m.hasVariable(VARIABLE_NAME1)
        assert m.hasVariable(VARIABLE_NAME2)
        # Returned variable data is same
        assert m.getVariableSize(VARIABLE_NAME1) == pyflamegpu.FloatType.size()
        assert m.getVariableSize(VARIABLE_NAME2) == pyflamegpu.Int16Type.size()
        assert pyflamegpu.typeName(m.getVariableType(VARIABLE_NAME1)) == pyflamegpu.FloatType.typeName()
        assert pyflamegpu.typeName(m.getVariableType(VARIABLE_NAME2)) == pyflamegpu.Int16Type.typeName()


    NoInput = """
    FLAMEGPU_AGENT_FUNCTION(NoInput, flamegpu::MsgNone, flamegpu::MsgSpatial3D) {
        return flamegpu::ALIVE
    }
    """
    
    NoOutput = """
    FLAMEGPU_AGENT_FUNCTION(NoOutput, flamegpu::MsgSpatial2D, flamegpu::MsgNone) {
        return flamegpu::ALIVE
    }
    """

    def test_correct_message_type_bound1(self):
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent("foo")
        fo = a.newRTCFunction("bar", self.NoInput)
        lo = m.newLayer("foo2")
        lo.addAgentFunction(fo)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            c = pyflamegpu.CUDASimulation(m)
        assert e.value.type() == "InvalidMessageType"

    def test_correct_message_type_bound2(self):
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent("foo")
        fo = a.newRTCFunction("bar", self.NoOutput)
        lo = m.newLayer("foo2")
        lo.addAgentFunction(fo)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            c = pyflamegpu.CUDASimulation(m)
        assert e.value.type() == "InvalidMessageType"

    def test_correct_message_type_bound3(self):
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent("foo")
        fo = a.newRTCFunction("bar", self.NoInput)
        md = m.newMessageBruteForce("foo2")
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            fo.setMessageOutput(md)
        assert e.value.type() == "InvalidMessageType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            fo.setMessageInput(md)
        assert e.value.type() == "InvalidMessageType"

    def test_correct_message_type_bound4(self):
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent("foo")
        fo = a.newRTCFunction("bar", self.NoOutput)
        md = m.newMessageBruteForce("foo2")
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            fo.setMessageOutput(md)
        assert e.value.type() == "InvalidMessageType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            fo.setMessageInput(md)
        assert e.value.type() == "InvalidMessageType"

