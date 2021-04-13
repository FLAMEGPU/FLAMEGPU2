import pytest
from unittest import TestCase
from pyflamegpu import *


AGENT_NAME1 = "Agent1"
AGENT_NAME2 = "Agent2"
VARIABLE_NAME1 = "Var1"
VARIABLE_NAME2 = "Var2"
VARIABLE_NAME3 = "Var3"
VARIABLE_NAME4 = "Var4"
FUNCTION_NAME1 = "Func1"
FUNCTION_NAME2 = "Func2"
STATE_NAME1 = "State1"
STATE_NAME2 = "State2"


AGENT_COUNT = 100

class AgentDescriptionTest(TestCase):

    agent_fn1 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgNone, MsgNone) {
        // do nothing
        return ALIVE
    }
    """
    agent_fn2 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn2, MsgNone, MsgNone) {
        // do nothing
        return ALIVE
    }
    """

    def test_functions(self):
        m = pyflamegpu.ModelDescription("test_functions")
        a = m.newAgent(AGENT_NAME1)
        assert a.hasFunction(FUNCTION_NAME1) == False
        assert a.hasFunction(FUNCTION_NAME2) == False
        assert a.getFunctionsCount() == 0
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        assert a.getFunctionsCount() == 1
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        assert a.getFunctionsCount() == 2
        # Cannot create function with same name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentFunc exception
            a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentFunc exception
            a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentFunc exception
            a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        assert e.value.type() == "InvalidAgentFunc"
        # Functions have the right name
        assert a.hasFunction(FUNCTION_NAME1) == True
        assert a.hasFunction(FUNCTION_NAME2) == True
        # Returned function data is same
        assert f1 == a.getFunction(FUNCTION_NAME1)
        assert f2 ==  a.getFunction(FUNCTION_NAME2)
        assert f1 == a.Function(FUNCTION_NAME1)
        assert f2 == a.Function(FUNCTION_NAME2)
        assert f1.getName() == FUNCTION_NAME1
        assert f2.getName() == FUNCTION_NAME2


    def test_variables(self):
        m = pyflamegpu.ModelDescription("test_variables")
        a = m.newAgent(AGENT_NAME1)
        assert a.hasVariable(VARIABLE_NAME1) == False
        assert a.hasVariable(VARIABLE_NAME2) == False
        # When created, agent has 1 internal variable _id
        assert a.getVariablesCount() == 1
        a.newVariableFloat(VARIABLE_NAME1)
        assert a.getVariablesCount() == 2
        a.newVariableInt16(VARIABLE_NAME2)
        assert a.getVariablesCount() == 3
        a.newVariableID("ID")
        assert a.getVariablesCount() == 4
        # Cannot create variable with same name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            a.newVariableInt64(VARIABLE_NAME1)
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            a.newVariableArrayInt64(VARIABLE_NAME1, 3)
        assert e.value.type() == "InvalidAgentVar"
        # Variable have the right name
        assert a.hasVariable(VARIABLE_NAME1)
        assert a.hasVariable(VARIABLE_NAME2)
        # Returned variable data is same
        assert a.getVariableLength(VARIABLE_NAME1) == 1
        assert a.getVariableLength(VARIABLE_NAME2) == 1
        assert a.getVariableSize(VARIABLE_NAME1) == pyflamegpu.FloatType.size()
        assert a.getVariableSize(VARIABLE_NAME2) == pyflamegpu.Int16Type.size() 
        # comparing type index requires conversation to string (this will not be demangled in unix but it doesn't matter)
        assert pyflamegpu.typeName(a.getVariableType(VARIABLE_NAME1)) ==  pyflamegpu.FloatType.typeName()
        assert pyflamegpu.typeName(a.getVariableType(VARIABLE_NAME2)) ==  pyflamegpu.Int16Type.typeName()

    def test_variables_array(self):
        m = pyflamegpu.ModelDescription("test_variables_array")
        a = m.newAgent(AGENT_NAME1)
        assert a.hasVariable(VARIABLE_NAME1) == False
        assert a.hasVariable(VARIABLE_NAME2) == False
        # When created, agent has 1 internal variable _id
        assert a.getVariablesCount() == 1
        a.newVariableArrayFloat(VARIABLE_NAME1, 2)
        assert a.getVariablesCount() == 2
        a.newVariableInt16(VARIABLE_NAME2, 1)
        assert a.getVariablesCount() == 3
        a.newVariableArrayInt16(VARIABLE_NAME3, 32)
        assert a.getVariablesCount() == 4
        a.newVariableArrayID("IDs", 5)
        assert a.getVariablesCount() == 5
        # Cannot create variable with same name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            a.newVariableInt64(VARIABLE_NAME1)
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            a.newVariableInt64(VARIABLE_NAME1, 0)   # with default arg
        assert e.value.type() == "InvalidAgentVar"
        assert a.hasVariable(VARIABLE_NAME1)
        assert a.hasVariable(VARIABLE_NAME2)
        assert a.hasVariable(VARIABLE_NAME3)
        # Returned variable data is same
        assert a.getVariableLength(VARIABLE_NAME1) == 2
        assert a.getVariableLength(VARIABLE_NAME2) == 1
        assert a.getVariableLength(VARIABLE_NAME3) == 32
        assert a.getVariableSize(VARIABLE_NAME1) == pyflamegpu.FloatType.size()
        assert a.getVariableSize(VARIABLE_NAME2) == pyflamegpu.Int16Type.size() 
        assert a.getVariableSize(VARIABLE_NAME3) == pyflamegpu.Int16Type.size() 
        assert pyflamegpu.typeName(a.getVariableType(VARIABLE_NAME1)) ==  pyflamegpu.FloatType.typeName()
        assert pyflamegpu.typeName(a.getVariableType(VARIABLE_NAME2)) ==  pyflamegpu.Int16Type.typeName()
        assert pyflamegpu.typeName(a.getVariableType(VARIABLE_NAME3)) ==  pyflamegpu.Int16Type.typeName()

    def test_states(self):
        m = pyflamegpu.ModelDescription("test_states")
        a = m.newAgent(AGENT_NAME1)
        assert a.hasState(STATE_NAME1) == False
        assert a.hasState(STATE_NAME2) == False
        assert a.getStatesCount() == 1  # Initially just default state
        a.newState(STATE_NAME1)
        assert a.getStatesCount() == 1  # Remains 1 state after first set
        a.newState(STATE_NAME2)
        assert a.getStatesCount() == 2
        # Cannot create state with same name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidStateName exception
            a.newState(STATE_NAME1)
        assert e.value.type() == "InvalidStateName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidStateName exception
            a.newState(STATE_NAME2)
        assert e.value.type() == "InvalidStateName"
        # States have the right name
        assert a.hasState(STATE_NAME1)
        assert a.hasState(STATE_NAME2)

    

    def test_initial_state1(self):
        m = pyflamegpu.ModelDescription("test_initial_state1")
        a = m.newAgent(AGENT_NAME1)
        # Initial state starts out default (hard coded to avoid requiring wrap of ModelData)
        assert a.getInitialState() == "default"
        # Initial state changes when first state added
        a.newState(STATE_NAME1)
        assert a.getInitialState() == STATE_NAME1
        # Initial state does not change when next state added
        a.newState(STATE_NAME2)
        assert a.getInitialState() == STATE_NAME1

    def test_initial_state2(self):
        m = pyflamegpu.ModelDescription("test_initial_state2")
        a = m.newAgent(AGENT_NAME1)
        # Initial state starts out default
        assert a.getInitialState() == "default"
        # Initial state changes when first state added
        a.newState("default")
        assert a.getStatesCount() == 1  # Remains 1 state after first set
        assert a.getInitialState() ==  "default"
        # Initial state does not change when next state added
        a.newState(STATE_NAME2)
        assert a.getInitialState() == "default"
        assert a.getStatesCount() == 2  # Increases to 2 state

    def test_agent_outputs(self):
        m = pyflamegpu.ModelDescription("test_variables_array")
        a = m.newAgent(AGENT_NAME1)
        b = m.newAgent(AGENT_NAME2)
        assert a.getAgentOutputsCount() == 0
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        # Count increases as we set values
        f1.setAgentOutput(a)
        assert a.getAgentOutputsCount() == 1
        f2.setAgentOutput(a)
        assert a.getAgentOutputsCount() == 2
        # Replacing value doesnt break the count
        f2.setAgentOutput(a)
        assert a.getAgentOutputsCount() == 2
        f2.setAgentOutput(b)
        assert a.getAgentOutputsCount() == 1

    def test_reserved_name(self):
        m = pyflamegpu.ModelDescription("test_variables_array")
        a = m.newAgent(AGENT_NAME1)
        # scalar variable tests
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableInt("_")
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableInt("name")
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableInt("state")
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableInt("nAme")
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableInt("sTate")
        assert e.value.type() == "ReservedName"
        # Array versions of the above
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableArrayInt("_", 3)
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableArrayInt("name", 3)
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableArrayInt("state", 3)
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableArrayInt("nAme", 3)
        assert e.value.type() == "ReservedName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # ReservedName exception
            a.newVariableArrayInt("sTate", 3)
        assert e.value.type() == "ReservedName"

