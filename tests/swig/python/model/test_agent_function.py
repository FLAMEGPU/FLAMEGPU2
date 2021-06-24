import pytest
from unittest import TestCase
from pyflamegpu import *


MODEL_NAME = "Model"
WRONG_MODEL_NAME = "Model2"
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
STATE_NAME = "State1"
NEW_STATE_NAME = "State2"
WRONG_STATE_NAME = "State3"
OTHER_STATE_NAME = "State4"


class AgentFunctionDescriptionTest(TestCase):

    agent_fn1 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn1, flamegpu::MsgBruteForce, flamegpu::MsgBruteForce) {
        # do nothing
        return flamegpu::ALIVE
    }
    """
    
    agent_fn2 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn2, flamegpu::MsgNone, flamegpu::MsgNone) {
        # do nothing
        return flamegpu::ALIVE
    }
    """
    
    agent_fn3 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn3, flamegpu::MsgNone, flamegpu::MsgNone) {
        # do nothing
        return flamegpu::ALIVE
    }
    """

    def test_initial_state(self):
        m = pyflamegpu.ModelDescription("test_initial_state")
        a = m.newAgent(AGENT_NAME)
        a2 = m.newAgent(AGENT_NAME2)
        a3 = m.newAgent(AGENT_NAME3)
        a2.newState(STATE_NAME)
        a3.newState("default")
        a2.newState(NEW_STATE_NAME)
        a3.newState(NEW_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a2.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a3.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        # Initial state begins whatever agent's initial state is
        assert f.getInitialState() == a.getInitialState()
        assert f2.getInitialState() == a2.getInitialState()
        assert f3.getInitialState() == a3.getInitialState()
        # Can change the initial state
        f2.setInitialState(NEW_STATE_NAME)
        f3.setInitialState(NEW_STATE_NAME)
        # Returned value is same
        assert f2.getInitialState() == NEW_STATE_NAME
        assert f3.getInitialState() == NEW_STATE_NAME
        # Replacing agent's default state will replace their initial state
        a.newState(NEW_STATE_NAME)
        assert f.getInitialState() == NEW_STATE_NAME
        # Can't set state to one not held by parent agent
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidStateName exception
            f.setInitialState(WRONG_STATE_NAME)
        assert e.value.type() == "InvalidStateName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidStateName exception
            f2.setInitialState(WRONG_STATE_NAME)
        assert e.value.type() == "InvalidStateName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidStateName exception
            f3.setInitialState(WRONG_STATE_NAME)
        assert e.value.type() == "InvalidStateName"
    
    def test_end_state(self):
        m = pyflamegpu.ModelDescription("test_end_state")
        a = m.newAgent(AGENT_NAME)
        a2 = m.newAgent(AGENT_NAME2)
        a3 = m.newAgent(AGENT_NAME3)
        a2.newState(STATE_NAME)
        a3.newState("default")
        a2.newState(NEW_STATE_NAME)
        a3.newState(NEW_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        f2 = a2.newRTCFunction(FUNCTION_NAME2, self.agent_fn2)
        f3 = a3.newRTCFunction(FUNCTION_NAME3, self.agent_fn3)
        # End state begins whatever agent's end state is
        assert f.getEndState() == a.getInitialState()
        assert f2.getEndState()== a2.getInitialState()
        assert f3.getEndState()== a3.getInitialState()
        # Can change the end state
        f2.setEndState(NEW_STATE_NAME)
        f3.setEndState(NEW_STATE_NAME)
        # Returned value is same
        assert f2.getEndState() == NEW_STATE_NAME
        assert f3.getEndState() == NEW_STATE_NAME
        # Replacing agent's default state will replace their end state
        a.newState(NEW_STATE_NAME)
        assert f.getEndState() == NEW_STATE_NAME
        # Can't set state to one not held by parent agent
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidStateName exception
            f.setEndState(WRONG_STATE_NAME)
        assert e.value.type() == "InvalidStateName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidStateName exception
            f2.setEndState(WRONG_STATE_NAME)
        assert e.value.type() == "InvalidStateName"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidStateName exception
            f3.setEndState(WRONG_STATE_NAME)
        assert e.value.type() == "InvalidStateName"

    def test_message_input(self):
        m = pyflamegpu.ModelDescription("test_message_input")
        a = m.newAgent(AGENT_NAME)
        msg1 = m.newMessageBruteForce(MESSAGE_NAME1)
        msg2 = m.newMessageBruteForce(MESSAGE_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Begins empty
        assert f.hasMessageInput() == False
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.getMessageInput()
        assert e.value.type() == "OutOfBoundsException"
        # Can be set
        f.setMessageInput(msg1)
        assert f.hasMessageInput()
        # Returns the expected value
        assert f.getMessageInput() == msg1
        # Can be updated
        f.setMessageInput(msg2)
        assert f.hasMessageInput()
        # Returns the expected value
        assert f.getMessageInput() == msg2

    def test_message_output(self):
        m = pyflamegpu.ModelDescription("test_message_output")
        a = m.newAgent(AGENT_NAME)
        msg1 = m.newMessageBruteForce(MESSAGE_NAME1)
        msg2 = m.newMessageBruteForce(MESSAGE_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Begins empty
        assert f.hasMessageOutput() == False
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.getMessageOutput()
        assert e.value.type() == "OutOfBoundsException"
        # Can be set
        f.setMessageOutput(msg1)
        assert f.hasMessageOutput()
        # Returns the expected value
        assert f.getMessageOutput() == msg1
        # Can be updated
        f.setMessageOutput(msg2)
        assert f.hasMessageOutput()
        # Returns the expected value
        assert f.getMessageOutput() == msg2

    def test_message_output_optional(self):
        m = pyflamegpu.ModelDescription("test_message_output_optional")
        a = m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Begins empty
        assert f.hasMessageOutput() == False
        assert f.getMessageOutputOptional() == False
        # Can be updated
        f.setMessageOutputOptional(True)
        assert f.getMessageOutputOptional()
        # assert f.MessageOutputOptional() # Mutable version not sensible in Python
        f.setMessageOutputOptional(False)
        assert f.getMessageOutputOptional() == False
        # assert f.MessageOutputOptional() == False # Mutable version not sensible in Python

    def test_agent_output(self):
        m = pyflamegpu.ModelDescription("test_agent_output")
        a = m.newAgent(AGENT_NAME)
        a2 = m.newAgent(AGENT_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Begins empty
        assert f.hasAgentOutput() == False
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.getAgentOutput()
        assert e.value.type() == "OutOfBoundsException"
        # Can be set
        f.setAgentOutput(a)
        assert f.hasAgentOutput()
        # Returns the expected value
        assert f.getAgentOutput() == a
        # Can be updated
        f.setAgentOutput(a2)
        assert f.hasAgentOutput()
        # Returns the expected value
        assert f.getAgentOutput() == a2


    def test_agent_output_state(self):
        m = pyflamegpu.ModelDescription("test_agent_output_state")
        a = m.newAgent(AGENT_NAME)
        a2 = m.newAgent(AGENT_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Can't set it to a state that doesn't exist
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setAgentOutput(a, "wrong")
        assert e.value.type() == "InvalidStateName"
        a.newState("a")
        a.newState("b")
        a2.newState("c")
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setAgentOutput(a, "c")
        assert e.value.type() == "InvalidStateName"
        # Can set it to a valid state though
        f.setAgentOutput(a, "a")
        # Can't set it to default if default not a state
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setAgentOutput(a)
        assert e.value.type() == "InvalidStateName"
        # Returns the expected value
        assert f.getAgentOutputState() == "a"
        # Can be updated
        f.setAgentOutput(a, "b")
        assert f.hasAgentOutput()
        # Returns the expected value
        assert f.getAgentOutputState() == "b"
        # Can be updated different agent
        f.setAgentOutput(a2, "c")
        # Returns the expected value
        assert f.getAgentOutputState() == "c"

    def test_allow_agent_death(self):
        m = pyflamegpu.ModelDescription("test_allow_agent_death")
        a = m.newAgent(AGENT_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Begins disabled
        assert f.getAllowAgentDeath() == False
        # Can be updated
        f.setAllowAgentDeath(True)
        assert f.getAllowAgentDeath()
        f.setAllowAgentDeath(False)
        assert f.getAllowAgentDeath() == False


    def test_message_input_wrong_model(self):
        m = pyflamegpu.ModelDescription("test_message_input_wrong_model")
        m2 = pyflamegpu.ModelDescription(WRONG_MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        msg1 = m.newMessageBruteForce(MESSAGE_NAME1)
        msg2 = m2.newMessageBruteForce(MESSAGE_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setMessageInput(msg2)
        assert e.value.type() == "DifferentModel"
        f.setMessageInput(msg1)

    def test_message_output_wrong_model(self):
        m = pyflamegpu.ModelDescription("test_message_output_wrong_model")
        m2 = pyflamegpu.ModelDescription(WRONG_MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        msg1 = m.newMessageBruteForce(MESSAGE_NAME1)
        msg2 = m2.newMessageBruteForce(MESSAGE_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setMessageOutput(msg2)
        assert e.value.type() == "DifferentModel"
        f.setMessageOutput(msg1)

    def test_agent_output_wrong_model(self):
        m = pyflamegpu.ModelDescription("test_agent_output_wrong_model")
        m2 = pyflamegpu.ModelDescription(WRONG_MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        a2 = m2.newAgent(AGENT_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setAgentOutput(a2)
        assert e.value.type() == "DifferentModel"
        f.setAgentOutput(a)

    def test_message_input_output(self):
        m = pyflamegpu.ModelDescription("test_message_input_output")
        a = m.newAgent(AGENT_NAME)
        msg1 = m.newMessageBruteForce(MESSAGE_NAME1)
        msg2 = m.newMessageBruteForce(MESSAGE_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Cannot bind same message to input and output
        f.setMessageInput(msg1)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setMessageOutput(msg1)
        assert e.value.type() == "InvalidMessageName"
        f.setMessageOutput(msg2)

    def test_message_output_input(self):
        m = pyflamegpu.ModelDescription("test_message_output_input")
        a = m.newAgent(AGENT_NAME)
        msg1 = m.newMessageBruteForce(MESSAGE_NAME1)
        msg2 = m.newMessageBruteForce(MESSAGE_NAME2)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1)
        # Cannot bind same message to input and output
        f.setMessageOutput(msg1)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f.setMessageInput(msg1)
        assert e.value.type() == "InvalidMessageName"
        f.setMessageInput(msg2)

    def test_same_agent_and_state_in_layer(self):
        m = pyflamegpu.ModelDescription("test_message_output_input")
        a = m.newAgent(AGENT_NAME)
        a.newState(STATE_NAME)
        a.newState(NEW_STATE_NAME)
        a.newState(WRONG_STATE_NAME)
        a.newState(OTHER_STATE_NAME)
        f = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn2)
        f.setInitialState(STATE_NAME)
        f.setEndState(NEW_STATE_NAME)
        f2 = a.newRTCFunction(FUNCTION_NAME2, self.agent_fn3)
        f2.setInitialState(WRONG_STATE_NAME)
        f2.setEndState(OTHER_STATE_NAME)
        l = m.newLayer()
        # start matches end state
        l.addAgentFunction(f)
        l.addAgentFunction(f2)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f2.setInitialState(STATE_NAME)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f2.setInitialState(NEW_STATE_NAME)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f2.setEndState(STATE_NAME)
        assert e.value.type() == "InvalidAgentFunc"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e: 
            f2.setEndState(NEW_STATE_NAME)
        assert e.value.type() == "InvalidAgentFunc"

