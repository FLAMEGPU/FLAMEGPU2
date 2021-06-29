import pytest
from unittest import TestCase
from pyflamegpu import *

 
MODEL_NAME = "something"
AGENT_NAME1 = "something2"
AGENT_NAME2 = "something3"


class ModelDescriptionTest(TestCase):

    def test_name(self):
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        # Model has the right name
        assert m.getName() == MODEL_NAME

    def test_agent(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        assert m.hasAgent(AGENT_NAME1) == False
        assert m.hasAgent(AGENT_NAME2) == False
        assert m.getAgentsCount() == 0
        a = m.newAgent(AGENT_NAME1)
        assert m.getAgentsCount() == 1
        b = m.newAgent(AGENT_NAME2)
        assert m.getAgentsCount() == 2
        # Cannot create agent with same name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m.newAgent(AGENT_NAME1)
        assert e.value.type() == "InvalidAgentName"
        # Two created agents are different
        assert a != b
        # Agents have the right name
        assert a.getName() == AGENT_NAME1
        assert b.getName() == AGENT_NAME2
        assert m.hasAgent(AGENT_NAME1)
        assert m.hasAgent(AGENT_NAME2)
        # Returned agent is same
        assert a == m.Agent(AGENT_NAME1)
        assert b == m.Agent(AGENT_NAME2)
        assert a == m.getAgent(AGENT_NAME1)
        assert b == m.getAgent(AGENT_NAME2)

    def test_message(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        assert m.hasMessage(AGENT_NAME1) == False
        assert m.hasMessage(AGENT_NAME2) == False
        assert m.getMessagesCount() == 0
        a = m.newMessageBruteForce(AGENT_NAME1)
        assert m.getMessagesCount() == 1
        b = m.newMessageBruteForce(AGENT_NAME2)
        assert m.getMessagesCount() == 2
        # Cannot create message with same name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m.newMessage(AGENT_NAME1)
        assert e.value.type() == "InvalidMessageName"
        # Two created messages are different
        assert a != b
        # Messages have the right name
        assert a.getName() == AGENT_NAME1
        assert b.getName() == AGENT_NAME2
        assert m.hasMessage(AGENT_NAME1)
        assert m.hasMessage(AGENT_NAME2)
        # Returned message is same
        assert a == m.Message(AGENT_NAME1)
        assert b == m.Message(AGENT_NAME2)
        assert a == m.getMessage(AGENT_NAME1)
        assert b == m.getMessage(AGENT_NAME2)

    def test_layer(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        assert m.hasLayer(AGENT_NAME1) == False
        assert m.hasLayer(AGENT_NAME2) == False
        assert m.hasLayer(0) == False
        assert m.hasLayer(1) == False
        assert m.getLayersCount() == 0
        a = m.newLayer(AGENT_NAME1)
        assert m.getLayersCount() == 1
        assert m.hasLayer(0)
        assert m.hasLayer(AGENT_NAME1)
        b = m.newLayer(AGENT_NAME2)
        assert m.getLayersCount() == 2
        # Cannot create layer with same name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            m.newLayer(AGENT_NAME1)
        assert e.value.type() == "InvalidFuncLayerIndx"
        # Two created layers are different
        assert a != b
        # Layers have the right name
        assert a.getName() == AGENT_NAME1
        assert b.getName() == AGENT_NAME2
        assert m.hasLayer(AGENT_NAME1)
        assert m.hasLayer(AGENT_NAME2)
        assert m.hasLayer(0)
        assert m.hasLayer(1)
        # Returned layer is same
        assert a == m.Layer(AGENT_NAME1)
        assert b == m.Layer(AGENT_NAME2)
        assert a == m.Layer(0)
        assert b == m.Layer(1)
        assert a == m.getLayer(AGENT_NAME1)
        assert b == m.getLayer(AGENT_NAME2)
        assert a == m.getLayer(0)
        assert b == m.getLayer(1)
        assert 0 == a.getIndex()
        assert 1 == b.getIndex()

