import pytest
from unittest import TestCase
from pyflamegpu import *



AGENT_COUNT = 1000
MODEL_NAME = "Model"
AGENT_NAME = "Agent"
FUNCTION_NAME1 = "Function1"
FUNCTION_NAME2 = "Function2"
STATE1 = "Start"
STATE2 = "End"
STATE3 = "End2"


class TestAgentFunctionConditions(TestCase):
   
    NullFn1 = """
    FLAMEGPU_AGENT_FUNCTION(NullFn1, MsgNone, MsgNone) {
        FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") + 1);
        FLAMEGPU->setVariable<int, 4>("y", 0, 3);
        FLAMEGPU->setVariable<int, 4>("y", 1, 4);
        FLAMEGPU->setVariable<int, 4>("y", 2, 5);
        FLAMEGPU->setVariable<int, 4>("y", 3, 6);
        return ALIVE;
    }"""
    
    NullFn2 = """
    FLAMEGPU_AGENT_FUNCTION(NullFn2, MsgNone, MsgNone) {
        FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") - 1);
        FLAMEGPU->setVariable<int, 4>("y", 0, 23);
        FLAMEGPU->setVariable<int, 4>("y", 1, 24);
        FLAMEGPU->setVariable<int, 4>("y", 2, 25);
        FLAMEGPU->setVariable<int, 4>("y", 3, 26);
        return ALIVE;
    }"""
    
    Condition1 = """
    FLAMEGPU_AGENT_FUNCTION_CONDITION(Condition1) {
        return FLAMEGPU->getVariable<int>("x") == 1;
    }
    """
    
    Condition2 = """
    FLAMEGPU_AGENT_FUNCTION_CONDITION(Condition2) {
        return FLAMEGPU->getVariable<int>("x") != 1;
    }
    """
    
    AllFail = """
    FLAMEGPU_AGENT_FUNCTION_CONDITION(AllFail) {
        return false;
    }
    """
   
    def test_split_agents(self): 
        ARRAY_REFERENCE =  (13, 14, 15, 16) 
        ARRAY_REFERENCE2 =  (23, 24, 25, 26) 
        ARRAY_REFERENCE3 =  (3, 4, 5, 6) 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableArrayInt("y", 4)
        a.newState(STATE1)
        a.newState(STATE2)
        a.newState(STATE3)
        af1 = a.newRTCFunction(FUNCTION_NAME1, self.NullFn1)
        af1.setInitialState(STATE1)
        af1.setEndState(STATE2)
        af1.setRTCFunctionCondition(self.Condition1)
        af2 = a.newRTCFunction(FUNCTION_NAME2, self.NullFn2)
        af2.setInitialState(STATE1)
        af2.setEndState(STATE3)
        af2.setRTCFunctionCondition(self.Condition2)
        l1 = m.newLayer()
        l1.addAgentFunction(af1)
        l2 = m.newLayer()
        l2.addAgentFunction(af2)
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT * 2)
        for i in range(AGENT_COUNT*2): 
            ai = pop[i]
            val = i % 2  # 0, 1, 0, 1, etc
            ai.setVariableInt("x", val)
            ai.setVariableArrayInt("y", ARRAY_REFERENCE)
        
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop, STATE1)
        c.step()
        pop_STATE1 = pyflamegpu.AgentVector(a)
        pop_STATE2 = pyflamegpu.AgentVector(a)
        pop_STATE3 = pyflamegpu.AgentVector(a)
        c.getPopulationData(pop_STATE1, STATE1)
        c.getPopulationData(pop_STATE2, STATE2)
        c.getPopulationData(pop_STATE3, STATE3)
        assert len(pop_STATE1) == 0
        assert len(pop_STATE2) == AGENT_COUNT
        assert len(pop_STATE3) == AGENT_COUNT
        # Check val of agents in STATE2 state
        for ai in pop_STATE2:
            assert ai.getVariableInt("x") == 2
            test = ai.getVariableArrayInt("y")
            assert test == ARRAY_REFERENCE3
        
        # Check val of agents in STATE3 state
        for ai in pop_STATE3:
            assert ai.getVariableInt("x") == -1
            test = ai.getVariableArrayInt("y")
            assert test == ARRAY_REFERENCE2
        
    
    def test_all_disabled(self): 
        # Tests for a bug created by #342, fixed by #343
        # If all agents got disabled by an agent function condition, they would not get re-enabled
        # This would lead to an exception later on
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableArrayInt("y", 4)
        af1 = a.newRTCFunction(FUNCTION_NAME1, self.NullFn1)
        af1.setRTCFunctionCondition(self.AllFail)
        af2 = a.newRTCFunction(FUNCTION_NAME2, self.NullFn2)
        l1 = m.newLayer()
        l1.addAgentFunction(af1)
        l2 = m.newLayer()
        l2.addAgentFunction(af2)
        # Create a bunch of empty agents
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        # should not throw exception
        c.step()
        c.step()
        c.step()
        
