import pytest
from unittest import TestCase
from pyflamegpu import *


AGENT_COUNT = 10
MODEL_NAME = "Model"
AGENT_NAME = "Agent"
FUNCTION_NAME1 = "Function1"
FUNCTION_NAME2 = "Function2"
START_STATE = "Start"
END_STATE = "End"
END_STATE2 = "End2"
LAYER_NAME1 = "Layer1"
LAYER_NAME2 = "Layer2"

class TestAgentStateTransitions(TestCase):

    AgentGood = """
	FLAMEGPU_AGENT_FUNCTION(AgentGood, MsgNone, MsgNone) {
		FLAMEGPU->setVariable("x", 11);
		FLAMEGPU->setVariable<int, 4>("y", 0, 23);
		FLAMEGPU->setVariable<int, 4>("y", 1, 24);
		FLAMEGPU->setVariable<int, 4>("y", 2, 25);
		FLAMEGPU->setVariable<int, 4>("y", 3, 26);
		return ALIVE;
	}
    """
    
    AgentBad = """
	FLAMEGPU_AGENT_FUNCTION(AgentBad, MsgNone, MsgNone) {
		FLAMEGPU->setVariable("x", 13);
		FLAMEGPU->setVariable<int, 4>("y", 0, 3);
		FLAMEGPU->setVariable<int, 4>("y", 1, 4);
		FLAMEGPU->setVariable<int, 4>("y", 2, 5);
		FLAMEGPU->setVariable<int, 4>("y", 3, 6);
		return ALIVE;
	}
    """
    
    AgentDecrement = """
	FLAMEGPU_AGENT_FUNCTION(AgentDecrement, MsgNone, MsgNone) {
		unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
		FLAMEGPU->setVariable("x", x == 0 ? 0 : x - 1);
		FLAMEGPU->setVariable<int, 4>("z", 0, 23);
		FLAMEGPU->setVariable<int, 4>("z", 1, 24);
		FLAMEGPU->setVariable<int, 4>("z", 2, 25);
		FLAMEGPU->setVariable<int, 4>("z", 3, 26);
		return ALIVE;
	}
    """
    
    AgentNull = """
	FLAMEGPU_AGENT_FUNCTION(AgentNull, MsgNone, MsgNone) {
		FLAMEGPU->setVariable("x", UINT_MAX);
		FLAMEGPU->setVariable<int, 4>("z", 0, 3);
		FLAMEGPU->setVariable<int, 4>("z", 1, 4);
		FLAMEGPU->setVariable<int, 4>("z", 2, 5);
		FLAMEGPU->setVariable<int, 4>("z", 3, 6);
		return ALIVE;
	}
    """
    
    Zero_X = """
	FLAMEGPU_AGENT_FUNCTION_CONDITION(Zero_X) {
		// Agent's only transition when counter reaches zero
		return FLAMEGPU->getVariable<unsigned int>("x") == 0;
	}
    """

    def test_src_0_dest_10(self): 
        ARRAY_REFERENCE =  (13, 14, 15, 16) 
        ARRAY_REFERENCE2 =  (23, 24, 25, 26) 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        a.newState(START_STATE)
        a.newState(END_STATE)
        a.setInitialState(START_STATE)
        a.newVariableInt("x")
        a.newVariableIntArray4("y")
        af1 = a.newRTCFunction(FUNCTION_NAME1, self.AgentGood)
        af1.setInitialState(START_STATE)
        af1.setEndState(END_STATE)
        lo1 = m.newLayer(LAYER_NAME1)
        lo1.addAgentFunction(af1)
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        for i in range(AGENT_COUNT): 
            ai = pop.getNextInstance(START_STATE)
            ai.setVariableInt("x", 12)
            ai.setVariableIntArray4("y", ARRAY_REFERENCE)
        
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        # Step 1, all agents go from Start->End state, and value become 11
        c.step()
        c.getPopulationData(pop)
        assert pop.getCurrentListSize(START_STATE) == 0
        assert pop.getCurrentListSize(END_STATE) == AGENT_COUNT
        for i in range (pop.getCurrentListSize(END_STATE)): 
            ai = pop.getInstanceAt(i, END_STATE)
            assert ai.getVariableInt("x") == 11
            test = ai.getVariableIntArray4("y")
            assert test == ARRAY_REFERENCE2
        
        # Step 2, no agents in start state, nothing changes
        c.step()
        c.getPopulationData(pop)
        assert pop.getCurrentListSize(START_STATE) == 0
        assert pop.getCurrentListSize(END_STATE) == AGENT_COUNT
        for i in range (pop.getCurrentListSize(END_STATE)): 
            ai = pop.getInstanceAt(i, END_STATE)
            assert ai.getVariableInt("x") == 11
            test = ai.getVariableIntArray4("y")
            assert test == ARRAY_REFERENCE2
        
    def test_src_10_dest_0(self): 
        ARRAY_REFERENCE =  (13, 14, 15, 16) 
        ARRAY_REFERENCE2 =  (23, 24, 25, 26) 
        ARRAY_REFERENCE3 =  (3, 4, 5, 6) 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        a.newState(START_STATE)
        a.newState(END_STATE)
        a.newState(END_STATE2)
        a.setInitialState(START_STATE)
        a.newVariableInt("x")
        a.newVariableIntArray4("y")
        af1 = a.newRTCFunction(FUNCTION_NAME1, self.AgentGood)
        af1.setInitialState(START_STATE)
        af1.setEndState(END_STATE)
        af2 = a.newRTCFunction(FUNCTION_NAME2, self.AgentBad)
        af2.setInitialState(END_STATE)
        af2.setEndState(END_STATE2)
        lo1 = m.newLayer(LAYER_NAME1)
        lo2 = m.newLayer(LAYER_NAME2)
        lo1.addAgentFunction(af2)
        lo2.addAgentFunction(af1)
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = pop.getNextInstance(START_STATE)
            ai.setVariableInt("x", 12)
            ai.setVariableIntArray4("y", ARRAY_REFERENCE)
        
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        # Step 1, all agents go from Start->End state, and value become 11
        c.step()
        c.getPopulationData(pop)
        assert pop.getCurrentListSize(START_STATE) == 0
        assert pop.getCurrentListSize(END_STATE) == AGENT_COUNT
        assert pop.getCurrentListSize(END_STATE2) == 0
        for i in range(pop.getCurrentListSize(END_STATE)): 
            ai = pop.getInstanceAt(i, END_STATE)
            assert ai.getVariableInt("x") == 11
            test = ai.getVariableIntArray4("y")
            assert test == ARRAY_REFERENCE2
        
        # Step 2, all agents go from End->End2 state, and value become 13
        c.step()
        c.getPopulationData(pop)
        assert pop.getCurrentListSize(START_STATE) == 0
        assert pop.getCurrentListSize(END_STATE) == 0
        assert pop.getCurrentListSize(END_STATE2) == AGENT_COUNT
        for i in range(pop.getCurrentListSize(END_STATE2)): 
            ai = pop.getInstanceAt(i, END_STATE2)
            assert ai.getVariableInt("x") == 13
            test = ai.getVariableIntArray4("y")
            assert test == ARRAY_REFERENCE3
        

    def test_src_10_dest_10(self): 
        ARRAY_REFERENCE =  (13, 14, 15, 16) 
        ARRAY_REFERENCE2 =  (23, 24, 25, 26) 
        ARRAY_REFERENCE3 =  (3, 4, 5, 6) 
        # Init agents with two vars x and y
        # 3 round, 10 agents per round, with x and y value all set to 1 * round no
        # Each round 10 agents move from start to end state
        # Confirm why values are as expected
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME)
        a.newState(START_STATE)
        a.newState(END_STATE)
        a.setInitialState(START_STATE)
        a.newVariableUInt("x")
        a.newVariableUInt("y")
        a.newVariableIntArray4("z")
        af1 = a.newRTCFunction(FUNCTION_NAME1, self.AgentDecrement)
        af1.setInitialState(START_STATE)
        af1.setEndState(START_STATE)
        # Does nothing, just serves to demonstrate conditional state transition
        af2 = a.newRTCFunction(FUNCTION_NAME2, self.AgentNull)
        af2.setInitialState(START_STATE)
        af2.setEndState(END_STATE)
        af2.setRTCFunctionCondition(self.Zero_X)
        lo1 = m.newLayer(LAYER_NAME1)
        lo2 = m.newLayer(LAYER_NAME2)
        lo1.addAgentFunction(af1)
        lo2.addAgentFunction(af2)
        # Init pop
        ROUNDS = 3
        pop = pyflamegpu.AgentPopulation(a, ROUNDS * AGENT_COUNT)
        for i in range(ROUNDS * AGENT_COUNT): 
            ai = pop.getNextInstance(START_STATE)
            val = 1 + (i % ROUNDS)  # 1, 2, 3, 1, 2, 3 etc
            ai.setVariableUInt("x", val)
            ai.setVariableUInt("y", val)
            ai.setVariableIntArray4("z", ARRAY_REFERENCE)
        
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)

        # Step 1, all agents go from Start->End state, and value become 11
        for i in range(1, ROUNDS): 
            out = [0] * (ROUNDS + 1)
            c.step()
            c.getPopulationData(pop)
            assert pop.getCurrentListSize(START_STATE) == (ROUNDS - i) * AGENT_COUNT
            assert pop.getCurrentListSize(END_STATE) == i * AGENT_COUNT
            # Check val of agents in start state
            for j in range (pop.getCurrentListSize(START_STATE)): 
                ai = pop.getInstanceAt(j, START_STATE)
                y = ai.getVariableUInt("y")
                out[y] += 1
                test = ai.getVariableIntArray4("z")
                assert test == ARRAY_REFERENCE2
            
            assert out[0] == 0
            for j in range(1+i, ROUNDS): 
                assert out[j] == AGENT_COUNT
            
            for j in range(ROUNDS - i, 1 + i): 
                assert out[j] == 0
            
            # Check val of agents in end state
            out = [0] * (ROUNDS + 1)
            for j in range(pop.getCurrentListSize(END_STATE)): 
                ai = pop.getInstanceAt(j, END_STATE)
                y = ai.getVariableUInt("y")
                out[y] += 1
                test = ai.getVariableIntArray4("z")
                assert test == ARRAY_REFERENCE3
            
            for j in range(1, i + 1): 
                assert out[j] == AGENT_COUNT
            
            for j in range(i + 1, ROUNDS): 
                assert out[j] == 0
