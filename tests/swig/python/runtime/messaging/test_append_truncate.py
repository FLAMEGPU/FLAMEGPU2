import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

MODEL_NAME = "Model"
AGENT_NAME = "Agent"
MESSAGE_NAME = "Message"
IN_FUNCTION_NAME = "In_AppendTruncate"
OUT_FUNCTION_NAME = "Out_AppendTruncate"
OUT_FUNCTION_NAME2 = "Out_AppendTruncate2"
IN_LAYER_NAME = "InLayer"
OUT_LAYER_NAME = "OutLayer"
OUT_LAYER2_NAME = "OutLayer2"
AGENT_COUNT = 1024

Out_AppendTruncate = """
FLAMEGPU_AGENT_FUNCTION(Out_AppendTruncate, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable("x", 0);
    return flamegpu::ALIVE;
}
"""

Out_AppendTruncate2 = """
FLAMEGPU_AGENT_FUNCTION(Out_AppendTruncate2, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable("x", 1);
    return flamegpu::ALIVE;
}
"""

In_AppendTruncate = """
FLAMEGPU_AGENT_FUNCTION(In_AppendTruncate, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    int count = 0;
    for (auto &message : FLAMEGPU->message_in) {
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return flamegpu::ALIVE;
}
"""

In_AppendTruncate2 = """
FLAMEGPU_AGENT_FUNCTION(In_AppendTruncate2, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    int count0 = 0;
    int count1 = 0;
    for (auto &message : FLAMEGPU->message_in) {
        if (message.getVariable<int>("x") == 0) {
            count0++;
        } else if (message.getVariable<int>("x") == 1) {
            count1++;
        }
    }
    FLAMEGPU->setVariable<unsigned int>("count0", count0);
    FLAMEGPU->setVariable<unsigned int>("count1", count1);
    return flamegpu::ALIVE;
}
"""

OptionalOut_AppendTruncate = """
FLAMEGPU_AGENT_FUNCTION(OptionalOut_AppendTruncate, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    if (FLAMEGPU->getVariable<unsigned int>("do_out") > 0) {
        FLAMEGPU->message_out.setVariable("x", 0);
        FLAMEGPU->setVariable<unsigned int>("do_out", 0);
    } else {
        FLAMEGPU->setVariable<unsigned int>("do_out", 1);
    }
    return flamegpu::ALIVE;
}
"""

OptionalOut_AppendTruncate2 = """
FLAMEGPU_AGENT_FUNCTION(OptionalOut_AppendTruncate2, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    if (FLAMEGPU->getVariable<unsigned int>("do_out") > 0) {
        FLAMEGPU->message_out.setVariable("x", 1);
    }
    return flamegpu::ALIVE;
}
"""    
    


class TestMessageAppendTruncate(TestCase):
    """
     Test whether a group of agents can output unique messages which can then all be read back by the same (all) agents
    """

    def test_Truncate(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("count")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, Out_AppendTruncate)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, In_AppendTruncate)
        fi.setMessageInput(message)

        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for ai in pop:
            ai.setVariableUInt("count", 0)
        
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count") == AGENT_COUNT
        
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count") == AGENT_COUNT
        
    
    def test_Append_KeepData(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("count0")
        a.newVariableUInt("count1")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, Out_AppendTruncate)
        fo.setMessageOutput(message)
        fo2 = a.newRTCFunction(OUT_FUNCTION_NAME2, Out_AppendTruncate2)
        fo2.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, In_AppendTruncate2)
        fi.setMessageInput(message)

        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for ai in pop:
            ai.setVariableUInt("count0", 0)
            ai.setVariableUInt("count1", 0)
        
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        lo2 = m.newLayer(OUT_LAYER2_NAME)
        lo2.addAgentFunction(fo2)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count0") == AGENT_COUNT
            assert ai.getVariableUInt("count1") == AGENT_COUNT
        
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count0") == AGENT_COUNT
            assert ai.getVariableUInt("count1") == AGENT_COUNT
        
    
    def test_OptionalTruncate(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("count")
        a.newVariableUInt("do_out")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OptionalOut_AppendTruncate)
        fo.setMessageOutputOptional(True)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, In_AppendTruncate)
        fi.setMessageInput(message)
        result_count = 0
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for ai in pop:
            if rand.random() < 0.7:   # 70% chance of outputting
                ai.setVariableUInt("do_out", 1)
                result_count += 1
            else:
                ai.setVariableUInt("do_out", 0)
            
            ai.setVariableUInt("count", 0)
        
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count") == result_count
        
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count") == AGENT_COUNT - result_count
        
    
    def test_OptionalAppend_KeepData(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableUInt("count0")
        a.newVariableUInt("count1")
        a.newVariableUInt("do_out")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OptionalOut_AppendTruncate)
        fo.setMessageOutputOptional(True)
        fo.setMessageOutput(message)
        fo2 = a.newRTCFunction(OUT_FUNCTION_NAME2, OptionalOut_AppendTruncate2)
        fo2.setMessageOutputOptional(True)
        fo2.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, In_AppendTruncate2)
        fi.setMessageInput(message)
        result_count = 0
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for ai in pop:
            if rand.random() < 0.7:   # 70% chance of outputting
                ai.setVariableUInt("do_out", 1)
                result_count += 1
            else:
                ai.setVariableUInt("do_out", 0)
            
            ai.setVariableUInt("count0", 0)
            ai.setVariableUInt("count1", 0)
        
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        lo2 = m.newLayer(OUT_LAYER2_NAME)
        lo2.addAgentFunction(fo2)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count0") == result_count
            assert ai.getVariableUInt("count1") == AGENT_COUNT - result_count
        
        c.step()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableUInt("count0") == AGENT_COUNT - result_count
            assert ai.getVariableUInt("count1") == result_count
        
    
  # namespace test_message_AppendTruncate
