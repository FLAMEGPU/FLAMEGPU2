import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand


MODEL_NAME = "Model"
AGENT_NAME = "Agent"
MESSAGE_NAME = "Message"
IN_FUNCTION_NAME = "InFunction"
OUT_FUNCTION_NAME = "OutFunction"
IN_LAYER_NAME = "InLayer"
OUT_LAYER_NAME = "OutLayer"
AGENT_COUNT = 128


OutFunction = """
FLAMEGPU_AGENT_FUNCTION(OutFunction, flamegpu::MsgNone, flamegpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return flamegpu::ALIVE;
}
"""

OutFunction_Optional = """
FLAMEGPU_AGENT_FUNCTION(OutFunction_Optional, flamegpu::MsgNone, flamegpu::MsgBruteForce) {
    const int x = FLAMEGPU->getVariable<int>("x");
    if (x) FLAMEGPU->message_out.setVariable("x", x);
    return flamegpu::ALIVE;
}
"""

InFunction = """
FLAMEGPU_AGENT_FUNCTION(InFunction, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    return flamegpu::ALIVE;
}
"""

InFunction2 = """
FLAMEGPU_AGENT_FUNCTION(InFunction2, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + 1);
    return flamegpu::ALIVE;
}
"""

countBF = """
FLAMEGPU_AGENT_FUNCTION(countBF, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    unsigned int count = 0;
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3 Moore neighbourhood
    for (const auto &message : FLAMEGPU->message_in) {
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return flamegpu::ALIVE;
}
"""

class TestMessage_BruteForce(TestCase):


    def test_Mandatory1(self): 
        """
        Test whether a group of agents can output unique messages which can then all be read back by the same (all) agents
        """
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        msg = m.newMessageBruteForce(MESSAGE_NAME)
        msg.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction)
        fo.setMessageOutput(msg)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction)
        fi.setMessageInput(msg)
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        sum = 0
        product = 1
        for ai in pop:
            x = rand.randint(-3, 3)
            sum += x
            product *= x
            if product > 1000000:
                product = 1
            ai.setVariableInt("x", x)
            ai.setVariableInt("sum", 0)
            ai.setVariableInt("product", 1)
        
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 1
        c.setPopulationData(pop)
        c.simulate()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableInt("sum") == sum
            assert ai.getVariableInt("product") == product
        

    def test_Mandatory2(self): 
        """
        Ensures messages are correct on 2nd step
        """
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        msg = m.newMessageBruteForce(MESSAGE_NAME)
        msg.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction)
        fo.setMessageOutput(msg)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction2)
        fi.setMessageInput(msg)
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        sum = 0
        product = 1
        for ai in pop:
            x = rand.randint(-3, 3)
            sum += x
            product *= x
            if product > 1000000:
                product = 1
            sum += (x + 1)
            product *= (x + 1)
            if product > 1000000:
                product = 1
            ai.setVariableInt("x", x)
            ai.setVariableInt("sum", 0)
            ai.setVariableInt("product", 1)
        
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 2
        c.setPopulationData(pop)
        c.simulate()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableInt("sum") == sum
            assert ai.getVariableInt("product") == product
        

    def test_Optional1(self): 
        """
        Test whether a group of agents can optionally output unique messages which can then all be read back by the same agents
        """
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        msg = m.newMessageBruteForce(MESSAGE_NAME)
        msg.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction_Optional)
        fo.setMessageOutputOptional(True)
        fo.setMessageOutput(msg)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction)
        fi.setMessageInput(msg)

        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        sum = 0
        product = 1
        for ai in pop:
            x = rand.randint(-3, 3)
            if (x): 
                sum += x
                product *= x
                if product > 1000000:
                    product = 1
            
            ai.setVariableInt("x", x)
            ai.setVariableInt("sum", 0)
            ai.setVariableInt("product", 1)
        
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 1
        c.setPopulationData(pop)
        c.simulate()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableInt("sum") == sum
            assert ai.getVariableInt("product") == product
        

    def test_Optional2(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        msg = m.newMessageBruteForce(MESSAGE_NAME)
        msg.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction_Optional)
        fo.setMessageOutputOptional(True)
        fo.setMessageOutput(msg)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction2)
        fi.setMessageInput(msg)

        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        sum = 0
        product = 1
        for ai in pop:
            x = rand.randint(-3,3)
            if (x): 
                sum += x
                product *= x
                if product > 1000000:
                    product = 1
            
            ai.setVariableInt("x", x)
            ai.setVariableInt("sum", 0)
            ai.setVariableInt("product", 1)
        
        for ai in pop:
            x = ai.getVariableInt("x")
            if (x + 1): # dont proceed if x == -1
                sum += (x + 1)
                product *= (x + 1)
                if product > 1000000:
                    product = 1
            
        lo = m.newLayer(OUT_LAYER_NAME)
        lo.addAgentFunction(fo)
        li = m.newLayer(IN_LAYER_NAME)
        li.addAgentFunction(fi)
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 2
        c.setPopulationData(pop)
        c.simulate()
        c.getPopulationData(pop)
        # Validate each agent has same result
        for ai in pop:
            assert ai.getVariableInt("sum") == sum
            assert ai.getVariableInt("product") == product
        
    def test_reserved_name(self): 
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        msg = m.newMessageBruteForce(MESSAGE_NAME)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            msg.newVariableInt("_")
        assert e.value.type() == "ReservedName"


    def test_ReadEmpty(self): 
        # What happens if we read a message list before it has been output?
        model = pyflamegpu.ModelDescription("Model")
        # Location message
        message = model.newMessageBruteForce("location")
        message.newVariableInt("id")  # unused by current test

        # Circle agent
        agent = model.newAgent("agent")
        agent.newVariableUInt("count", 0)  # Count the number of messages read
        cf = agent.newRTCFunction("in", countBF)
        cf.setMessageInput("location")

        # Layer #1
        layer = model.newLayer()
        layer.addAgentFunction(cf)
        
        # Create 1 agent
        pop_in = pyflamegpu.AgentVector(model.Agent("agent"), 1)
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.setPopulationData(pop_in)
        # Execute model
        cudaSimulation.step()
        # Check result
        pop_out = pyflamegpu.AgentVector(model.Agent("agent"), 1)
        pop_out.front().setVariableUInt("count", 1)
        cudaSimulation.getPopulationData(pop_out)
        assert len(pop_out) == 1
        ai = pop_out.front()
        assert ai.getVariableUInt("count") == 0
