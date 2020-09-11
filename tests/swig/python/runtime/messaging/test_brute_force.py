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
FLAMEGPU_AGENT_FUNCTION(OutFunction, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
"""

OutFunction_Optional = """
FLAMEGPU_AGENT_FUNCTION(OutFunction_Optional, MsgNone, MsgBruteForce) {
    const int x = FLAMEGPU->getVariable<int>("x");
    if (x) FLAMEGPU->message_out.setVariable("x", x);
    return ALIVE;
}
"""

InFunction = """
FLAMEGPU_AGENT_FUNCTION(InFunction, MsgBruteForce, MsgNone) {
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
    return ALIVE;
}
"""

InFunction2 = """
FLAMEGPU_AGENT_FUNCTION(InFunction2, MsgBruteForce, MsgNone) {
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
    return ALIVE;
}
"""

countBF = """
FLAMEGPU_AGENT_FUNCTION(countBF, MsgBruteForce, MsgNone) {
    unsigned int count = 0;
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3 Moore neighbourhood
    for (const auto &message : FLAMEGPU->message_in) {
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return ALIVE;
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
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        sum = 0
        product = 1
        for i in range(AGENT_COUNT): 
            ai = pop.getNextInstance()
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
        for i in range(AGENT_COUNT): 
            ai = pop.getInstanceAt(i)
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
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        sum = 0
        product = 1
        for i in range(AGENT_COUNT): 
            ai = pop.getNextInstance()
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
        for i in range(AGENT_COUNT): 
            ai = pop.getInstanceAt(i)
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

        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        sum = 0
        product = 1
        for i in range(AGENT_COUNT): 
            ai = pop.getNextInstance()
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
        for i in range(AGENT_COUNT): 
            ai = pop.getInstanceAt(i)
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

        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        sum = 0
        product = 1
        for i in range(AGENT_COUNT): 
            ai = pop.getNextInstance()
            x = rand.randint(-3,3)
            if (x): 
                sum += x
                product *= x
                if product > 1000000:
                    product = 1
            
            ai.setVariableInt("x", x)
            ai.setVariableInt("sum", 0)
            ai.setVariableInt("product", 1)
        
        for i in range(AGENT_COUNT): 
            ai = pop.getInstanceAt(i)
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
        for i in range(AGENT_COUNT): 
            ai = pop.getInstanceAt(i)
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
        pop_in = pyflamegpu.AgentPopulation(model.Agent("agent"), 1)
        pop_in.getNextInstance()
        cuda_model = pyflamegpu.CUDASimulation(model)
        cuda_model.setPopulationData(pop_in)
        # Execute model
        cuda_model.step()
        # Check result
        pop_out = pyflamegpu.AgentPopulation(model.Agent("agent"), 1)
        pop_out.getNextInstance().setVariableUInt("count", 1)
        cuda_model.getPopulationData(pop_out)
        assert pop_out.getCurrentListSize() == 1
        ai = pop_out.getInstanceAt(0)
        assert ai.getVariableUInt("count") == 0
