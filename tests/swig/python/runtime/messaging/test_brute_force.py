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
FLAMEGPU_AGENT_FUNCTION(OutFunction, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return flamegpu::ALIVE;
}
"""

OutFunction_Optional = """
FLAMEGPU_AGENT_FUNCTION(OutFunction_Optional, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    const int x = FLAMEGPU->getVariable<int>("x");
    if (x) FLAMEGPU->message_out.setVariable<int>("x", x);
    return flamegpu::ALIVE;
}
"""

InFunction = """
FLAMEGPU_AGENT_FUNCTION(InFunction, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        product = product < -1000000 ? -1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    return flamegpu::ALIVE;
}
"""

InFunction2 = """
FLAMEGPU_AGENT_FUNCTION(InFunction2, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    int sum = FLAMEGPU->getVariable<int>("sum");
    int product = FLAMEGPU->getVariable<int>("product");
    for (auto &message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
        product *= x;
        product = product > 1000000 ? 1 : product;
        product = product < -1000000 ? -1 : product;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    FLAMEGPU->setVariable<int>("product", product);
    int x = FLAMEGPU->getVariable<int>("x");
    FLAMEGPU->setVariable<int>("x", x + 1);
    return flamegpu::ALIVE;
}
"""

countBF = """
FLAMEGPU_AGENT_FUNCTION(countBF, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
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
even_only_condition = """
FLAMEGPU_AGENT_FUNCTION_CONDITION(EvenOnlyCondition) {
    return FLAMEGPU->getStepCounter() % 2 == 0;
}
"""
out_simple = """
FLAMEGPU_AGENT_FUNCTION(out_simple, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->message_out.setVariable<int>("id", id);
    return flamegpu::ALIVE;
}
"""

in_simple = """
FLAMEGPU_AGENT_FUNCTION(in_simple, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    const int id = FLAMEGPU->getVariable<int>("id");
    unsigned int count = 0;
    unsigned int sum = 0;
    for (auto &m : FLAMEGPU->message_in) {
        count++;
        sum += m.getVariable<int>("id");
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    FLAMEGPU->setVariable<unsigned int>("sum", sum);
    return flamegpu::ALIVE;
}"""

class InitPopulationEvenOutputOnly(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Generate a basic pop
        agent = FLAMEGPU.agent(AGENT_NAME)
        for i in range (AGENT_COUNT) :
            instance = agent.newAgent()
            instance.setVariableInt("id", i)
            instance.setVariableUInt("count", 0)
            instance.setVariableUInt("sum", 0)

class AssertEvenOutputOnly(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent(AGENT_NAME)
        # Get the population data
        av = agent.getPopulationData()
        # Iterate the population, ensuring that each agent read the correct number of messages and got the correct sum of messages.
        # These values expect only a single bin is used, in the interest of simplicitly.
        exepctedCountEven = agent.count()
        expectedCountOdd = 0
        for a in av:
            if (FLAMEGPU.getStepCounter() % 2 == 0):
                # Even iterations expect the count to match the number of agents, and sum to be non zero.
                assert a.getVariableUInt("count") == exepctedCountEven
                assert a.getVariableUInt("sum") != 0
            else:
                # Odd iters expect 0 count and 0 sum
                assert a.getVariableUInt("count") == expectedCountOdd
                assert a.getVariableUInt("sum") == 0

class AssertPersistent(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent(AGENT_NAME)
        # Get the population data
        av = agent.getPopulationData()
        # Iterate the population, ensuring that each agent read the correct number of messages and got the correct sum of messages.
        # These values expect only a single bin is used, in the interest of simplicitly.
        exepctedCountEven = agent.count()
        for a in av:
            if (FLAMEGPU.getStepCounter() % 2 == 0):
                # all iterations expect the count to match the number of agents, and sum to be non zero.
                assert a.getVariableUInt("count") == exepctedCountEven
                assert a.getVariableUInt("sum") != 0

class TestMessage_BruteForce(TestCase):


    def test_Mandatory1(self): 
        """
        Test whether a group of agents can output unique messages which can then all be read back by the same (all) agents
        """
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction)
        fi.setMessageInput(message)
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
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction2)
        fi.setMessageInput(message)
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        sum = 0
        product = 1
        for ai in pop:
            x = rand.randint(-3, 3)
            ai.setVariableInt("x", x)
            ai.setVariableInt("sum", 0)
            ai.setVariableInt("product", 1)
            sum += x
            product *= x
            if product > 1000000:
                product = 1
            if product < -1000000:
                product = -1
        for ai in pop:
            x = ai.getVariableInt("x")
            # Calc second iteration sum/product
            sum += (x + 1)
            product *= (x + 1)
            if product > 1000000:
                product = 1
                if product < -1000000:
                    product = -1
        
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
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction_Optional)
        fo.setMessageOutputOptional(True)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction)
        fi.setMessageInput(message)

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
                if product < -1000000:
                    product = -1
            
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
        message = m.newMessageBruteForce(MESSAGE_NAME)
        message.newVariableInt("x")
        a = m.newAgent(AGENT_NAME)
        a.newVariableInt("x")
        a.newVariableInt("sum")
        a.newVariableInt("product")
        fo = a.newRTCFunction(OUT_FUNCTION_NAME, OutFunction_Optional)
        fo.setMessageOutputOptional(True)
        fo.setMessageOutput(message)
        fi = a.newRTCFunction(IN_FUNCTION_NAME, InFunction2)
        fi.setMessageInput(message)

        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        sum = 0
        product = 1
        for ai in pop:
            x = rand.randint(-3,3)            
            ai.setVariableInt("x", x)
            ai.setVariableInt("sum", 0)
            ai.setVariableInt("product", 1)
            # Calc first iteration sum/product
            if (x): 
                sum += x
                product *= x
                if product > 1000000:
                    product = 1
                if product < -1000000:
                    product = -1
        
        for ai in pop:
            x = ai.getVariableInt("x")
            # Calc second iteration sum/product
            if (x + 1): # dont proceed if x == -1
                sum += (x + 1)
                product *= (x + 1)
                if product > 1000000:
                    product = 1
                if product < -1000000:
                    product = -1
            
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
        message = m.newMessageBruteForce(MESSAGE_NAME)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.newVariableInt("_")
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

    def test_getSetPersistent(self):
        """Test that getting and setting a message lists's persistent flag behaves as intended
        """
        model = pyflamegpu.ModelDescription("Model")
        message = model.newMessageBruteForce("location")
        # message lists should be non-persistent by default
        assert message.getPersistent() == False
        # Settiog the persistent value ot true should not throw
        message.setPersistent(True)
        # The value should now be true
        assert message.getPersistent() == True
        # Set it to true again, to make sure it isn't an invert
        message.setPersistent(True)
        assert message.getPersistent() == True
        # And flip it back to false for good measure
        message.setPersistent(False)
        assert message.getPersistent() == False

    def test_PersistenceOff(self): 
        """Test for persistence / non persistence of messaging, by emitting messages on even iters, but reading on all iters.
        """
        model = pyflamegpu.ModelDescription("TestMessage_BruteForce")
        message = model.newMessageBruteForce("msg")
        message.newVariableInt("id")

        # agent
        agent = model.newAgent(AGENT_NAME)
        agent.newVariableInt("id")
        agent.newVariableUInt("count", 0)  # Count the number of messages read
        agent.newVariableUInt("sum", 0)  # Count of IDs
        ouf = agent.newRTCFunction("out", out_simple)
        ouf.setMessageOutput("msg")
        ouf.setMessageOutputOptional(True)
        ouf.setRTCFunctionCondition(even_only_condition)
        inf = agent.newRTCFunction("in", in_simple)
        inf.setMessageInput("msg")

        # Define layers
        model.newLayer().addAgentFunction(ouf)
        model.newLayer().addAgentFunction(inf)
        # init function for pop
        init_population_even_output_only = InitPopulationEvenOutputOnly()
        model.addInitFunctionCallback(init_population_even_output_only)
        # add a step function which validates the correct number of messages was read
        assert_even_output_only = AssertEvenOutputOnly()
        model.addStepFunctionCallback(assert_even_output_only)

        cudaSimulation = pyflamegpu.CUDASimulation(model)
        # Execute model
        cudaSimulation.SimulationConfig().steps = 2
        cudaSimulation.simulate()

    def test_PersistenceOn(self): 
        """Test for persistence / non persistence of messaging, by emitting messages on even iters, but reading on all iters.
        """
        model = pyflamegpu.ModelDescription("TestMessage_BruteForce")
        message = model.newMessageBruteForce("msg")
        message.newVariableInt("id")

        # agent
        agent = model.newAgent(AGENT_NAME)
        agent.newVariableInt("id")
        agent.newVariableUInt("count", 0)  # Count the number of messages read
        agent.newVariableUInt("sum", 0)  # Count of IDs
        ouf = agent.newRTCFunction("out", out_simple)
        ouf.setMessageOutput("msg")
        ouf.setMessageOutputOptional(True)
        ouf.setRTCFunctionCondition(even_only_condition)
        inf = agent.newRTCFunction("in", in_simple)
        inf.setMessageInput("msg")

        # Define layers
        model.newLayer().addAgentFunction(ouf)
        model.newLayer().addAgentFunction(inf)
        # init function for pop
        init_population_even_output_only = InitPopulationEvenOutputOnly()
        model.addInitFunctionCallback(init_population_even_output_only)
        # add a step function which validates the correct number of messages was read
        assert_persistent = AssertPersistent()
        model.addStepFunctionCallback(assert_persistent)

        cudaSimulation = pyflamegpu.CUDASimulation(model)
        # Execute model
        cudaSimulation.SimulationConfig().steps = 2
        cudaSimulation.simulate()