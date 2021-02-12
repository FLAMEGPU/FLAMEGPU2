import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint


AGENT_COUNT = 100

class GPUTest(TestCase):

    add_func = """
    FLAMEGPU_AGENT_FUNCTION(add_func, MsgNone, MsgNone) {
        int x = FLAMEGPU->getVariable<int>("x");
        FLAMEGPU->setVariable<int>("x", x + 2);
        return ALIVE;
    }
    """
    
    sub_func = """
    FLAMEGPU_AGENT_FUNCTION(sub_func, MsgNone, MsgNone) {
        int x = FLAMEGPU->getVariable<int>("x");
        int y = FLAMEGPU->getVariable<int>("y");
        FLAMEGPU->setVariable<int>("y", x - y);
        return ALIVE;
    }
    """

    def test_gpu_memory(self):
        """
        To ensure initial values for agent population is transferred correctly onto
        the GPU, this test compares the values copied back from the device with the initial
        population data.
        """
        m = pyflamegpu.ModelDescription("test_gpu_memory_test")
        a = m.newAgent("agent")
        a.newVariableInt("id")
        p = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = p[i]
            instance.setVariableInt("id", i)
        cm = pyflamegpu.CUDASimulation(m)
        # copy to device then back by setting and getting population data
        cm.setPopulationData(p)
        cm.getPopulationData(p)
        # check values are the same
        for i in range(AGENT_COUNT):
            instance = p[i]
            assert instance.getVariableInt("id") == i

    def test_gpu_simulation(self):
        """
        To ensure initial values for agent population is transferred correctly onto
        the GPU, this test checks the correctness of the values copied back from the device
        after being updated/changed during the simulation of an agent function. The 'add_function' agent function simply increases the agent variable x by a value of 2.
        This test will re-use the original population to read the results of the simulation step so it also acts to test that the original values are correctly overwritten.
        """
        m = pyflamegpu.ModelDescription("test_gpu_memory_test")
        a = m.newAgent("agent")
        a.newVariableInt("id")
        a.newVariableInt("x")
        func = a.newRTCFunction("add_func", self.add_func)
        p = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = p[i]
            instance.setVariableInt("x", i)
        layer = m.newLayer("add_layer")
        layer.addAgentFunction(func)
        cm = pyflamegpu.CUDASimulation(m)
        cm.SimulationConfig().steps = 5
        cm.setPopulationData(p)
        cm.simulate()
        # Re-use the same population to read back the simulation step results
        cm.getPopulationData(p)
        # check values are the same
        for i in range(AGENT_COUNT):
            instance = p[i]
            # use AgentInstance equality operator
            assert instance.getVariableInt("x") == (i + (2 * 5))
            
    def test_gpu_simulation_multiple(self):
        """
        To test CUDA streams for overlapping host and device operations. This test is a test for concurrency.
        It is expected that add and subtract functions should execute simultaneously as they are functions belonging to different agents with the functions on the same simulation layer.
        Note: To observe that the functions are actually executing concurrently requires that you profile the test and observe the kernels in the profiler.
        """
        m = pyflamegpu.ModelDescription("test_gpu_simulation_multiple")
        a1 = m.newAgent("a1")
        a1.newVariableInt("x")
        a2 = m.newAgent("a2")
        a2.newVariableInt("x")
        a2.newVariableInt("y")
        func_add = a1.newRTCFunction("add_func", self.add_func)
        func_sub = a2.newRTCFunction("sub_func", self.sub_func)

        pop1 = pyflamegpu.AgentVector(a1, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = pop1[i]
            instance.setVariableInt("x", i)

        pop2 = pyflamegpu.AgentVector(a2, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = pop2[i]
            instance.setVariableInt("x", i)
            instance.setVariableInt("y", i)

        # multiple functions per simulation layer (from different agents)
        layer = m.newLayer("layer1")
        layer.addAgentFunction(func_add)
        layer.addAgentFunction(func_sub) # error here

        cm = pyflamegpu.CUDASimulation(m)
        cm.SimulationConfig().steps = 1
        cm.setPopulationData(pop1)
        cm.setPopulationData(pop2)
        cm.simulate()
        cm.getPopulationData(pop1)
        cm.getPopulationData(pop2)

        # check values are the same
        for i in range(AGENT_COUNT):
            instance1 = pop1[i]
            instance2 = pop2[i]
            assert instance1.getVariableInt("x") == i + 2
            assert instance2.getVariableInt("y") == 0
