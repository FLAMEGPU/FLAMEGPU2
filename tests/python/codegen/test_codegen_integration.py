import pytest
from unittest import TestCase
from pyflamegpu import *
import pyflamegpu.codegen
from random import randint
import typing

def return_two():
    return 2

AGENT_COUNT = 100
STEPS = 5

TEN: pyflamegpu.constant = 10
TWO: typing.Final = return_two()

@pyflamegpu.device_function
def add_2(a : int) -> int:
    """
    Pure python agent device function that can be called from a @pyflamegpu.agent_function
    Function adds two to an int variable
    """
    return a + TWO

@pyflamegpu.agent_function
def add_func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    x = pyflamegpu.getVariableInt("x")
    pyflamegpu.setVariableInt("x", add_2(x))
    return pyflamegpu.ALIVE


@pyflamegpu.agent_function_condition
def cond_func() -> bool:
    i = pyflamegpu.getVariableInt("i")
    return i < TEN

# agent function with error on line 4
agent_func_with_error = """
@pyflamegpu.agent_function
def error_func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    x = error_func_doesnt_exist()
    return pyflamegpu.ALIVE
"""

class GPUTest(TestCase):
    """
    Test provides a basic integration test of the pure python code generator (pyflamegpu.codegen). The test replicates the test_gpu_validation test but uses the 
    pure Python agent function syntax and decorators. Two device functions are used to ensure that the code generator correctly identifies these and translates them 
    into valid C++. It is not feasible to test all aspects of behavior via the code generator as these are tested in the python module via the C++ string agent 
    function syntax. 
    """


    @pyflamegpu.agent_function
    def agent_func_inclass(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
        return pyflamegpu.ALIVE

    def test_gpu_agent_func_in_class(self):
        """
        Only globally defined agent functions are supported. I.e. No class functions.
        These will be stripped by the translator and will result in an error at compilation time as no agent function will exist (only any valid device functions)
        """
        m = pyflamegpu.ModelDescription("test_gpu_agent_func_in_class")
        a = m.newAgent("agent")
        a.newVariableInt("x")
        func_translated = pyflamegpu.codegen.translate(self.agent_func_inclass)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            func = a.newRTCFunction("in_class_func", func_translated)
        assert "missing FLAMEGPU_AGENT_FUNCTION" in str(e.value)
        
    def test_gpu_codegen_simulation(self):
        """
        To ensure initial values for agent population is transferred correctly onto
        the GPU, this test checks the correctness of the values copied back from the device
        after being updated/changed during the simulation of an agent function. The 'add_func' agent function simply increases the agent variable x by a value of 2.
        This test will re-use the original population to read the results of the simulation step so it also acts to test that the original values are correctly overwritten.
        """
        m = pyflamegpu.ModelDescription("test_gpu_codegen_simulation")
        a = m.newAgent("agent")
        a.newVariableInt("x")
        func_translated = pyflamegpu.codegen.translate(add_func)
        func = a.newRTCFunction("add_func", func_translated)
        p = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = p[i]
            instance.setVariableInt("x", i)
        layer = m.newLayer("add_layer")
        layer.addAgentFunction(func)
        cm = pyflamegpu.CUDASimulation(m)
        cm.SimulationConfig().steps = STEPS
        cm.setPopulationData(p)
        cm.simulate()
        # Re-use the same population to read back the simulation step results
        cm.getPopulationData(p)
        # check values are the same
        for i in range(AGENT_COUNT):
            instance = p[i]
            # use AgentInstance equality operator
            assert instance.getVariableInt("x") == (i + (2 * STEPS))

    def test_gpu_codegen_function_condition(self):
        m = pyflamegpu.ModelDescription("test_gpu_codegen_function_condition")
        a = m.newAgent("agent")
        a.newVariableInt("i")
        a.newVariableInt("x")
        func_translated = pyflamegpu.codegen.translate(add_func)
        func_condition_translated = pyflamegpu.codegen.translate(cond_func)
        func = a.newRTCFunction("add_func", func_translated)
        func.setRTCFunctionCondition(func_condition_translated)
        p = pyflamegpu.AgentVector(a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = p[i]
            instance.setVariableInt("i", i) # store ordinal index as a variable as order is not preserved after function condition
            instance.setVariableInt("x", i)
        layer = m.newLayer("add_layer")
        layer.addAgentFunction(func)
        cm = pyflamegpu.CUDASimulation(m)
        cm.SimulationConfig().steps = STEPS
        cm.setPopulationData(p)
        cm.simulate()
        # Re-use the same population to read back the simulation step results
        cm.getPopulationData(p)
        # check values are the same (cant use a as order not guaranteed to be preserved)
        for a in range(AGENT_COUNT):
            instance = p[i]
            i = instance.getVariableInt("i")
            x = instance.getVariableInt("x")
            if i < 10:
                # Condition will have allowed agent function to multiply the x value if i < 10
                assert x == (i + (2 * STEPS))
            else:
                # Not meeting the condition means the agent function will now have executed and the x value should be unchanged
                assert x == i

    def test_gpu_codegen_line_error(self):
        """
        The following test will produce both a translation warning and error.
        Translation warning occurs as the function call to 'error_func_doesnt_exist' is non existant.
        Error occurs when this is compiled at start of simulation.
        The correct line number and 'PythonString' source file should be output by the compiler due to use of #line directive in translator
        """
        m = pyflamegpu.ModelDescription("test_gpu_codegen_line_error")
        a = m.newAgent("agent")
        a.newVariableInt("x")
        with pytest.warns() as record:
            func_translated = pyflamegpu.codegen.translate(agent_func_with_error)
        assert "not a defined FLAME GPU device function" in str(record[0].message)
        func = a.newRTCFunction("add_func", func_translated)
        layer = m.newLayer("add_layer")
        layer.addAgentFunction(func)
        cm = pyflamegpu.CUDASimulation(m)
        cm.SimulationConfig().steps = STEPS
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            cm.simulate()
        assert "PythonString(4)" in str(e.value)
