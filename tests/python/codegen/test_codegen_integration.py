import pytest
from unittest import TestCase
from pyflamegpu import *
import pyflamegpu.codegen
from random import randint
import typing
import importlib.util
import sys
from pathlib import Path
import codegen_example
import inspect

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

# agent function with error on line 4 (starting line is empty due to newline character)
agent_func_with_error_str = """
@pyflamegpu.agent_function
def error_func(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    x = error_func_doesnt_exist()
    return pyflamegpu.ALIVE
"""

"""
Agent function with an error on line `line_of_agent_func_error`. Note if this source file changes the line number may change.
"""
line_of_agent_func_error = 56
@pyflamegpu.agent_function
def agent_func_with_error(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    x = locals() # error on this line (function doesn't exist on GPU)
    return pyflamegpu.ALIVE




agent_func_complex_example_cpp = """

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
        """
        m = pyflamegpu.ModelDescription("test_gpu_agent_func_in_class")
        a = m.newAgent("agent")
        a.newVariableInt("x")
        with pytest.raises(pyflamegpu.codegen.CodeGenException) as e:
            func_translated = pyflamegpu.codegen.translate(self.agent_func_inclass)
            func = a.newRTCFunction("in_class_func", func_translated)
        assert "Function passed to translate is not a global" in str(e.value)
        
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

    def test_gpu_codegen_line_error_str(self):
        """
        The following test will produce both a translation warning and error.
        Translation warning occurs as the function call to 'error_func_doesnt_exist' is non existant.
        Error occurs when this is compiled at start of simulation.
        The correct line number and 'PythonString' source file should be output by the compiler due to use of #line directive in translator
        """
        m = pyflamegpu.ModelDescription("test_gpu_codegen_line_error_str")
        a = m.newAgent("agent")
        a.newVariableInt("x")
        with pytest.warns() as record:
            func_translated = pyflamegpu.codegen.translate(agent_func_with_error_str)
        assert "not a defined FLAME GPU device function" in str(record[0].message)
        func = a.newRTCFunction("add_func", func_translated)
        layer = m.newLayer("add_layer")
        layer.addAgentFunction(func)
        cm = pyflamegpu.CUDASimulation(m)
        cm.SimulationConfig().steps = STEPS
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            cm.simulate()
        assert "PythonString(4)" in str(e.value)

    def test_gpu_codegen_line_error_file(self):
        """
        Same as test_gpu_codegen_line_error_str but using an agent function from this file.
        The test will produce both a translation warning and error.
        Translation warning occurs as the function call to 'locals' is non existant.
        Error occurs when this is compiled at start of simulation.
        The correct line number and 'PythonString' source file should be output by the compiler due to use of #line directive in translator
        """
        m = pyflamegpu.ModelDescription("test_gpu_codegen_line_error_file")
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
        assert f"test_codegen_integration.py({line_of_agent_func_error})" in str(e.value)

    def test_gpu_codegen_line_error_dynamic(self):
        """
        Same as test_gpu_codegen_line_error_file but modifies a copy of the originating module by removing the __file__ attribute. 
            This replicates the behaviour of a Notebook where the __file__ attribute does not exist.
        Translation warning occurs as the function call to 'locals' is non existant.
        Error occurs when this is compiled at start of simulation.
        The correct line number and 'PythonDynamic' source file should be output by the compiler due to use of #line directive in translator
            The line number will represent the line in the agent function. Importantly any device functions prepended to the agent function 
            should not impact the line number.
        """
        
        # Generate a clone of the module with a different name so that it can be modified without impacting other tests.
        # Note that the mocking approach does not work as mocking does not support removal and reinstating of attributes.
        path = Path(__file__).resolve()
        spec = importlib.util.spec_from_file_location("test_codegen_integration_modified", path)
        test_codegen_integration_modified = importlib.util.module_from_spec(spec)
        loader = spec.loader
        loader.exec_module(test_codegen_integration_modified)
        sys.modules["test_codegen_integration_modified"] = test_codegen_integration_modified
        test_codegen_integration_modified.__module__ = "test_codegen_integration_modified"
        # delete __file__attribute
        del test_codegen_integration_modified.__file__
        
        m = pyflamegpu.ModelDescription("test_gpu_codegen_line_error_file")
        a = m.newAgent("agent")
        a.newVariableInt("x")
        with pytest.warns() as record:
            # pass the agent function from the modified module
            func_translated = pyflamegpu.codegen.translate(test_codegen_integration_modified.agent_func_with_error)
        assert "not a defined FLAME GPU device function" in str(record[0].message)
        func = a.newRTCFunction("add_func", func_translated)
        layer = m.newLayer("add_layer")
        layer.addAgentFunction(func)
        cm = pyflamegpu.CUDASimulation(m)
        cm.SimulationConfig().steps = STEPS
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            cm.simulate()
        assert f"DynamicPython(3)" in str(e.value)

    def test_gpu_codegen_complex_file(self):
        """
        C++ #line directives are injected into the generated C++ code so that compilation errors report the original Python filename and line numbers. A
        line directive should be inserted after each python statement or expression resulting in a new Python line. This test checks the full output
        of a "complex" code generation example where a whole bunch of different device API functionality is used. This is not exhaustive but a bit of a kitchen
        sink approach where it is just doing lots of stuff.
        The expected result will have a line directive for each (functional, i.e. not curly braces etc.) C++ line which sets the filename as 'codegen_example.py'
        and the line number to correspond to the originating line of 'codegen_example.py.'. Internally this will test the ModuleExtractor class from 'codegen.py'
        which extracts the original source of the module containing the agent function and strips all but the agent function and any device functions so as to preserve 
        the original line numbers.

        IMPORTANT: See comments in codegen_example.py relating to any changes in the source file as this which will impact line numbers and hence the expected result 
        of this test.
        """
        m = pyflamegpu.ModelDescription("test_gpu_codegen_complex")
        func_translated = pyflamegpu.codegen.translate(codegen_example.agent_func_complex_example)
        # Ensure expected result (use strip to remove any leading or trailing newlines)
        assert func_translated.strip() == codegen_example.agent_func_complex_example_file_cpp.strip()

    def test_gpu_codegen_complex_str(self):
        """
        This is the same test as test_gpu_codegen_complex however the source file will be translated as a string passed to the translator. This avoids the need to 
        use the ModuleExtractor and will not prepend any device agent functions which are not included in the string source already. Hence line numbers correspend 
        directly to the line number of the source text passed to translate().
        """
        m = pyflamegpu.ModelDescription("test_gpu_codegen_complex_str")
        func_translated = pyflamegpu.codegen.translate(codegen_example.agent_func_complex_example_str)
        # Ensure expected result (use strip to remove any leading or trailing newlines)
        assert func_translated.strip() == codegen_example.agent_func_complex_example_str_cpp.strip()

    def test_gpu_codegen_complex_dynamic(self):
        """
        This is the same test as test_gpu_codegen_complex however a copy of the module containing the agent function is made so that the __file__ attribute can be 
        removed. This emulates the behaviour of a Jupyter notebook (which has no __file__) and will force the translator to create a dynamic source input (rather 
        than use the ModuleExtractor) which contains the source of the agent function proceeded by any agent device functions which were defined in the same dynamic
        module (i.e. In a Notebook cell executed prior to the one calling translate). As line numbers become meaningless within this dynamically generated source 
        file the line directive is offset to start from the beginning of the agent function (with the decorator being line 1). Users then have a meaningful line number 
        to understand error codes. Any errors in agent device functions will unfortunately be reported at line 1 but this seems to be the most sensible thing to do.

        Note: Line numbers are not dependant on the codegen_example.py source file unless the agent or device functions are changed.
        """
        # Generate a clone of the module with a different name so that it can be modified without impacting other tests.
        # Note that the mocking approach does not work as mocking does not support removal and reinstating of attributes.
        path = Path(codegen_example.__file__).resolve()
        spec = importlib.util.spec_from_file_location("codegen_example_modified", path)
        codegen_example_modified = importlib.util.module_from_spec(spec)
        loader = spec.loader
        loader.exec_module(codegen_example_modified)
        sys.modules["codegen_example_modified"] = codegen_example_modified
        codegen_example_modified.__module__ = "codegen_example_modified"
        # delete __file__attribute
        del codegen_example_modified.__file__

        m = pyflamegpu.ModelDescription("test_gpu_codegen_complex_modified")
        func_translated = pyflamegpu.codegen.translate(codegen_example_modified.agent_func_complex_example)
        # Ensure expected result (use strip to remove any leading or trailing newlines)
        assert func_translated.strip() == codegen_example_modified.agent_func_complex_example_dynamic_cpp.strip()
        
