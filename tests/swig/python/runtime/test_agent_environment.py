import pytest
from unittest import TestCase
from pyflamegpu import *

SCALAR_TEST_VALUE = 12
ARRAY_TEST_VALUE = (1,2,3,4)

class MiniSim:
    """
    Replaces the mini sim class in C++ but provides a function to return a dynamic agent 
    function based off the provided type.
    """
 
    def __init__(self):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        self.population = None
        self.env = self.model.Environment() 

    @staticmethod
    def get_agent_function(c_type:str) -> str:

        return f"""
        FLAMEGPU_AGENT_FUNCTION(get_{c_type}, flamegpu::MsgNone, flamegpu::MsgNone) {{
            {c_type} a_out = FLAMEGPU->environment.getProperty<{c_type}>("a");
            int contains_a = FLAMEGPU->environment.containsProperty("a");
            FLAMEGPU->setVariable<{c_type}>("a_out", a_out);
            FLAMEGPU->setVariable<int>("contains_a", contains_a);
            return flamegpu::ALIVE;
        }}
        """

    @staticmethod
    def get_agent_array_function(c_type:str) -> str:

        return f"""
        FLAMEGPU_AGENT_FUNCTION(get_{c_type}, flamegpu::MsgNone, flamegpu::MsgNone) {{
            {c_type} a1_out = FLAMEGPU->environment.getProperty<{c_type}>("a", 1);
            int contains_b = FLAMEGPU->environment.containsProperty("b");
            FLAMEGPU->setVariable<{c_type}>("a1_out", a1_out);
            FLAMEGPU->setVariable<int>("contains_b", contains_b);
            return flamegpu::ALIVE;
        }}
        """
        
    def run_agent_function_test(self, c_type: str, python_type: str):
        # add agent variables
        new_agent_var_func = getattr(self.agent, f"newVariable{python_type}")
        new_agent_var_func("a_out")
        self.agent.newVariableInt("contains_a")
        # add the agent function
        func_str = MiniSim.get_agent_function(c_type)
        func = self.agent.newRTCFunction("device_function", func_str)
        layer = self.model.newLayer("devicefn_layer")
        layer.addAgentFunction(func)
        # Setup environment
        add_func = getattr(self.env, f"newProperty{python_type}")
        add_func("a", SCALAR_TEST_VALUE)
        # run
        self.__run()
        # get instance and test output values
        instance = self.population.front()
        instance_get_func = getattr(instance, f"getVariable{python_type}")
        assert instance_get_func("a_out") == SCALAR_TEST_VALUE
        assert instance.getVariableInt("contains_a") == True
        
    def run_agent_function_array_test(self, c_type: str, python_type: str):
        # add agent variables
        new_agent_var_func = getattr(self.agent, f"newVariable{python_type}")
        new_agent_var_func("a1_out")
        self.agent.newVariableInt("contains_b")
        # add the agent function
        func_str = MiniSim.get_agent_array_function(c_type)
        func = self.agent.newRTCFunction("device_function", func_str)
        layer = self.model.newLayer("devicefn_layer")
        layer.addAgentFunction(func)
        print(self.env)
        # Setup environment
        add_func = getattr(self.env, f"newPropertyArray{python_type}")
        add_func("a", 4, ARRAY_TEST_VALUE)
        # run
        self.__run()
        # get instance and test output values
        instance = self.population.front()
        instance_get_func = getattr(instance, f"getVariable{python_type}")
        assert instance_get_func("a1_out") == ARRAY_TEST_VALUE[1]
        assert instance.getVariableInt("contains_b") == False # B should not be found
            
    def __run(self):
        self.population = pyflamegpu.AgentVector(self.agent, 1)
        # CudaModel must be declared here
        # As the initial call to constructor fixes the agent population
        # This means if we haven't called model.newAgent(agent) first
        self.cudaSimulation = pyflamegpu.CUDASimulation(self.model)
        self.cudaSimulation.SimulationConfig().steps = 2
        # This fails as agentMap is empty
        self.cudaSimulation.setPopulationData(self.population)
        self.cudaSimulation.simulate()
        # The negative of this, is that cudaSimulation is inaccessible within the test!
        # So copy across population data here
        self.cudaSimulation.getPopulationData(self.population)
    


class AgentEnvironmentTest(TestCase):

    # Scalar environment variable tests

    def test_get_float(self): 
        ms = MiniSim()
        ms.run_agent_function_test("float", "Float")
        
    def test_get_double(self): 
        ms = MiniSim()
        ms.run_agent_function_test("double", "Double")
        
    def test_get_int8(self): 
        ms = MiniSim()
        ms.run_agent_function_test("int8_t", "Int8")
       
    def test_get_int16(self): 
        ms = MiniSim()
        ms.run_agent_function_test("int16_t", "Int16")
        
    def test_get_int32(self): 
        ms = MiniSim()
        ms.run_agent_function_test("int32_t", "Int32")
 
    def test_get_int64(self): 
        ms = MiniSim()
        ms.run_agent_function_test("int64_t", "Int64")
     
    def test_get_uint8(self): 
        ms = MiniSim()
        ms.run_agent_function_test("uint8_t", "UInt8")
        
    def test_get_uint16(self): 
        ms = MiniSim()
        ms.run_agent_function_test("uint16_t", "UInt16")
        
    def test_get_uint32(self): 
        ms = MiniSim()
        ms.run_agent_function_test("uint32_t", "UInt32")
        
    def test_get_uint64(self): 
        ms = MiniSim()
        ms.run_agent_function_test("uint64_t", "UInt64")
        
    # Array versions

    def test_get_array_element_float(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("float", "Float")
        
    def test_get_array_element_double(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("double", "Double")
        
    def test_get_array_element_int8(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("int8_t", "Int8")
       
    def test_get_array_element_int16(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("int16_t", "Int16")
        
    def test_get_array_element_int32(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("int32_t", "Int32")
 
    def test_get_array_element_int64(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("int64_t", "Int64")
     
    def test_get_array_element_uint8(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("uint8_t", "UInt8")
        
    def test_get_array_element_uint16(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("uint16_t", "UInt16")
        
    def test_get_array_element_uint32(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("uint32_t", "UInt32")
        
    def test_get_array_element_uint64(self): 
        ms = MiniSim()
        ms.run_agent_function_array_test("uint64_t", "UInt64")
        