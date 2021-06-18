import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

TEST_LEN = 256

INT8_MAX = 127 
INT16_MAX = 32767  
INT32_MAX = 2147483647  
INT64_MAX = 9223372036854775807

class step_func_min(pyflamegpu.HostFunctionCallback):
    def __init__(self, Type, variable):
        super().__init__()
        self.Type = Type
        self.variable = variable
        self.min = 0


    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent")
        min_func = getattr(agent, f"min{self.Type}")
        self.min = min_func(self.variable)

                
    def assert_min(self, expected):
        assert self.min == expected
            
    

class MiniSim():

    def __init__(self, Type, variable, range_max=INT16_MAX):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        
        # add agent variable
        new_var_func = getattr(self.agent, f"newVariable{Type}")
        new_var_func(variable)
        
        # create a count step function
        self.step = step_func_min(Type, variable)
        self.model.addStepFunctionCallback(self.step)
        
        # create a population and set random values to be counted
        self.population = pyflamegpu.AgentVector(self.agent, TEST_LEN)
        rand.seed() # Seed does not matter 
        self.expected_min = range_max;
        for instance in self.population:
            value = rand.randint(0, range_max)
            if (value < self.expected_min):
                self.expected_min = value
            # set instance value (will be cast to correct type)
            set_var_func = getattr(instance, f"setVariable{Type}")
            set_var_func(variable, value)

    def run(self): 
        self.cudaSimulation = pyflamegpu.CUDASimulation(self.model)
        self.cudaSimulation.SimulationConfig().steps = 1
        self.cudaSimulation.setPopulationData(self.population)      
        self.cudaSimulation.simulate()
        self.cudaSimulation.getPopulationData(self.population)
        # check assertions
        self.step.assert_min(self.expected_min)
        

class HostReductionTest(TestCase):


    def test_MinFloat(self):
        ms = MiniSim("Float", "float")         
        ms.run()
        
    def test_MinDouble(self):
        ms = MiniSim("Double", "double")         
        ms.run()
  
    def test_MinInt8(self):
        ms = MiniSim("Int8", "int8", INT8_MAX)         
        ms.run()
        
    def test_MinUInt8(self):
        ms = MiniSim("UInt8", "uint8", INT8_MAX)         
        ms.run()
  
    def test_MinInt16(self):
        ms = MiniSim("Int16", "int16", INT16_MAX)         
        ms.run()
        
    def test_MinUInt16(self):
        ms = MiniSim("UInt16", "uint16", INT16_MAX)         
        ms.run()
     
    def test_MinInt32(self):
        ms = MiniSim("Int32", "int32")         
        ms.run()
        
    def test_MinUInt32(self):
        ms = MiniSim("UInt32", "uint32")         
        ms.run()     

    def test_MinInt64(self):
        ms = MiniSim("Int64", "int64")         
        ms.run()
        
    def test_MinUInt64(self):
        ms = MiniSim("UInt64", "uint64")         
        ms.run() 

  