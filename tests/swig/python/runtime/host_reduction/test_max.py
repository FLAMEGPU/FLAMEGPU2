import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

TEST_LEN = 256

INT8_MAX = 127 
INT16_MAX = 32767  
INT32_MAX = 2147483647  
INT64_MAX = 9223372036854775807

class step_func_max(pyflamegpu.HostFunctionCallback):
    def __init__(self, Type, variable):
        super().__init__()
        self.Type = Type
        self.variable = variable
        self.max = 0


    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent")
        max_func = getattr(agent, f"max{self.Type}")
        self.max = max_func(self.variable)

                
    def assert_max(self, expected):
        assert self.max == expected
            
    

class MiniSim():

    def __init__(self, Type, variable, range_max=INT16_MAX):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        
        # add agent variable
        new_var_func = getattr(self.agent, f"newVariable{Type}")
        new_var_func(variable)
        
        # create a count step function
        self.step = step_func_max(Type, variable)
        self.model.addStepFunctionCallback(self.step)
        
        # create a population and set random values to be counted
        self.population = pyflamegpu.AgentVector(self.agent, TEST_LEN)
        rand.seed() # Seed does not matter 
        self.expected_max = 0;
        for instance in self.population:
            value = rand.randint(0, range_max)
            if (value > self.expected_max):
                self.expected_max = value
            # set instance value (will be cast to correct type)
            set_var_func = getattr(instance, f"setVariable{Type}")
            set_var_func(variable, value)

    def run(self): 
        self.cuda_model = pyflamegpu.CUDASimulation(self.model)
        self.cuda_model.SimulationConfig().steps = 1
        self.cuda_model.setPopulationData(self.population)      
        self.cuda_model.simulate()
        self.cuda_model.getPopulationData(self.population)
        # check assertions
        self.step.assert_max(self.expected_max)
        

class HostReductionTest(TestCase):


    def test_MaxFloat(self):
        ms = MiniSim("Float", "float")         
        ms.run()
        
    def test_MaxDouble(self):
        ms = MiniSim("Double", "double")         
        ms.run()
        
    def test_MaxInt8(self):
        ms = MiniSim("Int8", "int8", INT8_MAX)         
        ms.run()
        
    def test_MaxUInt8(self):
        ms = MiniSim("UInt8", "uint8", INT8_MAX)         
        ms.run() 
       
    def test_MaxInt16(self):
        ms = MiniSim("Int16", "int16", INT16_MAX)         
        ms.run()
        
    def test_MaxUInt16(self):
        ms = MiniSim("UInt16", "uint16", INT16_MAX)         
        ms.run()
     
    def test_MaxInt32(self):
        ms = MiniSim("Int32", "int32")         
        ms.run()
        
    def test_MaxUInt32(self):
        ms = MiniSim("UInt32", "uint32")         
        ms.run()     

    def test_MaxInt64(self):
        ms = MiniSim("Int64", "int64")         
        ms.run()
        
    def test_MaxUInt64(self):
        ms = MiniSim("UInt64", "uint64")         
        ms.run() 

  