import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

TEST_LEN = 256
AGENT_COUNT = 5

INT8_MAX = 127 
INT16_MAX = 32767  
INT32_MAX = 2147483647  
INT64_MAX = 9223372036854775807

class step_func_sum(pyflamegpu.HostFunctionCallback):
    def __init__(self, Type, variable):
        super().__init__()
        self.Type = Type
        self.variable = variable
        self.sum = 0


    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent")
        sum_func = getattr(agent, f"sum{self.Type}")
        self.sum = sum_func(self.variable)

                
    def assert_sum(self, expected):
        assert self.sum == expected
            
    

class MiniSim():

    def __init__(self, Type, variable, range_max=INT16_MAX):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        
        # add agent variable
        new_var_func = getattr(self.agent, f"newVariable{Type}")
        new_var_func(variable)
        
        # create a sum step function
        self.step = step_func_sum(Type, variable)
        self.model.addStepFunctionCallback(self.step)
        
        # create a population and set random values to be sumed
        self.population = pyflamegpu.AgentPopulation(self.agent, TEST_LEN)
        rand.seed() # Seed does not matter 
        self.expected_sum = 0;
        for i in range(TEST_LEN): 
            instance = self.population.getNextInstance()
            value = rand.randint(0, range_max) # value not actually important as suming the 0s
            # accumulate
            self.expected_sum += value
            # set instance value (will be cast to correct type)
            set_var_func = getattr(instance, f"setVariable{Type}")
            set_var_func(variable, value)

    
    def run(self): 
        self.cuda_model = pyflamegpu.CUDAAgentModel(self.model)
        self.cuda_model.SimulationConfig().steps = 1
        self.cuda_model.setPopulationData(self.population)      
        self.cuda_model.simulate()
        self.cuda_model.getPopulationData(self.population)
        # check assertions
        self.step.assert_sum(self.expected_sum)
        

class HostReductionTest(TestCase):

    def test_SumFloat(self):
        ms = MiniSim("Float", "float")         
        ms.run()
        
    def test_SumDouble(self):
        ms = MiniSim("Double", "double")         
        ms.run()
            
    def test_SumInt8(self):
        ms = MiniSim("Int8", "int8", INT8_MAX)         
        ms.run()
        
    def test_SumUInt8(self):
        ms = MiniSim("UInt8", "uint8", INT8_MAX)         
        ms.run()
        
    def test_SumInt16(self):
        ms = MiniSim("Int16", "int16", INT16_MAX)         
        ms.run()
        
    def test_SumUInt16(self):
        ms = MiniSim("UInt16", "uint16", INT16_MAX)         
        ms.run()
     
    def test_SumInt32(self):
        ms = MiniSim("Int32", "int32")         
        ms.run()
        
    def test_SumUInt32(self):
        ms = MiniSim("UInt32", "uint32")         
        ms.run()     

    def test_SumInt64(self):
        ms = MiniSim("Int64", "int64")         
        ms.run()
        
    def test_SumUInt64(self):
        ms = MiniSim("UInt64", "uint64")         
        ms.run() 

