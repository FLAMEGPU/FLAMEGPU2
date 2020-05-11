import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

TEST_LEN = 256

INT8_MAX = 127 
INT16_MAX = 32767  
INT32_MAX = 2147483647  
INT64_MAX = 9223372036854775807

class step_func_count(pyflamegpu.HostFunctionCallback):
    def __init__(self, Type, variable):
        super().__init__()
        self.Type = Type
        self.variable = variable
        self.count = 0


    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent")
        count_func = getattr(agent, f"count{self.Type}")
        self.count = count_func(self.variable, 0)

                
    def assert_count(self, expected):
        assert self.count == expected
            
    

class MiniSim():

    def __init__(self, Type, variable, range_max=INT32_MAX):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        
        # add agent variable
        new_var_func = getattr(self.agent, f"newVariable{Type}")
        new_var_func(variable)
        
        # create a count step function
        self.step = step_func_count(Type, variable)
        self.model.addStepFunctionCallback(self.step)
        
        # create a population and set random values to be counted
        self.population = pyflamegpu.AgentPopulation(self.agent, TEST_LEN)
        rand.seed() # Seed does not matter 
        self.expected_count = 0;
        for i in range(TEST_LEN): 
            instance = self.population.getNextInstance()
            if (i < TEST_LEN / 2):
                value = rand.randint(0, range_max) # value not actually important as counting the 0s
            else: 
                value = 0
            # save the number of occurrences of 0
            if value == 0:
                self.expected_count += 1
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
        self.step.assert_count(self.expected_count)
        

class HostReductionTest(TestCase):

    def test_CountFloat(self):
        ms = MiniSim("Float", "float")         
        ms.run()
        
    def test_CountDouble(self):
        ms = MiniSim("Double", "double")         
        ms.run()
            
    def test_CountInt8(self):
        ms = MiniSim("Int8", "int8", INT8_MAX)         
        ms.run()
        
    def test_CountUInt8(self):
        ms = MiniSim("UInt8", "uint8", INT8_MAX)         
        ms.run()
        
    def test_CountInt16(self):
        ms = MiniSim("Int16", "int16", INT16_MAX)         
        ms.run()
        
    def test_CountUInt16(self):
        ms = MiniSim("UInt16", "uint16", INT16_MAX)         
        ms.run()
     
    def test_CountInt32(self):
        ms = MiniSim("Int32", "int32")         
        ms.run()
        
    def test_CountUInt32(self):
        ms = MiniSim("UInt32", "uint32")         
        ms.run()     

    def test_CountInt64(self):
        ms = MiniSim("Int64", "int64")         
        ms.run()
        
    def test_CountUInt64(self):
        ms = MiniSim("UInt64", "uint64")         
        ms.run() 

