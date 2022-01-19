import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

TEST_LEN = 101

class step_func_mean_sd(pyflamegpu.HostFunctionCallback):
    def __init__(self, Type, variable):
        super().__init__()
        self.Type = Type
        self.variable = variable
        self.mean = 0
        self.sd = 0

    def run(self, FLAMEGPU):
        agent = FLAMEGPU.agent("agent")
        sum_func = getattr(agent, f"meanStandardDeviation{self.Type}")
        self.mean, self.sd = sum_func(self.variable)

    def assert_mean_sd(self, expected_mean, expected_sd):
        assert abs(expected_mean - self.mean) < 0.0001
        assert abs(expected_sd - self.sd) < 0.0001

class MiniSim():

    def __init__(self, Type, variable):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        
        # add agent variable
        new_var_func = getattr(self.agent, f"newVariable{Type}")
        new_var_func(variable)
        
        # create a mean sd step function
        self.step = step_func_mean_sd(Type, variable)
        self.model.addStepFunctionCallback(self.step)
        
        # create a population and set random values to be sumed
        self.population = pyflamegpu.AgentVector(self.agent, TEST_LEN)
        rand.seed() # Seed does not matter 
        self.expected_sum = 0
        i = 0
        for instance in self.population:
            self.expected_sum += i
            # set instance value (will be cast to correct type)
            set_var_func = getattr(instance, f"setVariable{Type}")
            set_var_func(variable, i)
            i += 1

    
    def run(self): 
        self.cudaSimulation = pyflamegpu.CUDASimulation(self.model)
        self.cudaSimulation.SimulationConfig().steps = 1
        self.cudaSimulation.setPopulationData(self.population)      
        self.cudaSimulation.simulate()
        # check assertions
        self.step.assert_mean_sd(self.expected_sum/101, 29.15476)
        

class HostReductionTest(TestCase):

    def test_SumFloat(self):
        ms = MiniSim("Float", "float")         
        ms.run()
        
    def test_SumDouble(self):
        ms = MiniSim("Double", "double")         
        ms.run()
            
    def test_SumInt8(self):
        ms = MiniSim("Int8", "int8")         
        ms.run()
        
    def test_SumUInt8(self):
        ms = MiniSim("UInt8", "uint8")         
        ms.run()
        
    def test_SumInt16(self):
        ms = MiniSim("Int16", "int16")         
        ms.run()
        
    def test_SumUInt16(self):
        ms = MiniSim("UInt16", "uint16")         
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

