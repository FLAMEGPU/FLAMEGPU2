import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

TEST_LEN = 256
AGENT_COUNT = 5

# @note seeds 0 and 1 conflict with std::linear_congruential_engine, the default on GCC so using mt19937 to avoid this.
args_1 =  ["process.exe", "-r", "0", "-s", "1" ]
args_2 =  ["process.exe", "-r", "1", "-s", "1" ]

# limits

INT16_MAX = 32767
INT32_MAX = 2147483647
INT64_MAX = 9223372036854775807
UINT16_MAX = int("0xffff", 16)
UINT32_MAX = int("0xffffffff", 16)
UINT64_MAX = int("0xffffffffffffffff",16)

class step_func(pyflamegpu.HostFunctionCallback):
    def __init__(self, function, Type, arga, argb):
        """
        arga and argb mayebe either min/max or mena/stdv
        """
        super().__init__()
        self.Type = Type
        self.function = function
        self.reset_out()
        self.arga = arga
        self.argb = argb

    def reset_out(self):
        self.out = [0] * TEST_LEN

    def run(self, FLAMEGPU):
        rand_func = getattr(FLAMEGPU.random, f"{self.function}{self.Type}")
        for i in range(TEST_LEN):
            # call the typed function and expect no throws
            if self.arga is not None and self.argb is not None:
                self.out[i] = rand_func(self.arga, self.argb)
            else:
                self.out[i] = rand_func()
            
    def assert_zero(self):
        # expect all values to be 0
        for i in range(TEST_LEN):
            assert self.out[i] == 0
            
    def assert_diff_zero(self):
        diff = 0
        for i in range(TEST_LEN):
            if self.out[i] != 0:
                diff += 1
        # expect at least one difference
        assert diff > 0
        
    def assert_diff_all(self):
        diff = 0
        for i in range(TEST_LEN):
            for j in range(TEST_LEN):
                if i != j:  
                    if self.out[i] != self.out[j]:
                        diff += 1
        # expect at least one difference
        assert diff > 0
        
    def assert_diff_list(self, other):
        diff = 0
        for i in range(TEST_LEN):
            if self.out[i] != other[i]:
                diff += 1
        # expect at least one difference
        assert diff > 0
        
    def assert_diff_same(self, other):
        diff = 0
        for i in range(TEST_LEN):
            if self.out[i] != other[i]:
                diff += 1
        # expect at least one difference
        assert diff == 0


class step_func_uniform_range(pyflamegpu.HostFunctionCallback):
    def __init__(self, Type, min, max):
        """
        arga and argb mayebe either min/max or mena/stdv
        """
        super().__init__()
        self.Type = Type
        self.reset_out()
        self.min = min
        self.max = max

    def reset_out(self):
        self.out = [0] * TEST_LEN

    def run(self, FLAMEGPU):
        rand_func = getattr(FLAMEGPU.random, f"uniform{self.Type}")
        for i in range(TEST_LEN):
            # call the typed function and expect no throws
            if self.min is not None and self.max is not None:
                self.out[i] = rand_func(self.min, self.max)
            else:
                self.out[i] = rand_func()
                
    def assert_range(self):
        for i in range(TEST_LEN):
            if self.min is not None:
                assert self.out[i] <= self.max
            else:
                assert self.out[i] <= 1.0
            if self.max is not None:
                assert self.out[i] >= self.min
            else:
                assert self.out[i] >= 0.0

class MiniSim():

    def __init__(self):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        self.ed = self.model.Environment()
        self.population = pyflamegpu.AgentPopulation(self.agent, TEST_LEN)
        for i in range(AGENT_COUNT):
            instance = self.population.getNextInstance()
    
    def run(self, args): 
        self.cuda_model = pyflamegpu.CUDASimulation(self.model)
        self.cuda_model.SimulationConfig().steps = 1
        self.cuda_model.setPopulationData(self.population)
        if len(args) > 0:
            self.cuda_model.initialise(args)
        
        self.cuda_model.simulate()
        # The negative of this, is that cuda_model is inaccessible within the test!
        # So copy across population data here
        self.cuda_model.getPopulationData(self.population)
        
    def rand_test(self, function, Type, arga=None, argb=None):
        """
        Main contexts of tests are parameterised in this function by random function type (normal/uniform) and the variable type
        """
        step = step_func(function, Type, arga, argb)
        self.model.addStepFunctionCallback(step)
        # Initially 0
        step.assert_zero()
        # Seed RNG
        self.run(args_1)
        # Value has changed
        step.assert_diff_zero()
        # Multiple calls == different values
        step.assert_diff_all()
        _out = step.out.copy()
        # reset out
        step.reset_out()
        # Different Seed
        self.run(args_2)
        # Value has changed
        step.assert_diff_zero
        # New Seed == new sequence
        step.assert_diff_list(_out)
        # reset
        step.reset_out()
        # First Seed
        self.run(args_1)
        # Value has changed
        step.assert_diff_zero()
        # Old Seed == old values
        step.assert_diff_same(_out)
        
    def range_test(self, Type, min=None, max=None):
        step = step_func_uniform_range(Type, min, max)
        self.model.addStepFunctionCallback(step)
        self.run([])
        step.assert_range()
        

class HostRandomTest(TestCase):

    def test_uniform_float(self):
        ms = MiniSim()
        ms.rand_test("uniform", "Float")
        
    def test_uniform_double(self):
        ms = MiniSim()
        ms.rand_test("uniform", "Double")
        
    def test_normal_float(self):
        ms = MiniSim()
        ms.rand_test("normal", "Float")
        
    def test_normal_double(self):
        ms = MiniSim()
        ms.rand_test("normal", "Double")
        
    def test_lognormal_float(self):
        ms = MiniSim()
        ms.rand_test("logNormal", "Float", 0, 1)
        
    def test_lognormal_double(self):
        ms = MiniSim()
        ms.rand_test("logNormal", "Double", 0, 1)
        
    def test_uniform_int16(self):
        ms = MiniSim()
        ms.rand_test("uniform", "Int16", 0, INT16_MAX)
        
    def test_uniform_int32(self):
        ms = MiniSim()
        ms.rand_test("uniform", "Int32", 0, INT32_MAX)
        
    def test_uniform_int64(self):
        ms = MiniSim()
        ms.rand_test("uniform", "Int64", 0, INT64_MAX)
        
    def test_uniform_uint16(self):
        ms = MiniSim()
        ms.rand_test("uniform", "UInt16", 0, UINT16_MAX)
        
    def test_uniform_uint32(self):
        ms = MiniSim()
        ms.rand_test("uniform", "UInt32", 0, UINT32_MAX)
        
    def test_uniform_uint64(self):
        ms = MiniSim()
        ms.rand_test("uniform", "UInt64", 0, UINT64_MAX)


    # Range tests


    def test_uniform_float_range(Self):
        ms = MiniSim()
        ms.range_test("Float")
        
    def test_uniform_double_range(Self):
        ms = MiniSim()
        ms.range_test("Double")
        
    def test_uniform_int16_range(Self):
        ms = MiniSim()
        ms.range_test("Int16", int(INT16_MAX*0.5), int(INT16_MAX*0.5))
        
    def test_uniform_uint16_range(Self):
        ms = MiniSim()
        ms.range_test("UInt16", int(UINT16_MAX*0.25), int(UINT16_MAX*0.75))
        
    def test_uniform_int32_range(Self):
        ms = MiniSim()
        ms.range_test("Int32", int(INT32_MAX*0.5), int(INT32_MAX*0.5))
        
    def test_uniform_uint32_range(Self):
        ms = MiniSim()
        ms.range_test("UInt32", int(UINT32_MAX*0.25), int(UINT32_MAX*0.75))
        
    def test_uniform_int64_range(Self):
        ms = MiniSim()
        ms.range_test("Int64", int(INT64_MAX*0.5), int(INT64_MAX*0.5))
        
    def test_uniform_uint64_range(Self):
        ms = MiniSim()
        ms.range_test("UInt64", int(UINT64_MAX*0.25), int(UINT64_MAX*0.75))
        
    





