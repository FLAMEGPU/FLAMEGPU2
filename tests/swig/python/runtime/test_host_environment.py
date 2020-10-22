"""
 # Tests of class: HostEnvironment
 # 
 # Tests cover:
 # > get() [per supported type, individual/array/element]
 # > set() [per supported type, individual/array/element]
 # > add() [per supported type, individual/array]
 # > remove() (implied by exception tests)
 # exceptions
"""

import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

TEST_LEN = 256
TEST_VALUE = 12
TEST_ARRAY_OFFSET = 2
TEST_ARRAY_LEN = 4

class MiniSim():

    def __init__(self):
        self.model = pyflamegpu.ModelDescription("model")
        self.agent = self.model.newAgent("agent")
        self.ed = self.model.Environment()
        self.population = pyflamegpu.AgentPopulation(self.agent, TEST_LEN)
        self.ed.newPropertyFloat("float_", TEST_VALUE)
        self.ed.newPropertyDouble("double_", TEST_VALUE)
        self.ed.newPropertyInt8("int8_", TEST_VALUE)
        self.ed.newPropertyUInt8("uint8_", TEST_VALUE)
        self.ed.newPropertyInt16("int16_", TEST_VALUE)
        self.ed.newPropertyUInt16("uint16_", TEST_VALUE)
        self.ed.newPropertyInt32("int32_", TEST_VALUE)
        self.ed.newPropertyUInt32("uint32_", TEST_VALUE)
        self.ed.newPropertyInt64("int64_", TEST_VALUE)
        self.ed.newPropertyUInt64("uint64_", TEST_VALUE)
        self.ed.newPropertyFloat("read_only", TEST_VALUE, True)
        
        self.ed.newPropertyArrayFloat("float_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayDouble("double_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayInt8("int8_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayUInt8("uint8_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayInt16("int16_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayUInt16("uint16_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayInt32("int32_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayUInt32("uint32_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayInt64("int64_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayUInt64("uint64_a_", 4, MiniSim.make_array())
        self.ed.newPropertyArrayInt("read_only_a", 4, MiniSim.make_array(), True)
    
    @staticmethod
    def make_array(offset = 0): 
        a = []
        for i in range(TEST_ARRAY_LEN):
            a.append(i + 1 + offset)
        return a
    
    def run(self, steps = 2): 
        # CudaModel must be declarself.ed here
        # As the initial call to constructor fixes the agent population
        # This means if we haven't callself.ed model.newAgent(agent) first
        self.cuda_model = pyflamegpu.CUDASimulation(self.model)
        self.cuda_model.SimulationConfig().steps = steps
        # This fails as agentMap is empty
        self.cuda_model.setPopulationData(self.population)
        self.cuda_model.simulate()
        # The negative of this, is that cuda_model is inaccessible within the test!
        # So copy across population data here
        self.cuda_model.getPopulationData(self.population)


class get_set_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.float = FLAMEGPU.environment.setPropertyFloat("float_", float(TEST_VALUE) * 2)
        # Test Get (Host func set value)
        self.float_ = FLAMEGPU.environment.getPropertyFloat("float_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyFloat("float_", float(TEST_VALUE))
        
    def apply_assertions(self):
        assert self.float == float(TEST_VALUE)
        assert self.float_ == float(TEST_VALUE * 2)
        
class get_set_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.double = FLAMEGPU.environment.setPropertyDouble("double_", TEST_VALUE * 2.0)
        # Test Get (Host func set value)
        self.double_ = FLAMEGPU.environment.getPropertyDouble("double_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyDouble("double_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.double == TEST_VALUE
        assert self.double_ == TEST_VALUE * 2.0



class get_set_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int8 = FLAMEGPU.environment.setPropertyInt8("int8_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int8_ = FLAMEGPU.environment.getPropertyInt8("int8_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyInt8("int8_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int8 == TEST_VALUE
        assert self.int8_ == TEST_VALUE * 2

class get_set_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int16 = FLAMEGPU.environment.setPropertyInt16("int16_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int16_ = FLAMEGPU.environment.getPropertyInt16("int16_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyInt16("int16_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int16 == TEST_VALUE
        assert self.int16_ == TEST_VALUE * 2
        
class get_set_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.uint16 = FLAMEGPU.environment.setPropertyUInt16("uint16_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.uint16_ = FLAMEGPU.environment.getPropertyUInt16("uint16_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyUInt16("uint16_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.uint16 == TEST_VALUE
        assert self.uint16_ == TEST_VALUE * 2
        
class get_set_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int32 = FLAMEGPU.environment.setPropertyInt32("int32_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int32_ = FLAMEGPU.environment.getPropertyInt32("int32_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyInt32("int32_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int32 == TEST_VALUE
        assert self.int32_ == TEST_VALUE * 2
        
        
class get_set_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.uint32 = FLAMEGPU.environment.setPropertyUInt32("uint32_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.uint32_ = FLAMEGPU.environment.getPropertyUInt32("uint32_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyUInt32("uint32_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.uint32 == TEST_VALUE
        assert self.uint32_ == TEST_VALUE * 2

class get_set_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int64 = FLAMEGPU.environment.setPropertyInt64("int64_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int64_ = FLAMEGPU.environment.getPropertyInt64("int64_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyInt64("int64_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int64 == TEST_VALUE
        assert self.int64_ == TEST_VALUE * 2      

class get_set_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.uint64 = FLAMEGPU.environment.setPropertyUInt64("uint64_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.uint64_ = FLAMEGPU.environment.getPropertyUInt64("uint64_")
        # Reset for next iteration
        FLAMEGPU.environment.setPropertyUInt64("uint64_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.uint64 == TEST_VALUE
        assert self.uint64_ == TEST_VALUE * 2      

# Arrays

class get_set_array_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_float_a = FLAMEGPU.environment.setPropertyArrayFloat("float_a_", init2)
        self.get_float_a = FLAMEGPU.environment.getPropertyArrayFloat("float_a_")
        FLAMEGPU.environment.setPropertyArrayFloat("float_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_float_a) == init1
        assert list(self.get_float_a) == init2
        
class get_set_array_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_double_a = FLAMEGPU.environment.setPropertyArrayDouble("double_a_", init2)
        self.get_double_a = FLAMEGPU.environment.getPropertyArrayDouble("double_a_")
        FLAMEGPU.environment.setPropertyArrayDouble("double_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_double_a) == init1
        assert list(self.get_double_a) == init2


       
class get_set_array_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_int8_a = FLAMEGPU.environment.setPropertyArrayInt8("int8_a_", init2)
        self.get_int8_a = FLAMEGPU.environment.getPropertyArrayInt8("int8_a_")
        FLAMEGPU.environment.setPropertyArrayInt8("int8_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_int8_a) == init1
        assert list(self.get_int8_a) == init2
        
class get_set_array_uint8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_uint8_a = FLAMEGPU.environment.setPropertyArrayUInt8("uint8_a_", init2)
        self.get_uint8_a = FLAMEGPU.environment.getPropertyArrayUInt8("uint8_a_")
        FLAMEGPU.environment.setPropertyArrayUInt8("uint8_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_uint8_a) == init1
        assert list(self.get_uint8_a) == init2
 
class get_set_array_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_int16_a = FLAMEGPU.environment.setPropertyArrayInt16("int16_a_", init2)
        self.get_int16_a = FLAMEGPU.environment.getPropertyArrayInt16("int16_a_")
        FLAMEGPU.environment.setPropertyArrayInt16("int16_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_int16_a) == init1
        assert list(self.get_int16_a) == init2
        
class get_set_array_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_uint16_a = FLAMEGPU.environment.setPropertyArrayUInt16("uint16_a_", init2)
        self.get_uint16_a = FLAMEGPU.environment.getPropertyArrayUInt16("uint16_a_")
        FLAMEGPU.environment.setPropertyArrayUInt16("uint16_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_uint16_a) == init1
        assert list(self.get_uint16_a) == init2

class get_set_array_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_int32_a = FLAMEGPU.environment.setPropertyArrayInt32("int32_a_", init2)
        self.get_int32_a = FLAMEGPU.environment.getPropertyArrayInt32("int32_a_")
        FLAMEGPU.environment.setPropertyArrayInt32("int32_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_int32_a) == init1
        assert list(self.get_int32_a) == init2
        
class get_set_array_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_uint32_a = FLAMEGPU.environment.setPropertyArrayUInt32("uint32_a_", init2)
        self.get_uint32_a = FLAMEGPU.environment.getPropertyArrayUInt32("uint32_a_")
        FLAMEGPU.environment.setPropertyArrayUInt32("uint32_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_uint32_a) == init1
        assert list(self.get_uint32_a) == init2    

class get_set_array_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_int64_a = FLAMEGPU.environment.setPropertyArrayInt64("int64_a_", init2)
        self.get_int64_a = FLAMEGPU.environment.getPropertyArrayInt64("int64_a_")
        FLAMEGPU.environment.setPropertyArrayInt64("int64_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_int64_a) == init1
        assert list(self.get_int64_a) == init2
        
class get_set_array_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        self.set_uint64_a = FLAMEGPU.environment.setPropertyArrayUInt64("uint64_a_", init2)
        self.get_uint64_a = FLAMEGPU.environment.getPropertyArrayUInt64("uint64_a_")
        FLAMEGPU.environment.setPropertyArrayUInt64("uint64_a_", init1)
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        init2 = MiniSim.make_array(TEST_ARRAY_OFFSET)
        assert list(self.set_uint64_a) == init1
        assert list(self.get_uint64_a) == init2 
        
# Array Elements

class get_set_array_element_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_float_a = FLAMEGPU.environment.setPropertyFloat("float_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_float_a = FLAMEGPU.environment.getPropertyFloat("float_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyFloat("float_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_float_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_float_a == TEST_VALUE * 2    

class get_set_array_element_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_double_a = FLAMEGPU.environment.setPropertyDouble("double_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_double_a = FLAMEGPU.environment.getPropertyDouble("double_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyDouble("double_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_double_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_double_a == TEST_VALUE * 2

class get_set_array_element_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int8_a = FLAMEGPU.environment.setPropertyInt8("int8_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int8_a = FLAMEGPU.environment.getPropertyInt8("int8_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyInt8("int8_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int8_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int8_a == TEST_VALUE * 2    

class get_set_array_element_uint8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint8_a = FLAMEGPU.environment.setPropertyUInt8("uint8_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint8_a = FLAMEGPU.environment.getPropertyUInt8("uint8_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyUInt8("uint8_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_uint8_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_uint8_a == TEST_VALUE * 2  

class get_set_array_element_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int16_a = FLAMEGPU.environment.setPropertyInt16("int16_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int16_a = FLAMEGPU.environment.getPropertyInt16("int16_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyInt16("int16_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int16_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int16_a == TEST_VALUE * 2    

class get_set_array_element_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint16_a = FLAMEGPU.environment.setPropertyUInt16("uint16_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint16_a = FLAMEGPU.environment.getPropertyUInt16("uint16_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyUInt16("uint16_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_uint16_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_uint16_a == TEST_VALUE * 2          

class get_set_array_element_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int32_a = FLAMEGPU.environment.setPropertyInt32("int32_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int32_a = FLAMEGPU.environment.getPropertyInt32("int32_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyInt32("int32_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int32_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int32_a == TEST_VALUE * 2    

class get_set_array_element_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint32_a = FLAMEGPU.environment.setPropertyUInt32("uint32_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint32_a = FLAMEGPU.environment.getPropertyUInt32("uint32_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyUInt32("uint32_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_uint32_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_uint32_a == TEST_VALUE * 2  

class get_set_array_element_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int64_a = FLAMEGPU.environment.setPropertyInt64("int64_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int64_a = FLAMEGPU.environment.getPropertyInt64("int64_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyInt64("int64_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int64_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int64_a == TEST_VALUE * 2    

class get_set_array_element_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint64_a = FLAMEGPU.environment.setPropertyUInt64("uint64_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint64_a = FLAMEGPU.environment.getPropertyUInt64("uint64_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setPropertyUInt64("uint64_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_uint64_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_uint64_a == TEST_VALUE * 2  
        
# Exception ProprtyType

class exception_property_type_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("float_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("float_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 

class exception_property_type_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("double_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("double_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 
        
class exception_property_type_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("int8_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("int8_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 
        
class exception_property_type_uint8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("uint8_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("uint8_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 
        
class exception_property_type_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("int16_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("int16_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 
        
class exception_property_type_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("uint16_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("uint16_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 

class exception_property_type_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("int32_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("int32_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 
        
class exception_property_type_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyUInt64("uint32_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("uint32_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType"

class exception_property_type_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyFloat("int64_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayFloat("int64_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType" 
        
class exception_property_type_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        try:
            FLAMEGPU.environment.setPropertyFloat("uint64_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayFloat("uint64_", init1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.e1 == "InvalidEnvPropertyType"
        assert self.e2 == "InvalidEnvPropertyType"

# Exceptions Length    

class exception_property_length_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayFloat("float_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayFloat("float_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayFloat("float_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"    

class exception_property_length_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayDouble("double_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayDouble("double_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayDouble("double_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"

class exception_property_length_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayInt8("int8_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayInt8("int8_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayInt8("int8_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"  

class exception_property_length_uint8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayUInt8("uint8_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayUInt8("uint8_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt8("uint8_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"           

class exception_property_length_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayInt16("int16_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayInt16("int16_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayInt16("int16_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"  

class exception_property_length_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayUInt16("uint16_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayUInt16("uint16_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt16("uint16_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException" 

class exception_property_length_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayInt32("int32_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayInt32("int32_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayInt32("int32_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"  

class exception_property_length_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayUInt32("uint32_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayUInt32("uint32_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt32("uint32_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException" 
        
class exception_property_length_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayInt64("int64_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayInt64("int64_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayInt64("int64_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"  

class exception_property_length_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        b = MiniSim.make_array()
        b1 = [0] * 2
        b2 = [0] * 8
        FLAMEGPU.environment.setPropertyArrayUInt64("uint64_a_", b)
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("uint64_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setPropertyArrayUInt64("uint64_a_", b2)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException" 
        
# Exception Range

class exception_property_range_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyFloat("float_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyFloat("float_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException" 
        
class exception_property_range_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyDouble("double_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyDouble("double_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"
        
class exception_property_range_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyInt8("int8_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyInt8("int8_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"
        
class exception_property_range_uint8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyUInt8("uint8_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyUInt8("uint8_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"
        
class exception_property_range_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyInt16("int16_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyInt16("int16_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"
        
class exception_property_range_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyUInt16("uint16_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyUInt16("uint16_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"
        
class exception_property_range_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyInt32("int32_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyInt32("int32_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"
        
class exception_property_range_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyUInt32("uint32_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyUInt32("uint32_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"

class exception_property_range_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyInt64("int64_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyInt64("int64_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"
        
class exception_property_range_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(TEST_ARRAY_LEN): 
            try:
                FLAMEGPU.environment.setPropertyUInt64("uint64_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getPropertyUInt64("uint64_a_", TEST_ARRAY_LEN + i)
            except pyflamegpu.FGPURuntimeException as e:
                self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "OutOfBoundsException"
        assert self.e2 == "OutOfBoundsException"

# Other

class exception_property_doesnt_exist(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        try:
            FLAMEGPU.environment.getPropertyFloat("a")
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.getPropertyFloat("a", 1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        
    def apply_assertions(self):
        assert self.e1 == "InvalidEnvProperty"
        assert self.e2 == "InvalidEnvProperty"


class exception_property_read_only(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        a = TEST_VALUE
        b = [0]* TEST_ARRAY_LEN
        try:
            FLAMEGPU.environment.setPropertyFloat("read_only", a)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        # no throw
        FLAMEGPU.environment.getPropertyFloat("read_only")
        # array version
        try:
            FLAMEGPU.environment.setPropertyArrayInt("read_only_a", b)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        # no throw
        FLAMEGPU.environment.getPropertyInt("read_only_a")
        FLAMEGPU.environment.getPropertyInt("read_only_a", 1)     
        
    def apply_assertions(self):
        assert self.e1 == "ReadOnlyEnvProperty"
        assert self.e2 == "ReadOnlyEnvProperty"


class reserved_name_set_step(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.environment.setPropertyInt("_", 1)     
      
class reserved_name_set_array_step(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.environment.setPropertyIntArray2("_",  1, 2 )      
    


# Test Class
class HostEnvironmentTest(TestCase):


    def test_get_set_get_float(self):
        ms = MiniSim()
        step = get_set_float()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_double(self):
        ms = MiniSim()
        step = get_set_double()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_int8(self):
        ms = MiniSim()
        step = get_set_int8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_int16(self):
        ms = MiniSim()
        step = get_set_int16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_uint16(self):
        ms = MiniSim()
        step = get_set_uint16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_int32(self):
        ms = MiniSim()
        step = get_set_int32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_uint32(self):
        ms = MiniSim()
        step = get_set_uint32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_int64(self):
        ms = MiniSim()
        step = get_set_int32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_uint64(self):
        ms = MiniSim()
        step = get_set_uint32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
# Arrays

    def test_get_set_get_array_float(self):
        ms = MiniSim()
        step = get_set_array_float()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_double(self):
        ms = MiniSim()
        step = get_set_array_double()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_int8(self):
        ms = MiniSim()
        step = get_set_array_int8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_uint8(self):
        ms = MiniSim()
        step = get_set_array_uint8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_int32(self):
        ms = MiniSim()
        step = get_set_array_int32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_uint32(self):
        ms = MiniSim()
        step = get_set_array_uint32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_int64(self):
        ms = MiniSim()
        step = get_set_array_int64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_uint64(self):
        ms = MiniSim()
        step = get_set_array_uint64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()

# Array elements

    def test_get_set_get_array_element_float(self):
        ms = MiniSim()
        step = get_set_array_element_float()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_double(self):
        ms = MiniSim()
        step = get_set_array_element_double()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_int8(self):
        ms = MiniSim()
        step = get_set_array_element_int8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_uint8(self):
        ms = MiniSim()
        step = get_set_array_element_uint8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_int16(self):
        ms = MiniSim()
        step = get_set_array_element_int16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_uint16(self):
        ms = MiniSim()
        step = get_set_array_element_uint16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_int32(self):
        ms = MiniSim()
        step = get_set_array_element_int32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_uint32(self):
        ms = MiniSim()
        step = get_set_array_element_uint32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_int64(self):
        ms = MiniSim()
        step = get_set_array_element_int64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_get_set_get_array_element_uint64(self):
        ms = MiniSim()
        step = get_set_array_element_uint64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
# Exceptions

    def test_exception_property_type_float(self):
        ms = MiniSim()
        step = exception_property_type_float()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_double(self):
        ms = MiniSim()
        step = exception_property_type_double()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_int8(self):
        ms = MiniSim()
        step = exception_property_type_int8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_uint8(self):
        ms = MiniSim()
        step = exception_property_type_uint8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_int16(self):
        ms = MiniSim()
        step = exception_property_type_int16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_uint16(self):
        ms = MiniSim()
        step = exception_property_type_uint16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_int32(self):
        ms = MiniSim()
        step = exception_property_type_int32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_uint32(self):
        ms = MiniSim()
        step = exception_property_type_uint32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_int64(self):
        ms = MiniSim()
        step = exception_property_type_int64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_type_uint64(self):
        ms = MiniSim()
        step = exception_property_type_uint64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()

# Exceptions Length

    def test_exception_property_length_float(self):
        ms = MiniSim()
        step = exception_property_length_float()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_double(self):
        ms = MiniSim()
        step = exception_property_length_double()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_int8(self):
        ms = MiniSim()
        step = exception_property_length_int8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_uint8(self):
        ms = MiniSim()
        step = exception_property_length_uint8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_int16(self):
        ms = MiniSim()
        step = exception_property_length_int16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_uint16(self):
        ms = MiniSim()
        step = exception_property_length_uint16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_int32(self):
        ms = MiniSim()
        step = exception_property_length_int32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_uint32(self):
        ms = MiniSim()
        step = exception_property_length_uint32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_int64(self):
        ms = MiniSim()
        step = exception_property_length_int64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_length_uint64(self):
        ms = MiniSim()
        step = exception_property_length_uint64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()

# Exception Range

    def test_exception_property_range_float(self):
        ms = MiniSim()
        step = exception_property_range_float()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_range_double(self):
        ms = MiniSim()
        step = exception_property_range_double()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_range_int8(self):
        ms = MiniSim()
        step = exception_property_range_int8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_range_uint8(self):
        ms = MiniSim()
        step = exception_property_range_uint8()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_range_int16(self):
        ms = MiniSim()
        step = exception_property_range_int16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_range_uint16(self):
        ms = MiniSim()
        step = exception_property_range_uint16()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
    
    def test_exception_property_range_int32(self):
        ms = MiniSim()
        step = exception_property_range_int32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_range_uint32(self):
        ms = MiniSim()
        step = exception_property_range_uint32()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions() 

    def test_exception_property_range_int64(self):
        ms = MiniSim()
        step = exception_property_range_int64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_range_uint64(self):
        ms = MiniSim()
        step = exception_property_range_uint64()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
    
# Other

    def test_exception_property_doesnt_exist(self):
        ms = MiniSim()
        step = exception_property_doesnt_exist()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()
        
    def test_exception_property_read_only(self):
        ms = MiniSim()
        step = exception_property_read_only()
        ms.model.addStepFunctionCallback(step)
        # Test and apply assertions
        ms.run()
        step.apply_assertions()

    def reserved_name_set(self):
        model = pyflamegpu.ModelDescription("model")
        step = reserved_name_set_step()
        model.addStepFunctionCallback(step)
        sim = pyflamegpu.CUDASimulation(model)
        # Test and apply assertions
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "ReservedName"
        
    def reserved_name_set_array(self):
        model = pyflamegpu.ModelDescription("model")
        step = reserved_name_set_array_step()
        model.addStepFunctionCallback(step)
        sim = pyflamegpu.CUDASimulation(model)
        # Test and apply assertions
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "ReservedName"
