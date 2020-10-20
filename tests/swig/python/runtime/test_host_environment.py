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
        self.ed.addFloat("float_", TEST_VALUE)
        self.ed.addDouble("double_", TEST_VALUE)
        self.ed.addInt8("int8_", TEST_VALUE)
        self.ed.addUInt8("uint8_", TEST_VALUE)
        self.ed.addInt16("int16_", TEST_VALUE)
        self.ed.addUInt16("uint16_", TEST_VALUE)
        self.ed.addInt32("int32_", TEST_VALUE)
        self.ed.addUInt32("uint32_", TEST_VALUE)
        self.ed.addInt64("int64_", TEST_VALUE)
        self.ed.addUInt64("uint64_", TEST_VALUE)
        self.ed.addFloat("read_only", TEST_VALUE, True)
        
        self.ed.addArrayFloat("float_a_", 4, MiniSim.make_array())
        self.ed.addArrayDouble("double_a_", 4, MiniSim.make_array())
        self.ed.addArrayInt8("int8_a_", 4, MiniSim.make_array())
        self.ed.addArrayUInt8("uint8_a_", 4, MiniSim.make_array())
        self.ed.addArrayInt16("int16_a_", 4, MiniSim.make_array())
        self.ed.addArrayUInt16("uint16_a_", 4, MiniSim.make_array())
        self.ed.addArrayInt32("int32_a_", 4, MiniSim.make_array())
        self.ed.addArrayUInt32("uint32_a_", 4, MiniSim.make_array())
        self.ed.addArrayInt64("int64_a_", 4, MiniSim.make_array())
        self.ed.addArrayUInt64("uint64_a_", 4, MiniSim.make_array())
        self.ed.addArrayInt("read_only_a", 4, MiniSim.make_array(), True)
    
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
        self.float = FLAMEGPU.environment.setFloat("float_", float(TEST_VALUE) * 2)
        # Test Get (Host func set value)
        self.float_ = FLAMEGPU.environment.getFloat("float_")
        # Reset for next iteration
        FLAMEGPU.environment.setFloat("float_", float(TEST_VALUE))
        
    def apply_assertions(self):
        assert self.float == float(TEST_VALUE)
        assert self.float_ == float(TEST_VALUE * 2)
        
class get_set_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.double = FLAMEGPU.environment.setDouble("double_", TEST_VALUE * 2.0)
        # Test Get (Host func set value)
        self.double_ = FLAMEGPU.environment.getDouble("double_")
        # Reset for next iteration
        FLAMEGPU.environment.setDouble("double_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.double == TEST_VALUE
        assert self.double_ == TEST_VALUE * 2.0



class get_set_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int8 = FLAMEGPU.environment.setInt8("int8_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int8_ = FLAMEGPU.environment.getInt8("int8_")
        # Reset for next iteration
        FLAMEGPU.environment.setInt8("int8_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int8 == TEST_VALUE
        assert self.int8_ == TEST_VALUE * 2

class get_set_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int16 = FLAMEGPU.environment.setInt16("int16_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int16_ = FLAMEGPU.environment.getInt16("int16_")
        # Reset for next iteration
        FLAMEGPU.environment.setInt16("int16_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int16 == TEST_VALUE
        assert self.int16_ == TEST_VALUE * 2
        
class get_set_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.uint16 = FLAMEGPU.environment.setUInt16("uint16_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.uint16_ = FLAMEGPU.environment.getUInt16("uint16_")
        # Reset for next iteration
        FLAMEGPU.environment.setUInt16("uint16_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.uint16 == TEST_VALUE
        assert self.uint16_ == TEST_VALUE * 2
        
class get_set_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int32 = FLAMEGPU.environment.setInt32("int32_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int32_ = FLAMEGPU.environment.getInt32("int32_")
        # Reset for next iteration
        FLAMEGPU.environment.setInt32("int32_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int32 == TEST_VALUE
        assert self.int32_ == TEST_VALUE * 2
        
        
class get_set_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.uint32 = FLAMEGPU.environment.setUInt32("uint32_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.uint32_ = FLAMEGPU.environment.getUInt32("uint32_")
        # Reset for next iteration
        FLAMEGPU.environment.setUInt32("uint32_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.uint32 == TEST_VALUE
        assert self.uint32_ == TEST_VALUE * 2

class get_set_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.int64 = FLAMEGPU.environment.setInt64("int64_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.int64_ = FLAMEGPU.environment.getInt64("int64_")
        # Reset for next iteration
        FLAMEGPU.environment.setInt64("int64_", TEST_VALUE)
        
    def apply_assertions(self):
        assert self.int64 == TEST_VALUE
        assert self.int64_ == TEST_VALUE * 2      

class get_set_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Test Set + Get (Description set value)
        self.uint64 = FLAMEGPU.environment.setUInt64("uint64_", TEST_VALUE * 2)
        # Test Get (Host func set value)
        self.uint64_ = FLAMEGPU.environment.getUInt64("uint64_")
        # Reset for next iteration
        FLAMEGPU.environment.setUInt64("uint64_", TEST_VALUE)
        
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
        self.set_float_a = FLAMEGPU.environment.setArrayFloat("float_a_", init2)
        self.get_float_a = FLAMEGPU.environment.getArrayFloat("float_a_")
        FLAMEGPU.environment.setArrayFloat("float_a_", init1)
        
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
        self.set_double_a = FLAMEGPU.environment.setArrayDouble("double_a_", init2)
        self.get_double_a = FLAMEGPU.environment.getArrayDouble("double_a_")
        FLAMEGPU.environment.setArrayDouble("double_a_", init1)
        
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
        self.set_int8_a = FLAMEGPU.environment.setArrayInt8("int8_a_", init2)
        self.get_int8_a = FLAMEGPU.environment.getArrayInt8("int8_a_")
        FLAMEGPU.environment.setArrayInt8("int8_a_", init1)
        
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
        self.set_uint8_a = FLAMEGPU.environment.setArrayUInt8("uint8_a_", init2)
        self.get_uint8_a = FLAMEGPU.environment.getArrayUInt8("uint8_a_")
        FLAMEGPU.environment.setArrayUInt8("uint8_a_", init1)
        
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
        self.set_int16_a = FLAMEGPU.environment.setArrayInt16("int16_a_", init2)
        self.get_int16_a = FLAMEGPU.environment.getArrayInt16("int16_a_")
        FLAMEGPU.environment.setArrayInt16("int16_a_", init1)
        
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
        self.set_uint16_a = FLAMEGPU.environment.setArrayUInt16("uint16_a_", init2)
        self.get_uint16_a = FLAMEGPU.environment.getArrayUInt16("uint16_a_")
        FLAMEGPU.environment.setArrayUInt16("uint16_a_", init1)
        
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
        self.set_int32_a = FLAMEGPU.environment.setArrayInt32("int32_a_", init2)
        self.get_int32_a = FLAMEGPU.environment.getArrayInt32("int32_a_")
        FLAMEGPU.environment.setArrayInt32("int32_a_", init1)
        
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
        self.set_uint32_a = FLAMEGPU.environment.setArrayUInt32("uint32_a_", init2)
        self.get_uint32_a = FLAMEGPU.environment.getArrayUInt32("uint32_a_")
        FLAMEGPU.environment.setArrayUInt32("uint32_a_", init1)
        
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
        self.set_int64_a = FLAMEGPU.environment.setArrayInt64("int64_a_", init2)
        self.get_int64_a = FLAMEGPU.environment.getArrayInt64("int64_a_")
        FLAMEGPU.environment.setArrayInt64("int64_a_", init1)
        
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
        self.set_uint64_a = FLAMEGPU.environment.setArrayUInt64("uint64_a_", init2)
        self.get_uint64_a = FLAMEGPU.environment.getArrayUInt64("uint64_a_")
        FLAMEGPU.environment.setArrayUInt64("uint64_a_", init1)
        
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
        self.set_float_a = FLAMEGPU.environment.setFloat("float_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_float_a = FLAMEGPU.environment.getFloat("float_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setFloat("float_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_float_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_float_a == TEST_VALUE * 2    

class get_set_array_element_double(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_double_a = FLAMEGPU.environment.setDouble("double_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_double_a = FLAMEGPU.environment.getDouble("double_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setDouble("double_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_double_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_double_a == TEST_VALUE * 2

class get_set_array_element_int8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int8_a = FLAMEGPU.environment.setInt8("int8_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int8_a = FLAMEGPU.environment.getInt8("int8_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setInt8("int8_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int8_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int8_a == TEST_VALUE * 2    

class get_set_array_element_uint8(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint8_a = FLAMEGPU.environment.setUInt8("uint8_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint8_a = FLAMEGPU.environment.getUInt8("uint8_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setUInt8("uint8_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_uint8_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_uint8_a == TEST_VALUE * 2  

class get_set_array_element_int16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int16_a = FLAMEGPU.environment.setInt16("int16_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int16_a = FLAMEGPU.environment.getInt16("int16_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setInt16("int16_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int16_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int16_a == TEST_VALUE * 2    

class get_set_array_element_uint16(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint16_a = FLAMEGPU.environment.setUInt16("uint16_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint16_a = FLAMEGPU.environment.getUInt16("uint16_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setUInt16("uint16_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_uint16_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_uint16_a == TEST_VALUE * 2          

class get_set_array_element_int32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int32_a = FLAMEGPU.environment.setInt32("int32_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int32_a = FLAMEGPU.environment.getInt32("int32_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setInt32("int32_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int32_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int32_a == TEST_VALUE * 2    

class get_set_array_element_uint32(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint32_a = FLAMEGPU.environment.setUInt32("uint32_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint32_a = FLAMEGPU.environment.getUInt32("uint32_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setUInt32("uint32_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_uint32_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_uint32_a == TEST_VALUE * 2  

class get_set_array_element_int64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_int64_a = FLAMEGPU.environment.setInt64("int64_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_int64_a = FLAMEGPU.environment.getInt64("int64_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setInt64("int64_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
    def apply_assertions(self):
        init1 = MiniSim.make_array()
        assert self.set_int64_a == init1[TEST_ARRAY_LEN - 1]
        assert self.get_int64_a == TEST_VALUE * 2    

class get_set_array_element_uint64(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        init1 = MiniSim.make_array()
        self.set_uint64_a = FLAMEGPU.environment.setUInt64("uint64_a_", TEST_ARRAY_LEN - 1, TEST_VALUE * 2)
        self.get_uint64_a = FLAMEGPU.environment.getUInt64("uint64_a_", TEST_ARRAY_LEN - 1)
        FLAMEGPU.environment.setUInt64("uint64_a_", TEST_ARRAY_LEN - 1, init1[TEST_ARRAY_LEN - 1])
        
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
            FLAMEGPU.environment.setUInt64("float_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("float_", init1)
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
            FLAMEGPU.environment.setUInt64("double_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("double_", init1)
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
            FLAMEGPU.environment.setUInt64("int8_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("int8_", init1)
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
            FLAMEGPU.environment.setUInt64("uint8_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("uint8_", init1)
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
            FLAMEGPU.environment.setUInt64("int16_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("int16_", init1)
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
            FLAMEGPU.environment.setUInt64("uint16_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("uint16_", init1)
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
            FLAMEGPU.environment.setUInt64("int32_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("int32_", init1)
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
            FLAMEGPU.environment.setUInt64("uint32_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("uint32_", init1)
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
            FLAMEGPU.environment.setFloat("int64_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayFloat("int64_", init1)
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
            FLAMEGPU.environment.setFloat("uint64_", TEST_VALUE)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayFloat("uint64_", init1)
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
        FLAMEGPU.environment.setArrayFloat("float_a_", b)
        try:
            FLAMEGPU.environment.setArrayFloat("float_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayFloat("float_a_", b2)
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
        FLAMEGPU.environment.setArrayDouble("double_a_", b)
        try:
            FLAMEGPU.environment.setArrayDouble("double_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayDouble("double_a_", b2)
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
        FLAMEGPU.environment.setArrayInt8("int8_a_", b)
        try:
            FLAMEGPU.environment.setArrayInt8("int8_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayInt8("int8_a_", b2)
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
        FLAMEGPU.environment.setArrayUInt8("uint8_a_", b)
        try:
            FLAMEGPU.environment.setArrayUInt8("uint8_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt8("uint8_a_", b2)
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
        FLAMEGPU.environment.setArrayInt16("int16_a_", b)
        try:
            FLAMEGPU.environment.setArrayInt16("int16_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayInt16("int16_a_", b2)
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
        FLAMEGPU.environment.setArrayUInt16("uint16_a_", b)
        try:
            FLAMEGPU.environment.setArrayUInt16("uint16_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt16("uint16_a_", b2)
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
        FLAMEGPU.environment.setArrayInt32("int32_a_", b)
        try:
            FLAMEGPU.environment.setArrayInt32("int32_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayInt32("int32_a_", b2)
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
        FLAMEGPU.environment.setArrayUInt32("uint32_a_", b)
        try:
            FLAMEGPU.environment.setArrayUInt32("uint32_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt32("uint32_a_", b2)
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
        FLAMEGPU.environment.setArrayInt64("int64_a_", b)
        try:
            FLAMEGPU.environment.setArrayInt64("int64_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayInt64("int64_a_", b2)
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
        FLAMEGPU.environment.setArrayUInt64("uint64_a_", b)
        try:
            FLAMEGPU.environment.setArrayUInt64("uint64_a_", b1)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.setArrayUInt64("uint64_a_", b2)
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
                FLAMEGPU.environment.setFloat("float_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getFloat("float_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setDouble("double_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getDouble("double_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setInt8("int8_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getInt8("int8_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setUInt8("uint8_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getUInt8("uint8_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setInt16("int16_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getInt16("int16_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setUInt16("uint16_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getUInt16("uint16_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setInt32("int32_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getInt32("int32_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setUInt32("uint32_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getUInt32("uint32_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setInt64("int64_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getInt64("int64_a_", TEST_ARRAY_LEN + i)
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
                FLAMEGPU.environment.setUInt64("uint64_a_", TEST_ARRAY_LEN + i, TEST_VALUE)
            except pyflamegpu.FGPURuntimeException as e:
                self.e1 = e.type()
            try:
                FLAMEGPU.environment.getUInt64("uint64_a_", TEST_ARRAY_LEN + i)
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
            FLAMEGPU.environment.getFloat("a")
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        try:
            FLAMEGPU.environment.getFloat("a", 1)
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
            FLAMEGPU.environment.setFloat("read_only", a)
        except pyflamegpu.FGPURuntimeException as e:
            self.e1 = e.type()
        # no throw
        FLAMEGPU.environment.getFloat("read_only")
        # array version
        try:
            FLAMEGPU.environment.setArrayInt("read_only_a", b)
        except pyflamegpu.FGPURuntimeException as e:
            self.e2 = e.type()
        # no throw
        FLAMEGPU.environment.getInt("read_only_a")
        FLAMEGPU.environment.getInt("read_only_a", 1)     
        
    def apply_assertions(self):
        assert self.e1 == "ReadOnlyEnvProperty"
        assert self.e2 == "ReadOnlyEnvProperty"


class reserved_name_set_step(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.environment.setInt("_", 1)     
      
class reserved_name_set_array_step(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.environment.setIntArray2("_",  1, 2 )      
    


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
