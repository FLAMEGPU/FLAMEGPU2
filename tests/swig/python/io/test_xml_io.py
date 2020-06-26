import pytest
from unittest import TestCase
from pyflamegpu import *
import os

XML_FILE_NAME = "test.xml"
AGENT_COUNT = 100

class ValidateEnv(pyflamegpu.HostFunctionCallback):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()
        # default values for assertion checks
        self.float =0
        self.double = 0
        self.int64_t = 0
        self.uint64_t = 0
        self.int32_t = 0
        self.uint32_t = 0
        self.int16_t = 0
        self.uint16_t = 0
        self.int8_t = 0
        self.uint8_t = 0
        self.float_a = ( )
        self.double_a = ( )
        self.int64_t_a = ( )
        self.uint64_t_a = ( )
        self.int32_t_a = ( )
        self.uint32_t_a = ( )
        self.int16_t_a = ( )
        self.uint16_t_a = ( )
        self.int8_t_a = ( )
        self.uint8_t_a = ( )
        self.validate_has_run = False

    def run(self, FLAMEGPU):
        """
        Assertions are not possible within the run function as this is a callback in the c++ library. 
        Instead values can be saved to the class and asserted after the model step function has completed.
        """
        self.float = FLAMEGPU.environment.getFloat("float")
        self.double = FLAMEGPU.environment.getDouble("double")
        self.int64_t = FLAMEGPU.environment.getInt64("int64_t")
        self.uint64_t = FLAMEGPU.environment.getUInt64("uint64_t")
        self.int32_t = FLAMEGPU.environment.getInt32("int32_t")
        self.uint32_t = FLAMEGPU.environment.getUInt32("uint32_t")
        self.int16_t = FLAMEGPU.environment.getInt16("int16_t")
        self.uint16_t = FLAMEGPU.environment.getUInt16("uint16_t")
        self.int8_t = FLAMEGPU.environment.getInt8("int8_t")
        self.uint8_t = FLAMEGPU.environment.getUInt8("uint8_t")
        self.float_a = FLAMEGPU.environment.getFloatArray3("float_a")
        self.double_a = FLAMEGPU.environment.getDoubleArray3("double_a")
        self.int64_t_a = FLAMEGPU.environment.getInt64Array3("int64_t_a")
        self.uint64_t_a = FLAMEGPU.environment.getUInt64Array3("uint64_t_a")
        self.int32_t_a = FLAMEGPU.environment.getInt32Array3("int32_t_a")
        self.uint32_t_a = FLAMEGPU.environment.getUInt32Array3("uint32_t_a")
        self.int16_t_a = FLAMEGPU.environment.getInt16Array3("int16_t_a")
        self.uint16_t_a = FLAMEGPU.environment.getUInt16Array3("uint16_t_a")
        self.int8_t_a = FLAMEGPU.environment.getInt8Array3("int8_t_a")
        self.uint8_t_a = FLAMEGPU.environment.getUInt8Array3("uint8_t_a")
        self.validate_has_run = True
        
    def apply_assertions(self):
        assert self.float == 12.0
        assert self.double == 13.0
        assert self.int64_t == 14
        assert self.uint64_t == 15
        assert self.int32_t == 16
        assert self.uint32_t == 17
        assert self.int16_t == 18
        assert self.uint16_t == 19
        assert self.int8_t == 20
        assert self.uint8_t == 21
        assert self.float_a == ( 12.0, 0.0, 1.0 )
        assert self.double_a == ( 13.0, 0.0, 1.0 )
        assert self.int64_t_a == ( 14, 0, 1 )
        assert self.uint64_t_a == ( 15, 0, 1 )
        assert self.int32_t_a == ( 16, 0, 1 )
        assert self.uint32_t_a == ( 17, 0, 1 )
        assert self.int16_t_a == ( 18, 0, 1 )
        assert self.uint16_t_a == ( 19, 0, 1 )
        assert self.int8_t_a == ( 20, 0, 1 )
        assert self.uint8_t_a == ( 21, 0, 1 )
        assert self.validate_has_run == True
        

class ResetEnv(pyflamegpu.HostFunctionCallback):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.environment.setFloat("float", 0)
        FLAMEGPU.environment.setDouble("double", 0)
        FLAMEGPU.environment.setInt64("int64_t", 0)
        FLAMEGPU.environment.setUInt64("uint64_t", 0)
        FLAMEGPU.environment.setInt32("int32_t", 0)
        FLAMEGPU.environment.setUInt32("uint32_t", 0)
        FLAMEGPU.environment.setInt16("int16_t", 0)
        FLAMEGPU.environment.setUInt16("uint16_t", 0)
        FLAMEGPU.environment.setInt8("int8_t", 0)
        FLAMEGPU.environment.setUInt8("uint8_t", 0)
        FLAMEGPU.environment.setFloatArray3("float_a", (0,0,0))
        FLAMEGPU.environment.setDoubleArray3("double_a", (0,0,0))
        FLAMEGPU.environment.setInt64Array3("int64_t_a", (0,0,0))
        FLAMEGPU.environment.setUInt64Array3("uint64_t_a", (0,0,0))
        FLAMEGPU.environment.setInt32Array3("int32_t_a", (0,0,0))
        FLAMEGPU.environment.setUInt32Array3("uint32_t_a", (0,0,0))
        FLAMEGPU.environment.setInt16Array3("int16_t_a", (0,0,0))
        FLAMEGPU.environment.setUInt16Array3("uint16_t_a", (0,0,0))
        FLAMEGPU.environment.setInt8Array3("int8_t_a", (0,0,0))
        FLAMEGPU.environment.setUInt8Array3("uint8_t_a", (0,0,0)) 

        

class IOTest(TestCase):



    def test_read_write(self):
        m = pyflamegpu.ModelDescription("test_read_write")
        a = m.newAgent("a")
        a.newVariableFloat("float")
        a.newVariableDouble("double")
        a.newVariableInt64("int64_t")
        a.newVariableUInt64("uint64_t")
        a.newVariableInt32("int32_t")
        a.newVariableUInt32("uint32_t")
        a.newVariableInt16("int16_t")
        a.newVariableUInt16("uint16_t")
        a.newVariableInt8("int8_t")
        a.newVariableUInt8("uint8_t")
        
        b = m.newAgent("b")
        b.newState("1")
        b.newState("2")
        b.newVariableFloatArray3("float")
        b.newVariableDoubleArray3("double")
        b.newVariableInt64Array3("int64_t")
        b.newVariableUInt64Array3("uint64_t")
        b.newVariableInt32Array3("int32_t")
        b.newVariableUInt32Array3("uint32_t")
        b.newVariableInt16Array3("int16_t")
        b.newVariableUInt16Array3("uint16_t")
        b.newVariableInt8Array3("int8_t")
        b.newVariableUInt8Array3("uint8_t")
    
        e = m.Environment()
        e.addFloat("float", 12.0)
        e.addDouble("double", 13.0)
        e.addInt64("int64_t", 14)
        e.addUInt64("uint64_t", 15)
        e.addInt32("int32_t", 16)
        e.addUInt32("uint32_t", 17)
        e.addInt16("int16_t", 18)
        e.addUInt16("uint16_t", 19)
        e.addInt8("int8_t", 20)
        e.addUInt8("uint8_t", 21)
        e.addFloatArray3("float_a", ( 12.0, 0.0, 1.0 ))
        e.addDoubleArray3("double_a", ( 13.0, 0.0, 1.0 ))
        e.addInt64Array3("int64_t_a", ( 14, 0, 1 ))
        e.addUInt64Array3("uint64_t_a", ( 15, 0, 1 ))
        e.addInt32Array3("int32_t_a", ( 16, 0, 1 ))
        e.addUInt32Array3("uint32_t_a", ( 17, 0, 1 ))
        e.addInt16Array3("int16_t_a", ( 18, 0, 1))
        e.addUInt16Array3("uint16_t_a", ( 19, 0, 1 ))
        e.addInt8Array3("int8_t_a", ( 20, 0, 1 ))
        e.addUInt8Array3("uint8_t_a", (21, 0, 1))

        pop_a_out = pyflamegpu.AgentPopulation (a, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            agent = pop_a_out.getNextInstance()
            agent.setVariableFloat("float", float(1.0 + i))
            agent.setVariableDouble("double", float(2.0 + i))
            agent.setVariableInt64("int64_t", 3 + i)
            agent.setVariableUInt64("uint64_t", 4 + i)
            agent.setVariableInt32("int32_t", 5 + i)
            agent.setVariableUInt32("uint32_t", 6 + i)
            agent.setVariableInt16("int16_t", (7 + i))
            agent.setVariableUInt16("uint16_t", (8 + i))
            agent.setVariableInt8("int8_t", (9 + i))
            agent.setVariableUInt8("uint8_t", (10 + i))


        pop_b_out = pyflamegpu.AgentPopulation(b, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            agent = pop_b_out.getNextInstance("2")  # Set Variables not in the initial state
            agent.setVariableFloatArray3("float", ( 1.0, float(i), 1.0 ))
            agent.setVariableDoubleArray3("double", ( 2.0, float(i), 1.0 ))
            agent.setVariableInt64Array3("int64_t", ( 3, i, 1 ))
            agent.setVariableUInt64Array3("uint64_t", ( 4, i, 1 ))
            agent.setVariableInt32Array3("int32_t", ( 5, i, 1 ))
            agent.setVariableUInt32Array3("uint32_t", ( 6, i, 1 ))
            agent.setVariableInt16Array3("int16_t", ( 7, i, 1 ))
            agent.setVariableUInt16Array3("uint16_t", ( 8, i, 1 ))
            agent.setVariableInt8Array3("int8_t", ( 9, i, 1 ))
            agent.setVariableUInt8Array3("uint8_t", ( 10, i, 1 ))
      
        # Add the validate and rest step functions in specific order.
        validate = ValidateEnv()
        reset = ResetEnv()
        m.newLayer().addHostFunctionCallback(validate)
        m.newLayer().addHostFunctionCallback(reset)
        
        # Run Export
        am_export = pyflamegpu.CUDAAgentModel(m)
        am_export.setPopulationData(pop_a_out)
        am_export.setPopulationData(pop_b_out)
        am_export.exportData(XML_FILE_NAME)
        del am_export # Delete previous CUDAAgentModel as multiple models with same name cant exist

        # Run Import
        am = pyflamegpu.CUDAAgentModel(m)
        am.SimulationConfig().xml_input_file = XML_FILE_NAME
        am.applyConfig()
        pop_a_in = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        pop_b_in = pyflamegpu.AgentPopulation(b, AGENT_COUNT)
        am.getPopulationData(pop_a_in)
        am.getPopulationData(pop_b_in)
        
        # Valid agent none array vars
        assert pop_a_in.getCurrentListSize() == pop_a_out.getCurrentListSize()
        for i in range(pop_a_in.getCurrentListSize()):
            agent_in = pop_a_in.getInstanceAt(i)
            agent_out = pop_a_out.getInstanceAt(i)
            assert agent_in.getVariableFloat("float") == agent_out.getVariableFloat("float")
            assert agent_in.getVariableDouble("double") == agent_out.getVariableDouble("double")
            assert agent_in.getVariableInt64("int64_t") == agent_out.getVariableInt64("int64_t")
            assert agent_in.getVariableUInt64("uint64_t") == agent_out.getVariableUInt64("uint64_t")
            assert agent_in.getVariableInt32("int32_t") == agent_out.getVariableInt32("int32_t")
            assert agent_in.getVariableUInt32("uint32_t") == agent_out.getVariableUInt32("uint32_t")
            assert agent_in.getVariableInt16("int16_t") == agent_out.getVariableInt16("int16_t")
            assert agent_in.getVariableUInt16("uint16_t") == agent_out.getVariableUInt16("uint16_t")
            assert agent_in.getVariableInt8("int8_t") == agent_out.getVariableInt8("int8_t")
            assert agent_in.getVariableUInt8("uint8_t") == agent_out.getVariableUInt8("uint8_t")

        # Valid agent array vars
        assert pop_b_in.getCurrentListSize("2") == pop_b_out.getCurrentListSize("2")
        for i in range(pop_b_in.getCurrentListSize("2")):
            agent_in = pop_b_in.getInstanceAt(i, "2")
            agent_out = pop_b_out.getInstanceAt(i, "2")
            assert agent_in.getVariableFloatArray3("float") == agent_out.getVariableFloatArray3("float")
            assert agent_in.getVariableDoubleArray3("double") == agent_out.getVariableDoubleArray3("double")
            assert agent_in.getVariableInt64Array3("int64_t") == agent_out.getVariableInt64Array3("int64_t")
            assert agent_in.getVariableUInt64Array3("uint64_t") == agent_out.getVariableUInt64Array3("uint64_t")
            assert agent_in.getVariableInt32Array3("int32_t") == agent_out.getVariableInt32Array3("int32_t")
            assert agent_in.getVariableUInt32Array3("uint32_t") == agent_out.getVariableUInt32Array3("uint32_t")
            assert agent_in.getVariableInt16Array3("int16_t") == agent_out.getVariableInt16Array3("int16_t")
            assert agent_in.getVariableUInt16Array3("uint16_t") == agent_out.getVariableUInt16Array3("uint16_t")
            assert agent_in.getVariableInt8Array3("int8_t") == agent_out.getVariableInt8Array3("int8_t")
            assert agent_in.getVariableUInt8Array3("uint8_t") == agent_out.getVariableUInt8Array3("uint8_t")

        am = pyflamegpu.CUDAAgentModel(m)

        # Step once, this checks and clears env vars
        am.step()
        # check step function assertions (with values loaded from runtime env)
        validate.apply_assertions()
        
        # Reload env vars from file
        am.SimulationConfig().xml_input_file = XML_FILE_NAME
        am.applyConfig()
        # Step again, check they have been loaded
        am.step()
        # check step function assertions (with values loaded from runtime env)
        validate.apply_assertions()

        # Cleanup
        os.remove(XML_FILE_NAME)
