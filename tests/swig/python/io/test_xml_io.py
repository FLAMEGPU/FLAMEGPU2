import pytest
from unittest import TestCase
from pyflamegpu import *
import numpy as np
import os

validate_has_run = False
XML_FILE_NAME = "test.xml"
AGENT_COUNT = 100

class ValidateEnv(pyflamegpu.HostFunctionCallback):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        assert FLAMEGPU.environment.getFloat("float") == 12.0
        assert FLAMEGPU.environment.getFloat("float") == 12.0
        assert FLAMEGPU.environment.getDouble("double") == 13.0
        assert FLAMEGPU.environment.getInt64("int64_t") == 14
        assert FLAMEGPU.environment.getUInt64("uint64_t") == 15
        assert FLAMEGPU.environment.getInt32("int32_t") == 16
        assert FLAMEGPU.environment.getUInt32("uint32_t") == 17
        assert FLAMEGPU.environment.getInt16("int16_t") == 18
        assert FLAMEGPU.environment.getUInt16("uint16_t") == 19
        assert FLAMEGPU.environment.getInt8("int8_t") == 20
        assert FLAMEGPU.environment.getUInt8("uint8_t") == 21
        assert FLAMEGPU.environment.getFloat3("float_a") == [ 12.0, 0.0, 1.0 ]
        assert FLAMEGPU.environment.getDouble3("double_a") == [ 13.0, 0.0, 1.0 ]
        assert FLAMEGPU.environment.getInt64A3("int64_t_a") == [ 14, 0, 1 ]
        assert FLAMEGPU.environment.getUInt64A3("uint64_t_a") == [ 15, 0, 1 ]
        assert FLAMEGPU.environment.getInt32A3("int32_t_a") == [ 16, 0, 1 ]
        assert FLAMEGPU.environment.getUInt323("uint32_t_a") == [ 17, 0, 1 ]
        assert FLAMEGPU.environment.getInt16A3("int16_t_a") == [ 18, 0, 1 ]
        assert FLAMEGPU.environment.getUInt16A3("uint16_t_a") == [ 19, 0, 1 ]
        assert FLAMEGPU.environment.getInt8A3("int8_t_a") == [ 20, 0, 1 ]
        assert FLAMEGPU.environment.getUInt8A3("uint8_t_a") == [ 21, 0, 1 ]
        validate_has_run = True

class ResetEnv(pyflamegpu.HostFunctionCallback):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.environment.setFloat("float", [])
        FLAMEGPU.environment.setDouble("double", [])
        FLAMEGPU.environment.setInt64("int64_t", [])
        FLAMEGPU.environment.setUInt64("uint64_t", [])
        FLAMEGPU.environment.setInt32("int32_t", [])
        FLAMEGPU.environment.setUInt32("uint32_t", [])
        FLAMEGPU.environment.setInt16("int16_t", [])
        FLAMEGPU.environment.setUInt16("uint16_t", [])
        FLAMEGPU.environment.setInt8("int8_t", [])
        FLAMEGPU.environment.setUInt8("uint8_t", [])
        FLAMEGPU.environment.setFloatA3("float_a", [])
        FLAMEGPU.environment.setDoubleA3("double_a", [])
        FLAMEGPU.environment.setInt64A3("int64_t_a", [])
        FLAMEGPU.environment.setUInt64A3("uint64_t_a", [])
        FLAMEGPU.environment.setInt32A3("int32_t_a", [])
        FLAMEGPU.environment.setUInt32A3("uint32_t_a", [])
        FLAMEGPU.environment.setInt16A3("int16_t_a", [])
        FLAMEGPU.environment.setUInt16A3("uint16_t_a", [])
        FLAMEGPU.environment.setInt8A3("int8_t_a", [])
        FLAMEGPU.environment.setUInt83("uint8_t_a", [])  

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
        e.addFloatArray3("float_a", [ 12.0, 0.0, 1.0 ]) # broken here
        e.addDoubleArray3("double_a", [ 13.0, 0.0, 1.0 ])
        e.addInt64Array3("int64_t_a", [ 14, 0, 1 ])
        e.addUInt64Array3("uint64_t_a", [ 15, 0, 1 ])
        e.addInt32Array3("int32_t_a", [ 16, 0, 1 ])
        e.addUInt32Array3("uint32_t_a", [ 17, 0, 1 ])
        e.addInt16Array3("int16_t_a", [ 18, 0, 1])
        e.addUInt16Array3("uint16_t_a", [ 19, 0, 1 ])
        e.addInt8Array3("int8_t_a", [ 20, 0, 1 ])
        e.addUInt8Array3("uint8_t_a", [21, 0, 1])

        pop_a_out = pyflamegpu.AgentPopulation (a, 5)
        for i in range(5):
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


        pop_b_out = pyflamegpu.AgentPopulation(b, 5)
        for i in range(5):
            agent = pop_b_out.getNextInstance("2")  # Set Variables not in the initial state
            agent.setVariableFloatArray3("float", [ 1.0, float(i), 1.0 ])
            agent.setVariableDoubleArray3("double", [ 2.0, float(i), 1.0 ])
            agent.setVariableInt64Array3("int64_t", [ 3, i, 1 ])
            agent.setVariableUInt64Array3("uint64_t", [ 4, i, 1 ])
            agent.setVariableInt32Array3("int32_t", [ 5, i, 1 ])
            agent.setVariableUInt32Array3("uint32_t", [ 6, i, 1 ])
            agent.setVariableInt16Array3("int16_t", [ 7, i, 1 ])
            agent.setVariableUInt16Array3("uint16_t", [ 8, i, 1 ])
            agent.setVariableInt8Array3("int8_t", [ 9, i, 1 ])
            agent.setVariableUInt8Array3("uint8_t", [ 10, i, 1 ])
      
        validate = ValidateEnv()
        reset = ResetEnv()
        m.newLayer().addHostFunctionCallback(validate)
        m.newLayer().addHostFunctionCallback(reset)
        
        # Run Export
        am_export = pyflamegpu.CUDAAgentModel(m)
        am_export.setPopulationData(pop_a_out)
        am_export.setPopulationData(pop_b_out)
        am_export.exportData(XML_FILE_NAME)

        # Run Import
        am = pyflamegpu.CUDAAgentModel(m)
        am.SimulationConfig().xml_input_file = XML_FILE_NAME
        am.applyConfig()
        pop_a_in = pyflamegpu.AgentPopulation(a, 5)
        pop_b_in = pyflamegpu.AgentPopulation(b, 5)
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
            assert agent_in.getVariableFloatArray3("float") == agent_out.getVariableFloatA3("float")
            assert agent_in.getVariableDoubleArray3("double") == agent_out.getVariableDoubleA3("double")
            assert agent_in.getVariableInt64Array3("int64_t") == agent_out.getVariableInt64A3("int64_t")
            assert agent_in.getVariableUInt64Array3("uint64_t") == agent_out.getVariableUInt64A3("uint64_t")
            assert agent_in.getVariableInt32Array3("int32_t") == agent_out.getVariableInt32A3("int32_t")
            assert agent_in.getVariableUInt32Array3("uint32_t") == agent_out.getVariableUInt32A3("uint32_t")
            assert agent_in.getVariableInt16Array3("int16_t") == agent_out.getVariableInt16A3("int16_t")
            assert agent_in.getVariableUInt16Array3("uint16_t") == agent_out.getVariableUInt16A3("uint16_t")
            assert agent_in.getVariableInt8Array3("int8_t") == agent_out.getVariableInt8A3("int8_t")
            assert agent_in.getVariableUInt8Array3("uint8_t") == agent_out.getVariableUInt8A3("uint8_t")

        # Loade m
        am = pyflamegpu.CUDAAgentModel(m)
        # Step once, this checks and clears env vars
        validate_has_run = False
        am.step()
        assert validate_has_run == True
        # Reload env vars from file
        am.SimulationConfig().xml_input_file = XML_FILE_NAME
        am.applyConfig()
        # Step again, check they have been loaded
        validate_has_run = False
        am.step()
        assert validate_has_run == True

        # Cleanup
        os.remove(XML_FILE_NAME)
