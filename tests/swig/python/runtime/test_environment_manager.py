import pytest
import sys
from unittest import TestCase
from pyflamegpu import *


TEST_LEN = 256  # Agent count
MS1_VAL = 12.0
MS1_VAL2 = 36.0
MS2_VAL = 13.0
UINT64_MAX = sys.maxsize
INT64_MAX = int(sys.maxsize/2)

class MiniSim:

    def __init__(self, model_name:str):
        """
        Initialise a mini simulation
        """
        self.model = pyflamegpu.ModelDescription(model_name)
        self.agent = self.model.newAgent("agent")
        self.env = self.model.Environment()
        self.population = pyflamegpu.AgentVector(self.agent, TEST_LEN)
        #self.model.addStepFunction(DEFAULT_STEP) # Default step not required
    
    def run(self): 
        # CudaModel must be declared here
        # As the initial call to constructor fixes the agent population
        # This means if we haven't called model.newAgent(agent) first
        self.cudaSimulation = pyflamegpu.CUDASimulation(self.model)
        self.cudaSimulation.SimulationConfig().steps = 1
        # This fails as agentMap is empty
        self.cudaSimulation.setPopulationData(self.population)
        self.cudaSimulation.simulate()
        # The negative of this, is that cudaSimulation is inaccessible within the test!
        # So copy across population data here
        self.cudaSimulation.getPopulationData(self.population)


class AlignTest(pyflamegpu.HostFunction):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()
        # default values for assertion checks
        self.a = 0
        self.b = 0
        self.c = 0
        self.d = 0
        self.e = 0
        self.f = 0

    def run(self, FLAMEGPU):
        """
        Assertions are not possible within the run function as this is a callback in the c++ library. 
        Instead values can be saved to the class and asserted after the model step function has completed.
        """
        # self.a = FLAMEGPU.environment.getPropertyBool("a")
        self.b = FLAMEGPU.environment.getPropertyUInt64("b")
        self.c = FLAMEGPU.environment.getPropertyInt8("c")
        self.d = FLAMEGPU.environment.getPropertyInt64("d")
        self.e = FLAMEGPU.environment.getPropertyInt8("e")
        self.f = FLAMEGPU.environment.getPropertyFloat("f")
        
    def apply_assertions(self):
        # assert self.a = True # no bool
        assert self.b == UINT64_MAX
        assert self.c == 12
        assert self.d == INT64_MAX
        assert self.e == 21
        assert self.f == 13.0


class Multi_ms1(pyflamegpu.HostFunction):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()
        # default values for assertion checks
        self.ms1_float = 0
        self.ms1_float2 = 0

    def run(self, FLAMEGPU):
        """
        Assertions are not possible within the run function as this is a callback in the c++ library. 
        Instead values can be saved to the class and asserted after the model step function has completed.
        """
        self.ms1_float = FLAMEGPU.environment.getPropertyFloat("ms1_float");
        self.ms1_float2 = FLAMEGPU.environment.getPropertyFloat("ms1_float2");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            FLAMEGPU.environment.getPropertyDouble("ms2_double")
        assert e.value.type() == "InvalidEnvProperty"
        
    def apply_assertions(self):
        assert self.ms1_float == MS1_VAL
        assert self.ms1_float2 == MS1_VAL2
        
class Multi_ms2(pyflamegpu.HostFunction):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the run function
    """
    def __init__(self):
        super().__init__()
        # default values for assertion checks
        self.ms2_double = 0
        self.ms2_float = 0

    def run(self, FLAMEGPU):
        """
        Assertions are not possible within the run function as this is a callback in the c++ library. 
        Instead values can be saved to the class and asserted after the model step function has completed.
        """
        self.ms1_double = FLAMEGPU.environment.getPropertyDouble("ms2_double");
        self.ms1_float = FLAMEGPU.environment.getPropertyFloat("ms2_float");
        
    def apply_assertions(self):
        assert self.ms2_double == MS2_VAL
        assert self.ms1_float == MS2_VAL


class EnvironmentManagerTest(TestCase):
    
    # Test alignment
    def test_alignment(self):
        ms = MiniSim("test_alignment")
        # add environment varibles (no bool)
        ms.env.newPropertyUInt64("b", UINT64_MAX)
        ms.env.newPropertyInt8("c", 12)
        ms.env.newPropertyInt64("d", INT64_MAX)
        ms.env.newPropertyInt8("e", 21)
        ms.env.newPropertyFloat("f", 13.0)
        # simulate
        at = AlignTest()
        ms.model.addStepFunction(at)
        ms.run()
        # apply assertions
        at.apply_assertions()
        
    # Test bounds limit
    def test_out_of_memory1(self):
        # Cant currently run out of memory with limited fixed sized types
        pass

    def test_out_of_memory2(self):
        # Cant currently run out of memory with limited fixed sized types
        pass


    # Multiple models
    def test_multiple_models(self): 
        ms1 = MiniSim("ms1")
        ms2 = MiniSim("ms2")
        ms1.env.newPropertyFloat("ms1_float", MS1_VAL)
        ms1.env.newPropertyFloat("ms1_float2", MS1_VAL2)
        ms2.env.newPropertyDouble("ms2_double", MS2_VAL)
        ms2.env.newPropertyFloat("ms2_float", MS2_VAL)
        multi_ms1 = Multi_ms1()
        multi_ms2 = Multi_ms2()
        ms1.model.addStepFunction(multi_ms1)
        ms2.model.addStepFunction(multi_ms2)
        ms1.run()
        ms2.run()


