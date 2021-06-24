import pytest
from unittest import TestCase
from pyflamegpu import *

class ExitAlways(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
      return pyflamegpu.EXIT;

class SubEnvironmentDescriptionTest(TestCase):
    
    def test_InvalidNames(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayFloat("a2", 2, [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0);
        m.Environment().newPropertyArrayFloat("b2", 2, [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("c", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("c", "b2");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "c");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "c");
        senv.mapProperty("a2", "b2");
        senv.mapProperty("a", "b");

    def test_TypesDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayInt("a2", 2, [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyUInt("b", 0);
        m.Environment().newPropertyArrayFloat("b2", 2, [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b2");

    def test_ElementsDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayFloat("a2", 2, [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0);
        m.Environment().newPropertyArrayFloat("b2", 2, [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b2");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b");
        senv.mapProperty("a2", "b2");
        senv.mapProperty("a", "b");

    def test_IsConstWrong(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayFloat("a2", 2, [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0, True);
        m.Environment().newPropertyArrayFloat("b2", 2, [0] * 2, True);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b2");

    def test_AlreadyBound(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayFloat("a2", 2, [0.0] * 2);
        m2.Environment().newPropertyFloat("a_", 0);
        m2.Environment().newPropertyArrayFloat("a2_", 2, [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0);
        m.Environment().newPropertyArrayFloat("b2", 2, [0] * 2);
        m.Environment().newPropertyFloat("b_", 0);
        m.Environment().newPropertyArrayFloat("b2_", 2, [0] * 2);
        # Missing exit condition
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.mapProperty("a", "b");
        senv.mapProperty("a2", "b2");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b_");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b2_");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a_", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2_", "b2");

