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
        m2.Environment().addFloat("a", 0);
        m2.Environment().addFloatArray2("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().addFloat("b", 0);
        m.Environment().addFloatArray2("b2", [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("c", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("c", "b2");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a", "c");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a2", "c");
        senv.mapProperty("a2", "b2");
        senv.mapProperty("a", "b");

    def test_TypesDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().addFloat("a", 0);
        m2.Environment().addIntArray2("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().addUInt("b", 0);
        m.Environment().addFloatArray2("b2", [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a2", "b2");

    def test_ElementsDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().addFloat("a", 0);
        m2.Environment().addFloatArray2("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().addFloat("b", 0);
        m.Environment().addFloatArray2("b2", [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a", "b2");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a2", "b");
        senv.mapProperty("a2", "b2");
        senv.mapProperty("a", "b");

    def test_IsConstWrong(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().addFloat("a", 0);
        m2.Environment().addFloatArray2("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().addFloat("b", 0, True);
        m.Environment().addFloatArray2("b2", [0] * 2, True);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a2", "b2");

    def test_AlreadyBound(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().addFloat("a", 0);
        m2.Environment().addFloatArray2("a2", [0.0] * 2);
        m2.Environment().addFloat("a_", 0);
        m2.Environment().addFloatArray2("a2_", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().addFloat("b", 0);
        m.Environment().addFloatArray2("b2", [0] * 2);
        m.Environment().addFloat("b_", 0);
        m.Environment().addFloatArray2("b2_", [0] * 2);
        # Missing exit condition
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.mapProperty("a", "b");
        senv.mapProperty("a2", "b2");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a", "b_");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a2", "b2_");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a_", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidEnvProperty exception
            senv.mapProperty("a2_", "b2");

