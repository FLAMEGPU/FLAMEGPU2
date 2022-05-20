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
        m2.Environment().newPropertyArrayFloat("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0);
        m.Environment().newPropertyArrayFloat("b2", [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("c", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("c", "b2");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "c");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "c");
        senv.mapProperty("a2", "b2");
        senv.mapProperty("a", "b");

    def test_TypesDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayInt("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyUInt("b", 0);
        m.Environment().newPropertyArrayFloat("b2", [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b2");

    def test_ElementsDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayFloat("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0);
        m.Environment().newPropertyArrayFloat("b2", [0] * 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b2");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b");
        senv.mapProperty("a2", "b2");
        senv.mapProperty("a", "b");

    def test_IsConstWrong(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayFloat("a2", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0, True);
        m.Environment().newPropertyArrayFloat("b2", [0] * 2, True);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b2");

    def test_AlreadyBound(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newPropertyFloat("a", 0);
        m2.Environment().newPropertyArrayFloat("a2", [0.0] * 2);
        m2.Environment().newPropertyFloat("a_", 0);
        m2.Environment().newPropertyArrayFloat("a2_", [0] * 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newPropertyFloat("b", 0);
        m.Environment().newPropertyArrayFloat("b2", [0] * 2);
        m.Environment().newPropertyFloat("b_", 0);
        m.Environment().newPropertyArrayFloat("b2_", [0] * 2);
        # Missing exit condition
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.mapProperty("a", "b");
        senv.mapProperty("a2", "b2");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a", "b_");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2", "b2_");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a_", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapProperty("a2_", "b2");
            
    def test_Macro_InvalidNames(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newMacroPropertyFloat("a");
        m2.Environment().newMacroPropertyFloat("a2", 2, 3, 4, 5);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyFloat("b");
        m.Environment().newMacroPropertyFloat("b2", 2, 3, 4, 5);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapMacroProperty("c", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapMacroProperty("c", "b2");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapMacroProperty("a", "c");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapMacroProperty("a2", "c");
        senv.mapMacroProperty("a2", "b2");
        senv.mapMacroProperty("a", "b");

    def test_Macro_TypesDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newMacroPropertyFloat("a");
        m2.Environment().newMacroPropertyInt("a2");
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyUInt("b");
        m.Environment().newMacroPropertyFloat("b2", 2, 3, 4);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapMacroProperty("a", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            senv.mapMacroProperty("a2", "b2");

    def test_Macro_DimensionsDoNotMatch(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newMacroPropertyFloat("a", 4, 3, 2, 1);
        m2.Environment().newMacroPropertyFloat("a2", 1, 2, 3, 4);
        m2.Environment().newMacroPropertyFloat("a3", 1, 2, 3);
        m2.Environment().newMacroPropertyFloat("a4", 2, 3, 4);
        m2.Environment().newMacroPropertyFloat("a5");
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyFloat("b", 4, 3, 2, 1);
        m.Environment().newMacroPropertyFloat("b2", 1, 2, 3, 4);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a", "b2");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a", "b3");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a", "b4");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a", "b5");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a2", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a3", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a4", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a5", "b");
        senv.mapMacroProperty("a2", "b2");
        senv.mapMacroProperty("a", "b");

    def test_Macro_AlreadyBound(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exitcdn = ExitAlways()
        m2.addExitConditionCallback(exitcdn);
        m2.Environment().newMacroPropertyFloat("a");
        m2.Environment().newMacroPropertyFloat("a2", 2);
        m2.Environment().newMacroPropertyFloat("a_");
        m2.Environment().newMacroPropertyFloat("a2_", 2);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyFloat("b");
        m.Environment().newMacroPropertyFloat("b2", 2);
        m.Environment().newMacroPropertyFloat("b_");
        m.Environment().newMacroPropertyFloat("b2_", 2);
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.mapMacroProperty("a", "b");
        senv.mapMacroProperty("a2", "b2");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a", "b_");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a2", "b2_");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a_", "b");
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
          senv.mapMacroProperty("a2_", "b2");
