import pytest
from unittest import TestCase
from pyflamegpu import *
import os

MODEL_NAME = "Model";
AGENT_NAME1 = "Agent1";
AGENT_NAME2 = "Agent2";

class LoggingExceptionTest(TestCase):

    def test_LoggerSupportedFileType(self):
        # LoggerFactory::createLogger() - exception::UnsupportedFileType
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        m.newAgent(AGENT_NAME1);
        sim = pyflamegpu.CUDASimulation(m);
        # Unsupported file types
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::UnsupportedFileType exception
            sim.exportLog("test.csv", True, True, True, True);
        assert e.value.type() == "UnsupportedFileType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::UnsupportedFileType exception
            sim.exportLog("test.html", True, True, True, True);
        assert e.value.type() == "UnsupportedFileType"
        # Does not throw
        sim.exportLog("test.json", True, True, True, True);
        # Cleanup
        os.remove("test.json")
        
    def test_LoggingConfigExceptions(self):
        # Define model
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        m.newAgent(AGENT_NAME1);
        m.Environment().newPropertyFloat("float_prop", 1.0);

        lcfg = pyflamegpu.LoggingConfig(m);
        # Property doesn't exist
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            lcfg.logEnvironment("int_prop")
        assert e.value.type() == "InvalidEnvProperty"
        # Property does exist
        lcfg.logEnvironment("float_prop")
        # Property already marked for logging
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            lcfg.logEnvironment("float_prop")
        assert e.value.type() == "InvalidEnvProperty"
        # THIS DOES NOT WORK, cfg holds a copy of the ModelDescription, not a reference to it.
        # Add new property after lcfg made
        m.Environment().newPropertyInt("int_prop", 1)
        # Property does not exist
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            lcfg.logEnvironment("int_prop")
        assert e.value.type() == "InvalidEnvProperty"

        # Agent does not exist
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentName exception
            lcfg.agent(AGENT_NAME2, "state2")
        assert e.value.type() == "InvalidAgentName"
        # Agent state does not exist
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentState exception
            lcfg.agent(AGENT_NAME1, "state2")
        assert e.value.type() == "InvalidAgentState"
        # Agent/State does exist
        lcfg.agent(AGENT_NAME1, "default"); # There isn't currently a py mapping of ModelData::DEFAULT_STATE
        # THIS DOES NOT WORK, cfg holds a copy of the ModelDescription, not a reference to it.
        # Add new agent after lcfg
        a2 = m.newAgent(AGENT_NAME2);
        a2.newState("state2");
        # Agent/State does not exist
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentName exception
            lcfg.agent(AGENT_NAME2, "state2")
        assert e.value.type() == "InvalidAgentName"
        
    def test_AgentLoggingConfigExceptions(self):
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        a = m.newAgent(AGENT_NAME1);
        a.newVariableFloat("float_var");
        a.newVariableArrayFloat("float_var_array", 2);

        lcfg = pyflamegpu.LoggingConfig(m);
        alcfg = lcfg.agent(AGENT_NAME1);

        # Log functions all pass to the same common method which contains the checks
        # Test 1, test them all (mean, standard dev, min, max, sum)
        # Bad variable name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidVarType exception
            alcfg.logMeanInt("int_var")
        assert e.value.type() == "InvalidAgentVar"
        # Type does not match variable name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentName exception
            alcfg.logMeanInt("float_var")
        assert e.value.type() == "InvalidVarType"
        # Array variables are not supported
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidVarType exception
            alcfg.logMeanFloat("float_var_array")
        assert e.value.type() == "InvalidVarType"
        # Variable is correct
        alcfg.logMeanFloat("float_var")
        # Variable has already been marked for logging
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidArgument exception
            alcfg.logMeanFloat("float_var")
        assert e.value.type() == "InvalidArgument"
        # THIS DOES NOT WORK, cfg holds a copy of the ModelDescription, not a reference to it.
        # Add new agent var after log creation
        a.newVariableInt("int_var");
        # Variable is not found
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentVar exception
            alcfg.logMeanInt("int_var")
        assert e.value.type() == "InvalidAgentVar"
        
    def test_LogFrameExceptions(self):
        # Define model
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        m.Environment().newPropertyFloat("float_prop", 1.0);
        m.Environment().newPropertyInt("int_prop", 1);
        m.Environment().newPropertyUInt("uint_prop", 1);
        m.Environment().newPropertyArrayFloat("float_prop_array", 2, [1.0, 2.0]);
        m.Environment().newPropertyArrayInt("int_prop_array", 3, [2, 3, 4]);
        m.Environment().newPropertyArrayUInt("uint_prop_array", 4, [3, 4, 5, 6]);

        # Define logging configs
        lcfg = pyflamegpu.LoggingConfig(m);

        slcfg = pyflamegpu.StepLoggingConfig(lcfg);
        slcfg.logEnvironment("float_prop");
        slcfg.logEnvironment("uint_prop_array");

        # Run model
        sim = pyflamegpu.CUDASimulation(m);
        sim.SimulationConfig().steps = 1;
        sim.setStepLog(slcfg);

        sim.step();

        # Fetch log
        log = sim.getRunLog();
        steps = log.getStepLog();
        assert steps.size() == 1
        slog = steps[0];
        # Property wasn't logged
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            slog.getEnvironmentPropertyFloat("float_prop2")
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            slog.getEnvironmentPropertyInt("int_prop")
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            slog.getEnvironmentPropertyArrayFloat("float_prop2")
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            slog.getEnvironmentPropertyArrayInt("int_prop")
        assert e.value.type() == "InvalidEnvProperty"
        # Property wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            slog.getEnvironmentPropertyInt("float_prop")
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            slog.getEnvironmentPropertyArrayInt("uint_prop_array")
        assert e.value.type() == "InvalidEnvPropertyType"
        # Property wrong length (array length exception not applicable to python)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            slog.getEnvironmentPropertyUInt("uint_prop_array")
        assert e.value.type() == "InvalidEnvPropertyType"        
        # Correct property settings work
        slog.getEnvironmentPropertyFloat("float_prop")
        slog.getEnvironmentPropertyArrayUInt("uint_prop_array")
        
    def test_AgentLogFrameExceptions(self):
        # Define model
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        a = m.newAgent(AGENT_NAME1);
        a.newVariableFloat("float_var");
        a.newVariableInt("int_var");
        a.newVariableUInt("uint_var");
        a.newVariableArrayFloat("float_var_array", 2);

        # Define logging configs
        lcfg = pyflamegpu.LoggingConfig(m);
        slcfg = pyflamegpu.StepLoggingConfig(lcfg);
        alcfg = slcfg.agent(AGENT_NAME1);
        alcfg.logMeanFloat("float_var");
        alcfg.logStandardDevFloat("float_var");
        alcfg.logMinInt("int_var");
        alcfg.logMaxInt("int_var");
        alcfg.logSumUInt("uint_var");

        # Run model
        sim = pyflamegpu.CUDASimulation(m);
        sim.SimulationConfig().steps = 1;
        sim.setStepLog(slcfg);
        sim.step();

        # Fetch log
        log = sim.getRunLog();
        steps = log.getStepLog();
        assert steps.size() == 1
        slog = steps[0];
        # Agent/state was not logged
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentState exception
            slog.getAgent("wrong_agent")
        assert e.value.type() == "InvalidAgentState"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentState exception
            slog.getAgent(AGENT_NAME1, "wrong_state")
        assert e.value.type() == "InvalidAgentState"
        alog = slog.getAgent(AGENT_NAME1);
        # Count was not logged
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidOperation exception
            alog.getCount()
        assert e.value.type() == "InvalidOperation"
        # Variable/Reduction wasn't logged
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentVar exception
            alog.getMean("int_var")
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentVar exception
            alog.getStandardDev("double_var")
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentVar exception
            alog.getMinFloat("float_var")
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentVar exception
            alog.getMaxFloat("float_var")
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidAgentVar exception
            alog.getSumUInt("uint_prop_array")
        assert e.value.type() == "InvalidAgentVar"
        # Property wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidVarType exception
            alog.getMinFloat("int_var")
        assert e.value.type() == "InvalidVarType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidVarType exception
            alog.getMaxFloat("int_var")
        assert e.value.type() == "InvalidVarType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidVarType exception
            alog.getSumInt("uint_var")
        assert e.value.type() == "InvalidVarType"
        # Correct property settings work
        alog.getMean("float_var")
        alog.getStandardDev("float_var")
        alog.getMinInt("int_var")
        alog.getMaxInt("int_var")
        alog.getSumUInt("uint_var")
