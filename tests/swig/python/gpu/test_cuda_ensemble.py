import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint
import time, sys


# Global vars needed in several classes
sleepDurationMilliseconds = 500
tracked_err_ct = 0;
tracked_runs_ct = 0;

class simulateInit(pyflamegpu.HostFunction):
    # Init should always be 0th iteration/step
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Generate a basic pop
        POPULATION_TO_GENERATE = FLAMEGPU.environment.getPropertyUInt("POPULATION_TO_GENERATE")
        agent = FLAMEGPU.agent("Agent")
        for i in range(POPULATION_TO_GENERATE):
            agent.newAgent().setVariableUInt("counter", 0)
class simulateExit(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        totalCounters = FLAMEGPU.agent("Agent").sumUInt("counter")
        # Add to the  file scoped atomic sum of sums. @todo
        # testSimulateSumOfSums += totalCounters
class elapsedInit(pyflamegpu.HostFunction):
    # Init should always be 0th iteration/step
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Generate a basic pop
        POPULATION_TO_GENERATE = FLAMEGPU.environment.getPropertyUInt("POPULATION_TO_GENERATE")
        agent = FLAMEGPU.agent("Agent")
        for i in range(POPULATION_TO_GENERATE):
            agent.newAgent().setVariableUInt("counter", 0)
class elapsedStep(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Sleep each thread for a duration of time.
        seconds = sleepDurationMilliseconds / 1000.0
        time.sleep(seconds)
class throwException(pyflamegpu.HostFunction):
    i = 0;
    def __init__(self):
        super().__init__()
        self.i = 0;
    def run(self, FLAMEGPU):
        global tracked_runs_ct
        global tracked_err_ct
        tracked_runs_ct += 1;
        self.i+=1;
        if (self.i % 2 == 0):
            tracked_err_ct += 1;
            FLAMEGPU.agent("does not exist");  # Just cause a failure

class TestCUDAEnsemble(TestCase):

    def test_constructor(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Declare a pointer
        ensemble = None
        # Use the ctor
        # explicit CUDAEnsemble(const ModelDescription& model, int argc = 0, const char** argv = None)
        ensemble = pyflamegpu.CUDAEnsemble(model, [])
        assert ensemble != None
        # Check a property
        assert ensemble.Config().timing == False
        # Run the destructor ~CUDAEnsemble
        ensemble = None
        # Check with simple argparsing.
        argv = ["ensemble.exe", "--timing"]
        ensemble = pyflamegpu.CUDAEnsemble(model, argv)
        assert ensemble.Config().timing == True
        ensemble = None

    def test_EnsembleConfig(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)

        # Verify that the getConfig method doesn't exist, as it is ignored.
        with pytest.raises(AttributeError):
            ensemble.getConfig()

        # Get a config object.
        # EnsembleConfig &Config()
        ensemble.Config()
        mutableConfig = ensemble.Config()

        # Check the default values are correct.
        assert mutableConfig.out_directory == ""
        assert mutableConfig.out_format == "json"
        assert mutableConfig.concurrent_runs == 4
        # assert mutableConfig.devices == std::set<int>()  # @todo - this will need to change
        assert mutableConfig.verbosity == pyflamegpu.Verbosity_Default
        assert mutableConfig.timing == False
        # Mutate the configuration
        mutableConfig.out_directory = "test"
        mutableConfig.out_format = "xml"
        mutableConfig.concurrent_runs = 1
        # mutableConfig.devices = std::set<int>({0}) # @todo - this will need to change.
        mutableConfig.verbosity = pyflamegpu.Verbosity_Verbose
        mutableConfig.timing = True
        # Check via the const ref, this should show the same value as config was a reference, not a copy.
        assert mutableConfig.out_directory == "test"
        assert mutableConfig.out_format == "xml"
        assert mutableConfig.concurrent_runs == 1
        # assert mutableConfig.devices == std::set<int>({0})  # @todo - this will need to change
        assert mutableConfig.verbosity == pyflamegpu.Verbosity_Verbose
        assert mutableConfig.timing == True

    @pytest.mark.skip(reason="--help cannot be tested due to exit()")
    def test_initialise_help(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        argv = ["ensemble.exe", "--help"]
        ensemble.initialise(argv)

    def test_initialise_out(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().out_directory == ""
        assert ensemble.Config().out_format == "json"
        argv = ["ensemble.exe", "--out", "test", "xml"]
        ensemble.initialise(argv)
        assert ensemble.Config().out_directory == "test"
        assert ensemble.Config().out_format == "xml"

    def test_initialise_concurrent_runs(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().concurrent_runs == 4
        argv = ["ensemble.exe", "--concurrent", "2"]
        ensemble.initialise(argv)
        assert ensemble.Config().concurrent_runs == 2

    @pytest.mark.skip(reason="EnsembleConfig::devices is not currently swig-usable")
    def test_initialise_devices(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().devices == () # @todo
        argv = ["ensemble.exe", "--devices", "0"]
        ensemble.initialise(argv)
        assert ensemble.Config().devices == (0) # @todo

    def test_initialise_quiet(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().verbosity == pyflamegpu.Verbosity_Default
        argv = ["ensemble.exe", "--quiet"]
        ensemble.initialise(argv)
        assert ensemble.Config().verbosity == pyflamegpu.Verbosity_Quiet
    
    def test_initialise_default(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().verbosity == pyflamegpu.Verbosity_Default

    def test_initialise_verbose(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().verbosity == pyflamegpu.Verbosity_Default
        argv = ["ensemble.exe", "--verbose"]
        ensemble.initialise(argv)
        assert ensemble.Config().verbosity == pyflamegpu.Verbosity_Verbose

    def test_initialise_timing(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().timing == False
        argv = ["ensemble.exe", "--timing"]
        ensemble.initialise(argv)
        assert ensemble.Config().timing == True

    def test_initialise_truncate(self):
        model = pyflamegpu.ModelDescription("test_initialise_truncate")
        ensemble = pyflamegpu.CUDAEnsemble(model)
        assert ensemble.Config().truncate_log_files == False
        argv = [ "prog.exe", "--truncate"]
        ensemble.initialise(argv)
        assert ensemble.Config().truncate_log_files == True
        
    def test_initialise_error_level(self):
        # Create a model
        model = pyflamegpu.ModelDescription("test")
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Call initialise with differnt cli arguments, which will mutate values. Check they have the new value.
        assert ensemble.Config().error_level == pyflamegpu.CUDAEnsembleConfig.Slow
        argv = ["ensemble.exe", "-e", "0"]
        ensemble.initialise(argv)
        assert ensemble.Config().error_level == pyflamegpu.CUDAEnsembleConfig.Off
        argv = ["ensemble.exe", "--error", "1"]
        ensemble.initialise(argv)
        assert ensemble.Config().error_level == pyflamegpu.CUDAEnsembleConfig.Slow
        argv = ["ensemble.exe", "-e", "2"]
        ensemble.initialise(argv)
        assert ensemble.Config().error_level == pyflamegpu.CUDAEnsembleConfig.Fast
        argv = ["ensemble.exe", "--error", "Off"]
        ensemble.initialise(argv)
        assert ensemble.Config().error_level == pyflamegpu.CUDAEnsembleConfig.Off
        argv = ["ensemble.exe", "-e", "SLOW"]
        ensemble.initialise(argv)
        assert ensemble.Config().error_level == pyflamegpu.CUDAEnsembleConfig.Slow
        argv = ["ensemble.exe", "--error", "fast"]
        ensemble.initialise(argv)
        assert ensemble.Config().error_level == pyflamegpu.CUDAEnsembleConfig.Fast

    # Agent function used to check the ensemble runs.
    simulateAgentFn = """
    FLAMEGPU_AGENT_FUNCTION(simulateAgentFn, flamegpu::MessageNone, flamegpu::MessageNone) {
        // Increment agent's counter by 1.
        FLAMEGPU->setVariable<int>("counter", FLAMEGPU->getVariable<int>("counter") + 1);
        return flamegpu::ALIVE;
    }
    """
    def test_simulate(self):
        # Number of simulations to run.
        planCount = 2
        populationSize = 32
        # Create a model containing atleast one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", populationSize, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        afn = agent.newRTCFunction("simulateAgentFn", self.simulateAgentFn)
        # Control flow
        model.newLayer().addAgentFunction(afn)
        init = simulateInit()
        model.addInitFunction(init)
        exitfn = simulateExit()
        model.addExitFunction(exitfn)
        # Crete a small runplan, using a different number of steps per sim.
        expectedResult = 0
        plans = pyflamegpu.RunPlanVector(model, planCount)
        for idx in range(plans.size()):
            plan = plans[idx]
            plan.setSteps(idx + 1)  # Can't have 0 steps without exit condition
            # Increment the expected result based on the number of steps.
            expectedResult += (idx + 1) * populationSize
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Make it quiet to avoid outputting during the test suite
        ensemble.Config().verbosity = pyflamegpu.Verbosity_Quiet
        # Simulate the ensemble,
        ensemble.simulate(plans)

        # @todo - actually check the simulations did execute. Can't abuse atomics like in c++. 

        # An exception should be thrown if the Plan and Ensemble are for different models.
        modelTwo = pyflamegpu.ModelDescription("two")
        modelTwoPlans = pyflamegpu.RunPlanVector(modelTwo, 1)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ensemble.simulate(modelTwoPlans)
        assert e.value.type() == "InvalidArgument"
        # Exceptions can also be thrown if output_directory cannot be created, but I'm unsure how to reliably test this cross platform.

    # Logging is more thoroughly tested in Logging. Here just make sure the methods work
    def test_setStepLog(self):
        # Create a model containing atleast one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyFloat("f", 0)
        # Add an agent so that the simulation can be ran, to check for presence of logs
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        # Define the logging configuraiton.
        lcfg = pyflamegpu.LoggingConfig(model)
        lcfg.logEnvironment("f")
        slcfg = pyflamegpu.StepLoggingConfig(lcfg)
        slcfg.setFrequency(1)
        # Create a single run.
        plans = pyflamegpu.RunPlanVector(model, 1)
        plans[0].setSteps(1)
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Make it quiet to avoid outputting during the test suite
        ensemble.Config().quiet = True
        ensemble.Config().out_format = ""  # Suppress warning
        # Set the StepLog config.
        ensemble.setStepLog(slcfg)
        # Run the ensemble, generating logs
        ensemble.simulate(plans)
        # Get the logs, checking the correct number are present.
        runLogs = ensemble.getLogs()
        assert runLogs.size() == plans.size()
        for log in runLogs:
            stepLogs = log.getStepLog()
            assert stepLogs.size() == 1 + 1  # This is 1 + 1 due to the always present init log.
            expectedStepCount = 0
            for stepLog in stepLogs:
                assert stepLog.getStepCount() == expectedStepCount
                expectedStepCount += 1

        # An exception will be thrown if the step log config is for a different model.
        modelTwo = pyflamegpu.ModelDescription("two")
        lcfgTwo = pyflamegpu.LoggingConfig(modelTwo)
        slcfgTwo = pyflamegpu.StepLoggingConfig(lcfgTwo)
        slcfgTwo.setFrequency(1)
        modelTwoPlans = pyflamegpu.RunPlanVector(modelTwo, 1)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ensemble.setStepLog(slcfgTwo)
        assert e.value.type() == "InvalidArgument"

    def test_setExitLog(self):
        # Create a model containing at least one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyFloat("f", 0)
        # Add an agent so that the simulation can be ran, to check for presence of logs
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        # Define the logging configuration.
        lcfg = pyflamegpu.LoggingConfig(model)
        lcfg.logEnvironment("f")
        # Create a single run.
        plans = pyflamegpu.RunPlanVector(model, 1)
        plans[0].setSteps(1)
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Make it quiet to avoid outputting during the test suite
        ensemble.Config().quiet = True
        ensemble.Config().out_format = ""  # Suppress warning
        # Set the StepLog config.
        ensemble.setExitLog(lcfg)
        # Run the ensemble, generating logs
        ensemble.simulate(plans)
        # Get the logs, checking the correct number are present.
        runLogs = ensemble.getLogs()
        assert runLogs.size() == plans.size()
        for log in runLogs:
            exitLog = log.getExitLog()
            assert exitLog.getStepCount() == 1

        # An exception will be thrown if the step log config is for a different model.
        modelTwo = pyflamegpu.ModelDescription("two")
        lcfgTwo = pyflamegpu.LoggingConfig(modelTwo)
        modelTwoPlans = pyflamegpu.RunPlanVector(modelTwo, 1)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ensemble.setExitLog(lcfgTwo)
        assert e.value.type() == "InvalidArgument"

    def test_getLogs(self):
        # Create an ensemble with no logging enabled, but call getLogs
        # Create a model containing atleast one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        plans = pyflamegpu.RunPlanVector(model, 1)
        plans[0].setSteps(1)
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        ensemble.getLogs()
        runLogs = ensemble.getLogs()
        assert runLogs.size() == 0

    # Agent function used to check the ensemble runs.
    elapsedAgentFn = """
    FLAMEGPU_AGENT_FUNCTION(elapsedAgentFn, flamegpu::MessageNone, flamegpu::MessageNone) {
        // Increment agent's counter by 1.
        FLAMEGPU->setVariable<int>("counter", FLAMEGPU->getVariable<int>("counter") + 1);
        return flamegpu::ALIVE;
    }
    """
    def test_getEnsembleElapsedTime(self):
        # Create a model containing at least one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", 1, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        afn = agent.newRTCFunction("elapsedAgentFn", self.elapsedAgentFn)
        # Control flow
        model.newLayer().addAgentFunction(afn)
        init = elapsedInit()
        model.addInitFunction(init)
        step = elapsedStep()
        model.addStepFunction(step)
        # Create a single run.
        plans = pyflamegpu.RunPlanVector(model, 1)
        plans[0].setSteps(1)
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Make it quiet to avoid outputting during the test suite
        ensemble.Config().quiet = True
        ensemble.Config().out_format = ""  # Suppress warning
        # Get the elapsed seconds before the sim has been executed
        ensemble.getEnsembleElapsedTime()
        # Assert that it is LE zero.
        assert ensemble.getEnsembleElapsedTime() <= 0.
        # Simulate the ensemble,
        ensemble.simulate(plans)
        # Get the elapsed seconds before the sim has been executed
        elapsedSeconds = ensemble.getEnsembleElapsedTime()
        # Ensure the elapsedMillis is larger than a threshold.
        # Sleep accuracy via callback seems very poor.
        assert elapsedSeconds > 0.0
        
    def test_ErrorOff(self):
        global tracked_runs_ct
        global tracked_err_ct
        tracked_runs_ct = 0
        tracked_err_ct = 0
        # Create a model containing at least one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", 1, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        init = elapsedInit()
        model.addInitFunction(init)
        step = throwException()
        model.addStepFunction(step)
        # Create a set of 10 Run plans
        ENSEMBLE_COUNT = 10
        plans = pyflamegpu.RunPlanVector(model, ENSEMBLE_COUNT)
        plans.setSteps(1)
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Make it quiet to avoid outputting during the test suite
        ensemble.Config().quiet = True
        ensemble.Config().out_format = ""  # Suppress warning
        ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Off
        ensemble.Config().concurrent_runs = 1  # Single device/no concurrency to ensure we get consistent data
        ensemble.Config().devices = pyflamegpu.IntSet([0])
        reported_err_ct = 0;
        # Simulate the ensemble,
        reported_err_ct = ensemble.simulate(plans)
        # Check correct number of fails is reported
        assert reported_err_ct == ENSEMBLE_COUNT / 2
        assert tracked_err_ct == ENSEMBLE_COUNT / 2
        assert tracked_runs_ct == ENSEMBLE_COUNT
        
    def test_ErrorSlow(self):
        global tracked_runs_ct
        global tracked_err_ct
        tracked_runs_ct = 0
        tracked_err_ct = 0
        # Create a model containing at least one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", 1, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        init = elapsedInit()
        model.addInitFunction(init)
        step = throwException()
        model.addStepFunction(step)
        # Create a set of 10 Run plans
        ENSEMBLE_COUNT = 10
        plans = pyflamegpu.RunPlanVector(model, ENSEMBLE_COUNT)
        plans.setSteps(1)
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Make it quiet to avoid outputting during the test suite
        ensemble.Config().quiet = True
        ensemble.Config().out_format = ""  # Suppress warning
        ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Slow
        ensemble.Config().concurrent_runs = 1  # Single device/no concurrency to ensure we get consistent data
        ensemble.Config().devices = pyflamegpu.IntSet([0])
        reported_err_ct = 0;
        # Simulate the ensemble,
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ensemble.simulate(plans)
        assert e.value.type() == "EnsembleError"
        # Check correct number of fails is reported
        assert tracked_err_ct == ENSEMBLE_COUNT / 2
        assert tracked_runs_ct == ENSEMBLE_COUNT
        
    def test_ErrorSlow(self):
        global tracked_runs_ct
        global tracked_err_ct
        tracked_runs_ct = 0
        tracked_err_ct = 0
        # Create a model containing at least one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", 1, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        init = elapsedInit()
        model.addInitFunction(init)
        step = throwException()
        model.addStepFunction(step)
        # Create a set of 10 Run plans
        ENSEMBLE_COUNT = 10
        plans = pyflamegpu.RunPlanVector(model, ENSEMBLE_COUNT)
        plans.setSteps(1)
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Make it quiet to avoid outputting during the test suite
        ensemble.Config().quiet = True
        ensemble.Config().out_format = ""  # Suppress warning
        ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Fast
        ensemble.Config().concurrent_runs = 1  # Single device/no concurrency to ensure we get consistent data
        ensemble.Config().devices = pyflamegpu.IntSet([0])
        reported_err_ct = 0;
        # Simulate the ensemble,
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ensemble.simulate(plans)
        assert e.value.type() == "EnsembleError"
        # Check correct number of fails is reported
        assert tracked_err_ct == 1
        # The first run does not throw
        assert tracked_runs_ct == 2
    
class TestEnsembleVerbosity(TestCase):
    """
    Tests to check the verbosity levels produce the expected outputs.
    Currently all disabled as SWIG does not pipe output via pythons sys.stdout/sys.stderr
    See issue #966 
    """

    # Agent function used to check the ensemble runs.
    simulateAgentFn = """
    FLAMEGPU_AGENT_FUNCTION(simulateAgentFn, flamegpu::MessageNone, flamegpu::MessageNone) {
        // Increment agent's counter by 1.
        FLAMEGPU->setVariable<int>("counter", FLAMEGPU->getVariable<int>("counter") + 1);
        return flamegpu::ALIVE;
    }
    """
    @pytest.mark.skip(reason="SWIG outputs not correctly captured")
    def test_ensemble_verbosity_quiet(self):
        # Number of simulations to run.
        planCount = 2
        populationSize = 32
        # Create a model containing at least one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", populationSize, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        afn = agent.newRTCFunction("simulateAgentFn", self.simulateAgentFn)
        # Control flow
        model.newLayer().addAgentFunction(afn)
        # Crete a small runplan, using a different number of steps per sim.
        plans = pyflamegpu.RunPlanVector(model, planCount)
        for idx in range(plans.size()):
            plan = plans[idx]
            plan.setSteps(idx + 1)  # Can't have 0 steps without exit condition
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Verbosity QUIET
        ensemble.Config().verbosity = pyflamegpu.Verbosity_Quiet
        ensemble.simulate(plans)
        captured = self.capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    @pytest.mark.skip(reason="SWIG outputs not correctly captured")
    def test_ensemble_verbosity_default(self):
        # Number of simulations to run.
        planCount = 2
        populationSize = 32
        # Create a model containing at least one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", populationSize, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        afn = agent.newRTCFunction("simulateAgentFn", self.simulateAgentFn)
        # Control flow
        model.newLayer().addAgentFunction(afn)
        # Crete a small runplan, using a different number of steps per sim.
        plans = pyflamegpu.RunPlanVector(model, planCount)
        for idx in range(plans.size()):
            plan = plans[idx]
            plan.setSteps(idx + 1)  # Can't have 0 steps without exit condition
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Verbosity QUIET
        ensemble.Config().verbosity = pyflamegpu.Verbosity_Default
        ensemble.simulate(plans)
        captured = self.capsys.readouterr()
        assert "CUDAEnsemble progress" in captured.out
        assert "CUDAEnsemble completed" in captured.out
        assert captured.err == ""

    @pytest.mark.skip(reason="SWIG outputs not correctly captured")
    def test_ensemble_verbosity_verbose(self):
        # Number of simulations to run.
        planCount = 2
        populationSize = 32
        # Create a model containing atleast one agent type and function.
        model = pyflamegpu.ModelDescription("test")
        # Environmental constant for initial population
        model.Environment().newPropertyUInt("POPULATION_TO_GENERATE", populationSize, True)
        # Agent(s)
        agent = model.newAgent("Agent")
        agent.newVariableUInt("counter", 0)
        afn = agent.newRTCFunction("simulateAgentFn", self.simulateAgentFn)
        # Control flow
        model.newLayer().addAgentFunction(afn)
        # Crete a small runplan, using a different number of steps per sim.
        plans = pyflamegpu.RunPlanVector(model, planCount)
        for idx in range(plans.size()):
            plan = plans[idx]
            plan.setSteps(idx + 1)  # Can't have 0 steps without exit condition
        # Create an ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        # Verbosity QUIET
        ensemble.Config().verbosity = pyflamegpu.Verbosity_Verbose
        ensemble.simulate(plans)
        captured = self.capsys.readouterr()
        assert "CUDAEnsemble progress" in captured.out
        assert "CUDAEnsemble completed" in captured.out
        assert "Ensemble time elapsed" in captured.out
        assert captured.err == ""

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        # Method of applying fixture to TestCases
        # Function will run prior to any test case in the class.
        # The capsys fixture is for capturing Pythons sys.stderr and sys.stdout
        self.capsys = capsys
        