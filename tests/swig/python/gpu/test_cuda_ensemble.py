import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint
import time


# Global var needed in several classes
sleepDurationMilliseconds = 500

class simulateInit(pyflamegpu.HostFunctionCallback):
    # Init should always be 0th iteration/step
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Generate a basic pop
        POPULATION_TO_GENERATE = FLAMEGPU.environment.getPropertyUInt("POPULATION_TO_GENERATE")
        agent = FLAMEGPU.agent("Agent")
        for i in range(POPULATION_TO_GENERATE):
            agent.newAgent().setVariableUint("counter", 0)
class simulateExit(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        totalCounters = FLAMEGPU.agent("Agent").sumUInt("counter")
        # Add to the  file scoped atomic sum of sums. @todo
        # testSimulateSumOfSums += totalCounters
class elapsedInit(pyflamegpu.HostFunctionCallback):
    # Init should always be 0th iteration/step
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        pass
        # Generate a basic pop
        POPULATION_TO_GENERATE = FLAMEGPU.environment.getPropertyUInt("POPULATION_TO_GENERATE")
        agent = FLAMEGPU.agent("Agent")
        for i in range(POPULATION_TO_GENERATE):
            agent.newAgent().setVariableUint("counter", 0)
class elapsedStep(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Sleep each thread for a duration of time.
        seconds = sleepDurationMilliseconds / 1000.0
        time.sleep(seconds)

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
        assert mutableConfig.quiet == False
        assert mutableConfig.timing == False
        # Mutate the configuration
        mutableConfig.out_directory = "test"
        mutableConfig.out_format = "xml"
        mutableConfig.concurrent_runs = 1
        # mutableConfig.devices = std::set<int>({0}) # @todo - this will need to change.
        mutableConfig.quiet = True
        mutableConfig.timing = True
        # Check via the const ref, this should show the same value as config was a reference, not a copy.
        assert mutableConfig.out_directory == "test"
        assert mutableConfig.out_format == "xml"
        assert mutableConfig.concurrent_runs == 1
        # assert mutableConfig.devices == std::set<int>({0})  # @todo - this will need to change
        assert mutableConfig.quiet == True
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
        assert ensemble.Config().quiet == False
        argv = ["ensemble.exe", "--quiet"]
        ensemble.initialise(argv)
        assert ensemble.Config().quiet == True

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
        model.addInitFunctionCallback(init)
        exitfn = simulateExit()
        model.addExitFunctionCallback(exitfn)
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
        ensemble.Config().quiet = True
        ensemble.Config().out_format = ""  # Suppress warning
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
        model.addInitFunctionCallback(init)
        step = elapsedStep()
        model.addStepFunctionCallback(step)
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
        elapsedMillis = ensemble.getEnsembleElapsedTime()
        # Ensure the elapsedMillis is larger than a threshold.
        # Sleep accuracy via callback seems very poor.
        assert elapsedMillis > 0.0
