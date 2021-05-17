import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint


AGENT_COUNT = 10
externalCounter = 0
    


class IncrementCounter(pyflamegpu.HostFunctionCallback):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the handle function
    """

    # Define Python class 'constructor'
    def __init__(self):
        super().__init__()

    # Override C++ method: virtual void run(FLAMEGPU_HOST_API*);
    def run(self, host_api):
        global externalCounter
        print ("Hello from step function")
        externalCounter += 1


class TestSimulation(TestCase):
    def test_argparse_inputfile_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_inputfile_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--in", "test" ]
        assert c.getSimulationConfig().input_file == ""
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # UnsupportedFileType exception
            c.initialise(argv)
        assert e.value.type() == "UnsupportedFileType"
        assert c.getSimulationConfig().input_file == argv[2]
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getSimulationConfig().input_file == ""

    def test_argparse_inputfile_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_inputfile_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-i", "I_DO_NOT_EXIST.xml" ]
        assert c.getSimulationConfig().input_file == ""
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidInputFile
            c.initialise(argv)
        assert e.value.type() == "InvalidInputFile"
        assert c.getSimulationConfig().input_file == argv[2]
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getSimulationConfig().input_file == ""

    def test_argparse_steps_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_steps_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--steps", "12" ]
        assert c.getSimulationConfig().steps == 0
        c.initialise(argv)
        assert c.getSimulationConfig().steps == 12
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getSimulationConfig().steps == 0
        
    def test_argparse_steps_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_steps_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-s", "12" ]
        assert c.getSimulationConfig().steps == 0
        c.initialise(argv)
        assert c.getSimulationConfig().steps == 12
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getSimulationConfig().steps == 0
        
    def test_argparse_randomseed_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_randomseed_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--random", "12" ]
        assert c.getSimulationConfig().random_seed != 12
        c.initialise(argv)
        assert c.getSimulationConfig().random_seed == 12
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getSimulationConfig().random_seed != 12
        
    def test_argparse_randomseed_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_randomseed_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-r", "12" ]
        assert c.getSimulationConfig().random_seed != 12
        c.initialise(argv)
        assert c.getSimulationConfig().random_seed == 12
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getSimulationConfig().random_seed != 12
        
    def test_argparse_device_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_device_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--device", "1200" ]
        assert c.getCUDAConfig().device_id == 0
        # Setting an invalid device ID is the only safe way to do this without making internal methods accessible
        # As can set to a valid device, we haven't build code for
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidCUDAdevice exception
            c.initialise(argv)
        assert e.value.type() == "InvalidCUDAdevice"
        assert c.getCUDAConfig().device_id == 1200
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getCUDAConfig().device_id == 0
        
    def test_argparse_device_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_device_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-d", "1200" ]
        assert c.getCUDAConfig().device_id == 0
        # Setting an invalid device ID is the only safe way to do this without making internal methods accessible
        # As can set to a valid device, we haven't build code for
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidCUDAdevice exception 
            c.initialise(argv)
        assert e.value.type() == "InvalidCUDAdevice"
        assert c.getCUDAConfig().device_id == 1200
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.getCUDAConfig().device_id == 0


    SetGetFn = """
    FLAMEGPU_AGENT_FUNCTION(SetGetFn, flamegpu::MsgNone, flamegpu::MsgNone) {
        int i = FLAMEGPU->getVariable<int>("test");
        FLAMEGPU->setVariable<int>("test", i * 3);
        return flamegpu::ALIVE;
    }
    """
    def test_set_get_population_data(self):
        m = pyflamegpu.ModelDescription("test_set_get_population_data")
        a = m.newAgent("Agent")
        m.newLayer("Layer").addAgentFunction(a.newRTCFunction("SetGetFn", self.SetGetFn))
        a.newVariableInt("test")
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT);
        # set agent variable data
        for _i in range(AGENT_COUNT):
            i = pop[_i];
            i.setVariableInt("test", _i)
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidVarType exception 
                i.setVariableFloat("test", float(_i))
            assert e.value.type() == "InvalidVarType"
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 1
        c.setPopulationData(pop)
        # perform simulation
        c.simulate()
        c.getPopulationData(pop);
        # Check results and reset agent variable data
        for _i in range(AGENT_COUNT):
            i = pop[_i];
            assert i.getVariableInt("test") == _i * 3
            i.setVariableInt("test", _i * 2);
        # perform second simulation
        c.setPopulationData(pop)
        c.simulate();
        c.getPopulationData(pop);
        for _i in range(AGENT_COUNT):
            i = pop[_i];
            assert i.getVariableInt("test") == _i * 3 * 2
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidVarType exception
                i.getVariableFloat("test")
            assert e.value.type() == "InvalidVarType"


    def test_set_get_population_data_invalid_cuda_agent(self):
        m2 = pyflamegpu.ModelDescription("test_set_get_population_data_invalid_cuda_agent_m2")
        a2 = m2.newAgent("Agent2");
        m = pyflamegpu.ModelDescription("test_set_get_population_data_invalid_cuda_agent")

        pop = pyflamegpu.AgentVector(a2, AGENT_COUNT);

        c = pyflamegpu.CUDASimulation(m)
        # Test setPopulationData
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidCudaAgent exception
            c.setPopulationData(pop)
        assert e.value.type() == "InvalidAgent"
        
        # Test getPopulationData
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidCudaAgent exception
            c.getPopulationData(pop)
        assert e.value.type() == "InvalidAgent"

    """
    GetAgent Test is not possible without CUDA bindings
    """
        
    def test_step(self):
        global externalCounter
        # Test that step does a single step
        m = pyflamegpu.ModelDescription("test_step")
        a = m.newAgent("Agent")
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        # Create IncrementCounter object to add a step function to the special addPythonStepFunction wrapper
        inc = IncrementCounter()
        m.addStepFunctionCallback(inc)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        externalCounter = 0
        c.resetStepCounter()
        c.step()
        assert externalCounter == 1
        assert c.getStepCounter() == 1
        externalCounter = 0;
        c.resetStepCounter();
        for i in range(5):
            c.step()
        assert externalCounter == 5
        assert c.getStepCounter() == 5

    def test_simulate(self):
        global externalCounter
        # Test that step does a single step
        m = pyflamegpu.ModelDescription("test_simulate")
        a = m.newAgent("Agent")
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        # Create IncrementCounter object to add a step function to the special addPythonStepFunction wrapper
        inc = IncrementCounter()
        m.addStepFunctionCallback(inc)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        externalCounter = 0
        c.resetStepCounter()
        c.SimulationConfig().steps = 7
        c.simulate()
        assert externalCounter == 7
        assert c.getStepCounter() == 7
        externalCounter = 0;
        c.resetStepCounter();
        c.SimulationConfig().steps = 3
        c.simulate()
        assert externalCounter == 3
        assert c.getStepCounter() == 3

    DeathFunc = """
        FLAMEGPU_AGENT_FUNCTION(DeathFunc, flamegpu::MsgNone, flamegpu::MsgNone) {
            unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
            // Agents with even value for 'x' die
            if (x % 2 == 0)
                return flamegpu::DEAD;
            return flamegpu::ALIVE;
        }
        """

    def test_agent_death(self):
        m = pyflamegpu.ModelDescription("test_agent_death")
        a = m.newAgent("Agent")
        a.newVariableInt("x")
        death_func = a.newRTCFunction("DeathFunc", self.DeathFunc)
        death_func.setAllowAgentDeath(True)
        m.newLayer().addAgentFunction(death_func)
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        c = pyflamegpu.CUDASimulation(m)
        expected_output = []
        for p in pop:
            rng = randint(0,9999)
            p.setVariableInt("x", rng)
            if rng % 2 != 0:
                expected_output.append(rng)
        c.setPopulationData(pop)
        c.SimulationConfig().steps = 1
        c.simulate()
        c.getPopulationData(pop)
        assert pop.size() == len(expected_output)
        for i in range(pop.size()):
            ai = pop[i]
            assert ai.getVariableInt("x") == expected_output[i]

    def test_config_inLayerConcurrency(self):
        m = pyflamegpu.ModelDescription("test_config_inLayerConcurrency")
        c = pyflamegpu.CUDASimulation(m)
        argv = []
        # Check it's enabled by deafault
        assert c.getCUDAConfig().inLayerConcurrency == True
        c.initialise(argv)
        # disable concurrency
        c.CUDAConfig().inLayerConcurrency = False
        # Assert that it is disabled.
        assert c.getCUDAConfig().inLayerConcurrency == False
