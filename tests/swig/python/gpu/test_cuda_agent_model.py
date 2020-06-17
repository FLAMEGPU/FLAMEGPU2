import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint


AGENT_NAME = "Agent";
AGENT_NAME2 = "Agent2"
FUNCTION_NAME = "Function"
LAYER_NAME = "Layer"
VARIABLE_NAME = "test"
AGENT_COUNT = 10
MULTIPLIER = 3
dMULTIPLIER = 3
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
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "--in", "test" ]
        self.assertEqual(c.getSimulationConfig().xml_input_file, "")
        with pytest.raises(RuntimeError) as e:  # UnsupportedFileType exception is thrown as python RuntimeError by swig
            c.initialise(argv)
        self.assertIn("File 'test' is not a type which can be read", str(e.value))
        self.assertEqual(c.getSimulationConfig().xml_input_file, argv[2])
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().xml_input_file, "")

    def test_argparse_inputfile_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_inputfile_short")
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "-i", "test.xml" ]
        self.assertEqual(c.getSimulationConfig().xml_input_file, "")
        with pytest.raises(RuntimeError) as e:  # InvalidInputFile exception is thrown as python RuntimeError by swig
            c.initialise(argv)
        self.assertIn("File could not be opened", str(e.value))
        self.assertEqual(c.getSimulationConfig().xml_input_file, argv[2])
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().xml_input_file, "")

    def test_argparse_steps_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_steps_long")
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "--steps", "12" ]
        self.assertEqual(c.getSimulationConfig().steps, 0)
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().steps, 12)
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().steps, 0)
        
    def test_argparse_steps_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_steps_short")
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "-s", "12" ]
        self.assertEqual(c.getSimulationConfig().steps, 0)
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().steps, 12)
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().steps, 0)
        
    def test_argparse_randomseed_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_randomseed_long")
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "--random", "12" ]
        self.assertNotEqual(c.getSimulationConfig().random_seed, 12)
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().random_seed, 12)
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertNotEqual(c.getSimulationConfig().random_seed, 12)
        
    def test_argparse_randomseed_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_randomseed_short")
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "-r", "12" ]
        self.assertNotEqual(c.getSimulationConfig().random_seed, 12)
        c.initialise(argv)
        self.assertEqual(c.getSimulationConfig().random_seed, 12)
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertNotEqual(c.getSimulationConfig().random_seed, 12)
        
    def test_argparse_device_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_device_long")
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "--device", "1200" ]
        self.assertEqual(c.getCUDAConfig().device_id, 0)
        # Setting an invalid device ID is the only safe way to do this without making internal methods accessible
        # As can set to a valid device, we haven't build code for
        with pytest.raises(RuntimeError) as e:  # InvalidCUDAdevice exception is thrown as python RuntimeError by swig
            c.initialise(argv)
        self.assertIn("Error setting CUDA device to '1200'", str(e.value))
        self.assertEqual(c.getCUDAConfig().device_id, 1200)
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertEqual(c.getCUDAConfig().device_id, 0)
        
    def test_argparse_device_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_device_short")
        c = pyflamegpu.CUDAAgentModel(m)
        argv = [ "prog.exe", "-d", "1200" ]
        self.assertEqual(c.getCUDAConfig().device_id, 0)
        # Setting an invalid device ID is the only safe way to do this without making internal methods accessible
        # As can set to a valid device, we haven't build code for
        with pytest.raises(RuntimeError) as e:  # InvalidCUDAdevice exception is thrown as python RuntimeError by swig
            c.initialise(argv)
        self.assertIn("Error setting CUDA device to '1200'", str(e.value))
        self.assertEqual(c.getCUDAConfig().device_id, 1200)
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        self.assertEqual(c.getCUDAConfig().device_id, 0)


    SetGetFn = """
    FLAMEGPU_AGENT_FUNCTION(SetGetFn, MsgNone, MsgNone) {
        int i = FLAMEGPU->getVariable<int>("test");
        FLAMEGPU->setVariable<int>("test", i * 3);
        return ALIVE;
    }
    """
    def test_set_get_population_data(self):
        m = pyflamegpu.ModelDescription("test_set_get_population_data")
        a = m.newAgent("Agent")
        m.newLayer("Layer").addAgentFunction(a.newRTCFunction("SetGetFn", self.SetGetFn))
        a.newVariableInt("test")
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT);
        # set agent variable data
        for _i in range(AGENT_COUNT):
            i = pop.getNextInstance();
            i.setVariableInt("test", _i)
            with pytest.raises(RuntimeError) as e:  # InvalidVarType exception is thrown as python RuntimeError by swig
                i.setVariableFloat("test", float(_i))
            self.assertIn("This expects 'int', but 'float' was requested", str(e.value))
        c = pyflamegpu.CUDAAgentModel(m)
        c.SimulationConfig().steps = 1
        c.setPopulationData(pop)
        # perform simulation
        c.simulate()
        c.getPopulationData(pop);
        # Check results and reset agent variable data
        for _i in range(AGENT_COUNT):
            i = pop.getInstanceAt(_i);
            self.assertEqual(i.getVariableInt("test"), _i * 3)
            i.setVariableInt("test", _i * 2);
        # perform second simulation
        c.setPopulationData(pop)
        c.simulate();
        c.getPopulationData(pop);
        for _i in range(AGENT_COUNT):
            i = pop.getInstanceAt(_i);
            self.assertEqual(i.getVariableInt("test"), _i * 3 * 2)
            with pytest.raises(RuntimeError) as e:  # InvalidVarType exception is thrown as python RuntimeError by swig
                i.getVariableFloat("test")
            self.assertIn("This expects 'int', but 'float' was requested", str(e.value))


    def test_set_get_population_data_invalid_cuda_agent(self):
        m2 = pyflamegpu.ModelDescription("test_set_get_population_data_invalid_cuda_agent_m2")
        a2 = m2.newAgent("Agent2");
        m = pyflamegpu.ModelDescription("test_set_get_population_data_invalid_cuda_agent")

        pop = pyflamegpu.AgentPopulation(a2, AGENT_COUNT);

        c = pyflamegpu.CUDAAgentModel(m)
        # Test setPopulationData
        with pytest.raises(RuntimeError) as e:  # InvalidCudaAgent exception is thrown as python RuntimeError by swig
            c.setPopulationData(pop)
        self.assertIn("Error: Agent ('Agent2') was not found", str(e.value))
        
        # Test getPopulationData
        with pytest.raises(RuntimeError) as e:  # InvalidCudaAgent exception is thrown as python RuntimeError by swig
            c.getPopulationData(pop)
        self.assertIn("Error: Agent ('Agent2') was not found", str(e.value))

    """
    GetAgent Test is not possible without CUDA bindings
    """
        
    def test_step(self):
        global externalCounter
        # Test that step does a single step
        m = pyflamegpu.ModelDescription("test_step")
        a = m.newAgent("Agent")
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        # Create IncrementCounter object to add a step function to the special addPythonStepFunction wrapper
        inc = IncrementCounter()
        m.addStepFunctionCallback(inc)
        c = pyflamegpu.CUDAAgentModel(m)
        c.setPopulationData(pop)
        externalCounter = 0
        c.resetStepCounter()
        c.step()
        self.assertEqual(externalCounter, 1)
        self.assertEqual(c.getStepCounter(), 1)
        externalCounter = 0;
        c.resetStepCounter();
        for i in range(5):
            c.step()
        self.assertEqual(externalCounter, 5)
        self.assertEqual(c.getStepCounter(), 5)

    def test_simulate(self):
        global externalCounter
        # Test that step does a single step
        m = pyflamegpu.ModelDescription("test_simulate")
        a = m.newAgent("Agent")
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        # Create IncrementCounter object to add a step function to the special addPythonStepFunction wrapper
        inc = IncrementCounter()
        m.addStepFunctionCallback(inc)
        c = pyflamegpu.CUDAAgentModel(m)
        c.setPopulationData(pop)
        externalCounter = 0
        c.resetStepCounter()
        c.SimulationConfig().steps = 7
        c.simulate()
        self.assertEqual(externalCounter, 7)
        self.assertEqual(c.getStepCounter(), 7)
        externalCounter = 0;
        c.resetStepCounter();
        c.SimulationConfig().steps = 3
        c.simulate()
        self.assertEqual(externalCounter, 3)
        self.assertEqual(c.getStepCounter(), 3)

    DeathFunc = """
        FLAMEGPU_AGENT_FUNCTION(DeathFunc, MsgNone, MsgNone) {
            unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
            // Agents with even value for 'x' die
            if (x % 2 == 0)
                return DEAD;
            return ALIVE;
        }
        """

    def test_agent_death(self):
        m = pyflamegpu.ModelDescription("test_agent_death")
        a = m.newAgent("Agent")
        a.newVariableInt("x")
        death_func = a.newRTCFunction("DeathFunc", self.DeathFunc)
        death_func.setAllowAgentDeath(True)
        m.newLayer().addAgentFunction(death_func)
        pop = pyflamegpu.AgentPopulation(a, AGENT_COUNT)
        c = pyflamegpu.CUDAAgentModel(m)
        expected_output = []
        for i in range(AGENT_COUNT):
            p = pop.getNextInstance()
            rng = randint(0,9999)
            p.setVariableInt("x", rng)
            if rng % 2 != 0:
                expected_output.append(rng)
        c.setPopulationData(pop)
        c.SimulationConfig().steps = 1
        c.simulate()
        c.getPopulationData(pop)
        self.assertEqual(pop.getCurrentListSize(), len(expected_output))
        for i in range(AGENT_COUNT):
            ai = pop.getInstanceAt(i)
            self.assertEqual(ai.getVariableInt("x"), expected_output[i])
