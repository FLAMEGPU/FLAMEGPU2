import pytest
from unittest import TestCase
from pyflamegpu import *


AGENT_NAME = "Agent";
AGENT_NAME2 = "Agent2"
FUNCTION_NAME = "Function"
LAYER_NAME = "Layer"
VARIABLE_NAME = "test"
AGENT_COUNT = 10
MULTIPLIER = 3
dMULTIPLIER = 3
externalCounter = 0
    
DeathTestFunc = """
FLAMEGPU_AGENT_FUNCTION(DeathTestFunc, MsgNone, MsgNone) {
    unsigned int x = FLAMEGPU->getVariable<unsigned int>("x");
    // Agents with even value for 'x' die
    if (x % 2 == 0)
        return DEAD;
    return ALIVE;
}
"""

class IncrementCounter(pyflamegpu.StepFunction):
    """
    pyflamegpu requires step functions to be a class which extends the StepFunction base class.
    This class must extend the handle function
    """

    # Define Python class 'constructor'
    def __init__(self):
        # Call C++ base class constructor
        pyflamegpu.StepFunction.__init__(self)

    # Override C++ method: virtual int handle(int a, int b) = 0;
    def step(self, host_api):
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
        m.addPythonStepFunction(inc)
        c = pyflamegpu.CUDAAgentModel(m)
        c.setPopulationData(pop)
        externalCounter = 0
        c.resetStepCounter()
        c.step();
        self.assertEqual(externalCounter, 1)
        self.assertEqual(c.getStepCounter(), 1)
        externalCounter = 0;
        c.resetStepCounter();
        for i in range(5):
            c.step()
        self.assertEqual(externalCounter, 5)
        self.assertEqual(c.getStepCounter(), 5)


"""
TEST(TestSimulation, Simulate) {
    // Simulation is abstract, so test via CUDAAgentModel
    // Depends on CUDAAgentModel::step()
    // Test that step does a single step
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
    m.addStepFunction(IncrementCounter);
    CUDAAgentModel c(m);
    c.setPopulationData(pop);
    externalCounter = 0;
    c.resetStepCounter();
    c.SimulationConfig().steps = 7;
    c.simulate();
    EXPECT_EQ(externalCounter, 7);
    EXPECT_EQ(c.getStepCounter(), 7u);
    externalCounter = 0;
    c.resetStepCounter();
    c.SimulationConfig().steps = 3;
    c.simulate();
    EXPECT_EQ(externalCounter, 3);
    EXPECT_EQ(c.getStepCounter(), 3u);
}

// Show that blank init resets the vals?

TEST(TestCUDAAgentModel, AgentDeath) {
    std::default_random_engine generator;
    std::uniform_int_distribution<unsigned int> distribution(0, 12);
    // Test that step does a single step
    ModelDescription m(MODEL_NAME);
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("x");
    a.newFunction("DeathFunc", DeathTestFunc).setAllowAgentDeath(true);
    m.newLayer().addAgentFunction(DeathTestFunc);
    CUDAAgentModel c(m);
    AgentPopulation pop(a, static_cast<unsigned int>(AGENT_COUNT));
    std::vector<unsigned int> expected_output;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        auto p = pop.getNextInstance();
        unsigned int rng = distribution(generator);
        p.setVariable<unsigned int>("x", rng);
        if (rng % 2 != 0)
            expected_output.push_back(rng);
    }
    c.setPopulationData(pop);
    c.SimulationConfig().steps = 1;
    c.simulate();
    c.getPopulationData(pop);
    EXPECT_EQ(static_cast<size_t>(pop.getCurrentListSize()), expected_output.size());
    for (unsigned int i = 0; i < pop.getCurrentListSize(); ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        // Check x is an expected value
        EXPECT_EQ(expected_output[i], ai.getVariable<unsigned int>("x"));
    }
}

"""
