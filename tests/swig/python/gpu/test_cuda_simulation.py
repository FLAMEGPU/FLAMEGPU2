import pytest
import os
from unittest import TestCase
from pyflamegpu import *
from random import randint


AGENT_COUNT = 10
externalCounter = 0
    


class IncrementCounter(pyflamegpu.HostFunction):
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
        
class Check_setEnvironmentProperty(pyflamegpu.HostFunction):
    # Override C++ method: virtual void run(FLAMEGPU_HOST_API*);
    def run(self, FLAMEGPU):
      # Check env property has expected value
      assert FLAMEGPU.environment.getPropertyInt("int") == 25
      assert FLAMEGPU.environment.getPropertyInt("int2", 0) == 22
      assert FLAMEGPU.environment.getPropertyInt("int2", 1) == 23
      assert FLAMEGPU.environment.getPropertyArrayInt("int3") == (6, 7, 8);


class TestSimulation(TestCase):
    # Ensure that getCUDAConfig() is disabled, as it would be mutable
    def test_ignored_getCUDAConfig(self):
        m = pyflamegpu.ModelDescription("test_ignored_getCUDAConfig")
        c = pyflamegpu.CUDASimulation(m)
        with pytest.raises(AttributeError):
            c.getCUDAConfig()

    def test_argparse_inputfile_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_inputfile_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--in", "test" ]
        assert c.SimulationConfig().input_file == ""
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::UnsupportedFileType exception
            c.initialise(argv)
        assert e.value.type() == "UnsupportedFileType"
        assert c.SimulationConfig().input_file == argv[2]
        # Blank init does not reset value to default
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::UnsupportedFileType exception
            c.initialise([])
        assert e.value.type() == "UnsupportedFileType"
        # assert c.SimulationConfig().input_file == ""
        assert c.SimulationConfig().input_file == argv[2]

    def test_argparse_inputfile_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_inputfile_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-i", "test" ]
        assert c.SimulationConfig().input_file == ""
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::UnsupportedFileType exception
            c.initialise(argv)
        assert e.value.type() == "UnsupportedFileType"
        assert c.SimulationConfig().input_file == argv[2]
        # Blank init does not reset value to default
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::UnsupportedFileType exception
            c.initialise([])
        assert e.value.type() == "UnsupportedFileType"
        # assert c.SimulationConfig().input_file == ""
        assert c.SimulationConfig().input_file == argv[2]

    @pytest.mark.skip(reason="no way of intercepting exit called with c/c++ code from python")
    def test_argparse_inputfile_short_exit(self):
        m = pyflamegpu.ModelDescription("test_argparse_inputfile_short_exit")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-i", "I_DO_NOT_EXIST.xml" ]
        assert c.SimulationConfig().input_file == ""
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidInputFile
            c.initialise(argv)
        assert e.value.type() == "InvalidInputFile"
        assert c.SimulationConfig().input_file == argv[2]
        # Blank init resets value to default
        argv = []
        c.initialise(argv)
        assert c.SimulationConfig().input_file == ""

    def test_argparse_steps_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_steps_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--steps", "12" ]
        assert c.SimulationConfig().steps == 1
        c.initialise(argv)
        assert c.SimulationConfig().steps == 12
        # Blank does not reset value to default
        c.initialise([])
        assert c.SimulationConfig().steps == 12
        
    def test_argparse_steps_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_steps_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-s", "12" ]
        assert c.SimulationConfig().steps == 1
        c.initialise(argv)
        assert c.SimulationConfig().steps == 12
        # Blank does not reset value to default
        c.initialise([])
        assert c.SimulationConfig().steps == 12
        
    def test_argparse_randomseed_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_randomseed_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--random", "12" ]
        assert c.SimulationConfig().random_seed != 12
        c.initialise(argv)
        assert c.SimulationConfig().random_seed == 12
        # Blank does not reset value to default
        c.initialise([])
        assert c.SimulationConfig().random_seed == 12
        
    def test_argparse_randomseed_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_randomseed_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-r", "12" ]
        assert c.SimulationConfig().random_seed != 12
        c.initialise(argv)
        assert c.SimulationConfig().random_seed == 12
        # Blank does not reset value to default
        c.initialise([])
        assert c.SimulationConfig().random_seed == 12
        
    def test_argparse_device_long(self):
        m = pyflamegpu.ModelDescription("test_argparse_device_long")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "--device", "1200" ]
        assert c.CUDAConfig().device_id == 0
        # Setting an invalid device ID is the only safe way to do this without making internal methods accessible
        # As can set to a valid device, we haven't build code for
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidCUDAdevice exception
            c.initialise(argv)
        assert e.value.type() == "InvalidCUDAdevice"
        assert c.CUDAConfig().device_id == 1200
        # Blank init does not reset value to default
        argv = []
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidCUDAdevice exception 
            c.initialise(argv)
            assert e.value.type() == "InvalidCUDAdevice"
        assert c.CUDAConfig().device_id == 1200
        
    def test_argparse_device_short(self):
        m = pyflamegpu.ModelDescription("test_argparse_device_short")
        c = pyflamegpu.CUDASimulation(m)
        argv = [ "prog.exe", "-d", "1200" ]
        assert c.CUDAConfig().device_id == 0
        # Setting an invalid device ID is the only safe way to do this without making internal methods accessible
        # As can set to a valid device, we haven't build code for
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidCUDAdevice exception 
            c.initialise(argv)
        assert e.value.type() == "InvalidCUDAdevice"
        assert c.CUDAConfig().device_id == 1200
        # Blank init does not reset value to default
        argv = []
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidCUDAdevice exception 
            c.initialise(argv)
            assert e.value.type() == "InvalidCUDAdevice"
        assert c.CUDAConfig().device_id == 1200

    def test_initialise_quiet(self):
        m = pyflamegpu.ModelDescription("test_initialise_quiet")
        c = pyflamegpu.CUDASimulation(m)
        assert c.SimulationConfig().verbosity == pyflamegpu.Verbosity_Default
        argv = [ "prog.exe", "--quiet"]
        c.initialise(argv)
        assert c.SimulationConfig().verbosity == pyflamegpu.Verbosity_Quiet

    def test_initialise_default(self):
        m = pyflamegpu.ModelDescription("test_initialise_default")
        c = pyflamegpu.CUDASimulation(m)
        assert c.SimulationConfig().verbosity == pyflamegpu.Verbosity_Default
        argv = [ "prog.exe"]
        c.initialise(argv)
        assert c.SimulationConfig().verbosity == pyflamegpu.Verbosity_Default

    def test_initialise_verbose(self):
        m = pyflamegpu.ModelDescription("test_initialise_verbose")
        c = pyflamegpu.CUDASimulation(m)
        assert c.SimulationConfig().verbosity == pyflamegpu.Verbosity_Default
        argv = [ "prog.exe", "--verbose"]
        c.initialise(argv)
        assert c.SimulationConfig().verbosity == pyflamegpu.Verbosity_Verbose

    def test_initalise_truncate(self):
        m = pyflamegpu.ModelDescription("test_initialise_truncate")
        c = pyflamegpu.CUDASimulation(m)
        assert c.SimulationConfig().truncate_log_files == False
        argv = [ "prog.exe", "--truncate"]
        c.initialise(argv)
        assert c.SimulationConfig().truncate_log_files == True

    SetGetFn = """
    FLAMEGPU_AGENT_FUNCTION(SetGetFn, flamegpu::MessageNone, flamegpu::MessageNone) {
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
            with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidVarType exception 
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
            with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidVarType exception
                i.getVariableFloat("test")
            assert e.value.type() == "InvalidVarType"


    def test_set_get_population_data_invalid_cuda_agent(self):
        m2 = pyflamegpu.ModelDescription("test_set_get_population_data_invalid_cuda_agent_m2")
        a2 = m2.newAgent("Agent2");
        m = pyflamegpu.ModelDescription("test_set_get_population_data_invalid_cuda_agent")

        pop = pyflamegpu.AgentVector(a2, AGENT_COUNT);

        c = pyflamegpu.CUDASimulation(m)
        # Test setPopulationData
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidCudaAgent exception
            c.setPopulationData(pop)
        assert e.value.type() == "InvalidAgent"
        
        # Test getPopulationData
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidCudaAgent exception
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
        m.addStepFunction(inc)
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
        m.addStepFunction(inc)
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
        FLAMEGPU_AGENT_FUNCTION(DeathFunc, flamegpu::MessageNone, flamegpu::MessageNone) {
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
        assert c.CUDAConfig().inLayerConcurrency == True
        c.initialise(argv)
        # disable concurrency
        c.CUDAConfig().inLayerConcurrency = False
        # Assert that it is disabled.
        assert c.CUDAConfig().inLayerConcurrency == False

    def test_config_randomseed_types(self):
        m = pyflamegpu.ModelDescription("test_config_inLayerConcurrency")
        c = pyflamegpu.CUDASimulation(m)
        argv = []
        c.initialise(argv)

        # Set to 0, and check the returned value matches.
        c.SimulationConfig().random_seed = 0
        assert c.SimulationConfig().random_seed == 0

        # Check that it can be set to the maximum int32_t value
        int32_max = 2 ** 31 - 1 
        c.SimulationConfig().random_seed = int32_max
        assert c.SimulationConfig().random_seed == int32_max

        # Check that it can be set to the maximum uint32_t value
        uint32_max = 2 ** 32 - 1
        c.SimulationConfig().random_seed = uint32_max
        assert c.SimulationConfig().random_seed == uint32_max

        # Check that it can be set to the maximum int32_t value
        int64_max = 2 ** 63 - 1 
        c.SimulationConfig().random_seed = int64_max
        assert c.SimulationConfig().random_seed == int64_max

        # # Check that it can be set to the maximum uint64_t value
        uint64_max = 2 ** 64 - 1
        c.SimulationConfig().random_seed = uint64_max
        assert c.SimulationConfig().random_seed == uint64_max

        # Check that it **cannot** be set to anyhing larger.
        uint64_max_plus_one = 2 ** 64
        # Expect this to throw an 
        with pytest.raises(OverflowError) as e:
            c.SimulationConfig().random_seed = uint64_max_plus_one
            assert c.SimulationConfig().random_seed == uint64_max_plus_one

        # @todo - not sure what tests to do for negative signed values, as -1 doesn't cast to an unsigned int32
        # Expect this to throw an 
        with pytest.raises(OverflowError) as e:
            c.SimulationConfig().random_seed = -1
            assert c.SimulationConfig().random_seed == -1



    CopyID = """
        FLAMEGPU_AGENT_FUNCTION(CopyID, flamegpu::MessageNone, flamegpu::MessageNone) {
            FLAMEGPU->setVariable<flamegpu::id_t>("id_copy", FLAMEGPU->getID());
            return flamegpu::ALIVE;
        }
        """

    def test_AgentID_MultipleStatesUniqueIDs(self):
        # Create agents via AgentVector to two agent states
        # Store agent IDs to an agent variable inside model
        # Export agents and check their IDs are unique
        # Also check that the id's copied during model match those at export

        model = pyflamegpu.ModelDescription("test_agentid")
        agent = model.newAgent("agent")
        agent.newVariableID("id_copy")
        agent.newState("a")
        agent.newState("b")
        af_a = agent.newRTCFunction("copy_id", self.CopyID)
        af_a.setInitialState("a")
        af_a.setEndState("a")
        af_b = agent.newRTCFunction("copy_id2", self.CopyID)
        af_b.setInitialState("b")
        af_b.setEndState("b")

        layer = model.newLayer()
        layer.addAgentFunction(af_a)
        layer.addAgentFunction(af_b)

        pop_in = pyflamegpu.AgentVector(agent, 100)

        sim = pyflamegpu.CUDASimulation(model)

        sim.setPopulationData(pop_in, "a")
        sim.setPopulationData(pop_in, "b")

        sim.step()

        pop_out_a = pyflamegpu.AgentVector(agent)
        pop_out_b = pyflamegpu.AgentVector(agent)

        sim.getPopulationData(pop_out_a, "a")
        sim.getPopulationData(pop_out_b, "b")

        ids_original = set()
        ids_copy = set()

        for a in pop_out_a:
            ids_original.add(a.getID())
            ids_copy.add(a.getVariableID("id_copy"))
            assert a.getID() == a.getVariableID("id_copy")

        for a in pop_out_b:
            ids_original.add(a.getID())
            ids_copy.add(a.getVariableID("id_copy"))
            assert a.getID() == a.getVariableID("id_copy")

        assert len(ids_original) == len(pop_out_a) + len(pop_out_b)
        assert len(ids_copy) == len(pop_out_a) + len(pop_out_b)


    # Ensure that Simulation::getSimulationConfig() is disabled, as it would be mutable
    def test_ignored_getSimulationConfig(self):
        m = pyflamegpu.ModelDescription("test_ignored_getSimulationConfig")
        c = pyflamegpu.CUDASimulation(m)
        with pytest.raises(AttributeError):
            c.getSimulationConfig()
            
            
            
    def test_setEnvironmentProperty(self):
        m = pyflamegpu.ModelDescription("test_agentid")
        m.newAgent("agent");
        m.Environment().newPropertyInt("int", 2);
        m.Environment().newPropertyArrayInt("int2", [ 12, 13 ]);
        m.Environment().newPropertyArrayInt("int3", [ 56, 57, 58 ]);
        m.newLayer().addHostFunction(Check_setEnvironmentProperty());
        s = pyflamegpu.CUDASimulation(m);
        s.SimulationConfig().steps = 1;
        # Test the getters work
        assert s.getEnvironmentPropertyInt("int") == 2;
        assert s.getEnvironmentPropertyInt("int2", 0) == 12;
        assert s.getEnvironmentPropertyInt("int2", 1) == 13;
        assert s.getEnvironmentPropertyArrayInt("int3") == (56, 57, 58);
        # Test the setters work
        s.setEnvironmentPropertyInt("int", 25);
        s.setEnvironmentPropertyInt("int2", 0, 22);
        s.setEnvironmentPropertyInt("int2", 1, 23);
        s.setEnvironmentPropertyArrayInt("int3", [6, 7, 8]);
        # Test the exceptions work
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            s.setEnvironmentPropertyInt("float", 2);  # Bad name
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.setEnvironmentPropertyInt("int3", 3);  # Bad length
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.setEnvironmentPropertyFloat("int", 3.0);  # Bad type
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            s.getEnvironmentPropertyInt("float");  # Bad name
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.getEnvironmentPropertyInt("int3");  # Bad length
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.getEnvironmentPropertyFloat("int");  # Bad type
        assert e.value.type() == "InvalidEnvPropertyType"
        # Test the exceptions work (array element methods)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            s.setEnvironmentPropertyInt("float", 0, 1);  # Bad name
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.setEnvironmentPropertyInt("int2", 10, 0);  # Bad length
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.setEnvironmentPropertyFloat("int2", 0, 1.0);  # Bad type
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            s.setEnvironmentPropertyInt("float", 0);  # Bad name
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.getEnvironmentPropertyFloat("int2", 0);  # Bad type
        assert e.value.type() == "InvalidEnvPropertyType"
        # Test the exceptions work (array methods)
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            s.setEnvironmentPropertyArrayInt("float", [56, 57, 58]);  # Bad name
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.setEnvironmentPropertyArrayInt("int3", [56, 57, 58, 59]);  # Bad length
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.setEnvironmentPropertyArrayFloat("int3", [56.0, 57.0, 58.0]);  # Bad type
        assert e.value.type() == "InvalidEnvPropertyType"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvProperty exception
            s.getEnvironmentPropertyArrayInt("float");  # Bad name
        assert e.value.type() == "InvalidEnvProperty"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:  # exception::InvalidEnvPropertyType exception
            s.getEnvironmentPropertyArrayFloat("int3");  # Bad type
        assert e.value.type() == "InvalidEnvPropertyType"
        # Run sim
        s.simulate();


class TestSimulationVerbosity(TestCase):
    """
    Tests to check the verbosity levels prodicue the expected outputs.
    Currently all disabled as SWIG does not pipe output via pythons sys.stdout/sys.stderr
    See issue #966 
    """

    @pytest.mark.skip(reason="SWIG outputs not correctly captured")
    def test_verbosity_quiet(self):
        # Test that step does a single step
        m = pyflamegpu.ModelDescription("test_verbosity_quiet")
        a = m.newAgent("Agent")
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.SimulationConfig().steps = 1
        # Verbosity QUIET
        c.SimulationConfig().verbosity = pyflamegpu.Verbosity_Quiet
        c.simulate()
        captured = self.capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    @pytest.mark.skip(reason="SWIG outputs not correctly captured")
    def test_verbosity_default(self):
        # Test that step does a single step
        m = pyflamegpu.ModelDescription("test_verbosity_default")
        a = m.newAgent("Agent")
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.SimulationConfig().steps = 1
        # Verbosity QUIET
        c.SimulationConfig().verbosity = pyflamegpu.Verbosity_Default
        c.simulate()
        captured = self.capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""

    @pytest.mark.skip(reason="SWIG outputs not correctly captured")
    def test_verbosity_verbose(self):
        # Test that step does a single step
        m = pyflamegpu.ModelDescription("test_verbosity_default")
        a = m.newAgent("Agent")
        pop = pyflamegpu.AgentVector(a, AGENT_COUNT)
        c = pyflamegpu.CUDASimulation(m)
        c.setPopulationData(pop)
        c.SimulationConfig().steps = 1
        # Verbosity QUIET
        c.SimulationConfig().verbosity = pyflamegpu.Verbosity_Verbose
        c.simulate()
        captured = self.capsys.readouterr()
        assert "Init Function Processing time" in captured.out
        assert "Processing Simulation Step 0" in captured.out
        assert "Step 0 Processing time" in captured.out
        assert "Exit Function Processing time" in captured.out
        assert "Total Processing time" in captured.out
        assert captured.err == ""

    @pytest.fixture(autouse=True)
    def capsys(self, capsys):
        # Method of applying fixture to TestCases
        # Function will run prior to any test case in the class.
        # The capsys fixture is for capturing Pythons sys.stderr and sys.stdout
        self.capsys = capsys

    def test_config_telemetry(self):
        """
        Tests the telemetry options to ensure that they are respected.
        Does not test the actual sending of telemetry.
        """

        # Get if telemetry is enabled or not
        telemetryIsEnabled = pyflamegpu.Telemetry.isEnabled()
        # This should be false in the test suite.
        assert telemetryIsEnabled == False

        # Define a simple model - doesn't need to do anything
        m = pyflamegpu.ModelDescription("tes_simulation_telemetry_function")
        a = m.newAgent("Agent")

        # Create a simulation, checking the default value matches the enabled/disabled setting
        c = pyflamegpu.CUDASimulation(m);
        assert c.SimulationConfig().telemetry == telemetryIsEnabled;

        # Enable the telemetry config option, and check that it is correct.
        c.SimulationConfig().telemetry = True
        assert True == c.SimulationConfig().telemetry

        # disable on the config object, check that it is false.
        c.SimulationConfig().telemetry = False
        assert False == c.SimulationConfig().telemetry
        
        # Flip it back to true once again, just incase it was true originally.
        c.SimulationConfig().telemetry = True
        assert True == c.SimulationConfig().telemetry

    