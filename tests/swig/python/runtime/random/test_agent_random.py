import pytest
from unittest import TestCase
from pyflamegpu import *

AGENT_COUNT = 5

class AgentRandomTest(TestCase):

    random1_func = """
    FLAMEGPU_AGENT_FUNCTION(random1_func, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->setVariable<float>("a", FLAMEGPU->random.uniform<float>());
        FLAMEGPU->setVariable<float>("b", FLAMEGPU->random.uniform<float>());
        FLAMEGPU->setVariable<float>("c", FLAMEGPU->random.uniform<float>());

        return flamegpu::ALIVE;
    }
    """
    
    random2_func = """
    FLAMEGPU_AGENT_FUNCTION(random2_func, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->setVariable<float>("uniform_float", FLAMEGPU->random.uniform<float>());
        FLAMEGPU->setVariable<double>("uniform_double", FLAMEGPU->random.uniform<double>());

        FLAMEGPU->setVariable<float>("normal_float", FLAMEGPU->random.normal<float>());
        FLAMEGPU->setVariable<double>("normal_double", FLAMEGPU->random.normal<double>());

        FLAMEGPU->setVariable<float>("logNormal_float", FLAMEGPU->random.logNormal<float>(0, 1));
        FLAMEGPU->setVariable<double>("logNormal_double", FLAMEGPU->random.logNormal<double>(0, 1));

        // char
        FLAMEGPU->setVariable<char>("uniform_char", FLAMEGPU->random.uniform<char>(CHAR_MIN, CHAR_MAX));
        FLAMEGPU->setVariable<unsigned char>("uniform_u_char", FLAMEGPU->random.uniform<unsigned char>(0, UCHAR_MAX));
        // short
        FLAMEGPU->setVariable<int16_t>("uniform_short", FLAMEGPU->random.uniform<int16_t>(INT16_MIN, INT16_MAX));
        FLAMEGPU->setVariable<uint16_t>("uniform_u_short", FLAMEGPU->random.uniform<uint16_t>(0, UINT16_MAX));
        // int
        FLAMEGPU->setVariable<int32_t>("uniform_int", FLAMEGPU->random.uniform<int32_t>(INT32_MIN, INT32_MAX));
        FLAMEGPU->setVariable<uint32_t>("uniform_u_int", FLAMEGPU->random.uniform<uint32_t>(0, UINT32_MAX));
        // long long
        FLAMEGPU->setVariable<int64_t>("uniform_longlong", FLAMEGPU->random.uniform<int64_t>(INT64_MIN, INT64_MAX));
        FLAMEGPU->setVariable<uint64_t>("uniform_u_longlong", FLAMEGPU->random.uniform<uint64_t>(0, UINT64_MAX));

        return flamegpu::ALIVE;
    }
    """

    def test_agent_random_check(self):
        model = pyflamegpu.ModelDescription("test_agent_random_check")
        agent = model.newAgent("agent")
        agent.newVariableFloat("a")
        agent.newVariableFloat("b")
        agent.newVariableFloat("c")
        af = agent.newRTCFunction("random1", self.random1_func)
        init_population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for instance in init_population:
            instance.setVariableFloat("a", 0)
            instance.setVariableFloat("b", 0)
            instance.setVariableFloat("c", 0)
        
        layer = model.newLayer("layer")
        layer.addAgentFunction(af)

        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.SimulationConfig().steps = 1
        args_1 =  ("process.exe", "-r", "0", "-s", "1")
        args_2 =  ("process.exe", "-r", "1", "-s", "1")
        
        results1 = []
        results2 = []
        
            
        # Test Model 1
        # Do agents generate different random numbers
        # Does random number change each time it's called

        # Seed random
        cudaSimulation.initialise(args_1)
        cudaSimulation.setPopulationData(init_population)
        cudaSimulation.simulate()
        cudaSimulation.getPopulationData(population)

        a1 = b1 = c1 = a2 = b2 = c2 = -1
        for i in range(len(population)): 
            if (i != 0):
                a2 = a1
                b2 = b1
                c2 = c1
            
            instance = population[i]
            a1 = instance.getVariableFloat("a")
            b1 = instance.getVariableFloat("b")
            c1 = instance.getVariableFloat("c")
            results1.append((a1, b1, c1))
            if (i != 0):
                # Different agents get different random numbers
                assert a1 != a2
                assert b1 != b2
                assert c1 != c2
            
            # Multiple calls get multiple random numbers
            assert a1 != b1
            assert b1 != c1
            assert a1 != c1
        
        assert len(results1) == AGENT_COUNT
    
        # Test Model 2
        # Different seed produces different random numbers
        # Seed random
        cudaSimulation.initialise(args_2)
        cudaSimulation.setPopulationData(init_population)
        cudaSimulation.simulate()
        cudaSimulation.getPopulationData(population)

        for i in range(len(population)):
            instance = population[i]
            results2.append((
                instance.getVariableFloat("a"),
                instance.getVariableFloat("b"),
                instance.getVariableFloat("c")
            ))
        
        assert len(results2) == AGENT_COUNT
        for i in range (len(results1)): 
            # Different seed produces different results
            assert results1[i] != results2[i]
        
        # Test Model 3
        # Different seed produces different random numbers
        results2.clear()
        # Seed random
        cudaSimulation.initialise(args_1)
        cudaSimulation.setPopulationData(init_population)
        cudaSimulation.simulate()
        cudaSimulation.getPopulationData(population)

        for i in range(len(population)):
            instance = population[i]
            results2.append((
                instance.getVariableFloat("a"),
                instance.getVariableFloat("b"),
                instance.getVariableFloat("c")
            ))
        
        assert len(results2) == AGENT_COUNT
        for i in range (len(results1)):
            # Same seed produces same results
            assert results1[i] == results2[i]
            
        
    def test_agent_random_functions_no_except(self): 
        model = pyflamegpu.ModelDescription("test_agent_random_functions_no_except")
        agent = model.newAgent("agent")

        agent.newVariableFloat("uniform_float")
        agent.newVariableDouble("uniform_double")

        agent.newVariableFloat("normal_float")
        agent.newVariableDouble("normal_double")

        agent.newVariableFloat("logNormal_float")
        agent.newVariableDouble("logNormal_double")

        # char
        agent.newVariableChar("uniform_char")
        agent.newVariableUChar("uniform_u_char")
        # short
        agent.newVariableInt16("uniform_short")
        agent.newVariableUInt16("uniform_u_short")
        # int
        agent.newVariableInt32("uniform_int")
        agent.newVariableUInt32("uniform_u_int")
        # long long
        agent.newVariableInt64("uniform_longlong")
        agent.newVariableUInt64("uniform_u_longlong")

        do_random = agent.newRTCFunction("random2", self.random2_func)

        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        
        layer = model.newLayer("layer")
        layer.addAgentFunction(do_random)

        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.SimulationConfig().steps = 1
        cudaSimulation.setPopulationData(population)
        cudaSimulation.simulate()
        # Success if we get this far without an exception being thrown.


    def test_agent_random_array_resize_no_except(self): 
        # TODO(Rob): Can't yet control agent population up/down
        # Success if we get this far without an exception being thrown.
        pass
