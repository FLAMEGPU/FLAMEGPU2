import pytest
from unittest import TestCase
from pyflamegpu import *


AGENT_COUNT = 1024

class DeviceAPITest(TestCase):


    agent_fn_da_set = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn_da_set, flamegpu::MessageNone, flamegpu::MessageNone) {
        // Read array from `array_var`
        // Store it's values back in `a1` -> `a4`
        FLAMEGPU->setVariable<int, 4>("array_var", 0, 2 + FLAMEGPU->getVariable<int>("id"));
        FLAMEGPU->setVariable<int, 4>("array_var", 1, 4 + FLAMEGPU->getVariable<int>("id"));
        FLAMEGPU->setVariable<int, 4>("array_var", 2, 8 + FLAMEGPU->getVariable<int>("id"));
        FLAMEGPU->setVariable<int, 4>("array_var", 3, 16 + FLAMEGPU->getVariable<int>("id"));
        return flamegpu::ALIVE;
    }"""
    
    agent_fn_da_get = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn_da_get, flamegpu::MessageNone, flamegpu::MessageNone) {
        // Read array from `array_var`
        // Store it's values back in `a1` -> `a4`
        FLAMEGPU->setVariable<int>("a1", FLAMEGPU->getVariable<int, 4>("array_var", 0));
        FLAMEGPU->setVariable<int>("a2", FLAMEGPU->getVariable<int, 4>("array_var", 1));
        FLAMEGPU->setVariable<int>("a3", FLAMEGPU->getVariable<int, 4>("array_var", 2));
        FLAMEGPU->setVariable<int>("a4", FLAMEGPU->getVariable<int, 4>("array_var", 3));
        return flamegpu::ALIVE;
    }
    """
    
    agent_fn_da_arrayunsuitable = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn_da_arrayunsuitable, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->setVariable<int>("array_var", 0);
        FLAMEGPU->setVariable<int>("array_var", 0);
        FLAMEGPU->setVariable<int>("array_var", 0);
        FLAMEGPU->setVariable<int>("array_var", 0);
        FLAMEGPU->setVariable<int>("var", FLAMEGPU->getVariable<int>("array_var"));
        return flamegpu::ALIVE;
    }
    """

    agent_fn_ad_array = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn_ad_array, flamegpu::MessageNone, flamegpu::MessageNone){
        if (threadIdx.x % 2 == 0)
            return flamegpu::DEAD;
        return flamegpu::ALIVE;
    }
    """

    def test_agent_death_array(self): 
        model = pyflamegpu.ModelDescription("test_agent_death_array")
        agent = model.newAgent("agent_name")
        agent.newVariableFloat("x")
        agent.newVariableArrayInt("array_var", 4)
        agent.newVariableFloat("y")
        agent.newVariableInt("id")
        # Do nothing, but ensure variables are made available on device
        func = agent.newRTCFunction("some_function", self.agent_fn_ad_array)
        func.setAllowAgentDeath(True)
        model.newLayer().addAgentFunction(func)
        # Init pop
        init_population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = init_population[i]
            instance.setVariableFloat("x", 12.0)
            instance.setVariableArrayInt("array_var", (2 + i, 4 + i, 8 + i, 16 + i) )
            instance.setVariableFloat("y", 14.0)
            instance.setVariableInt("id", i)
        
        # Setup Model
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.setPopulationData(init_population)
        # Run 1 step to ensure data is pushed to device
        cudaSimulation.step()
        # Recover data from device
        population = pyflamegpu.AgentVector(agent)
        cudaSimulation.getPopulationData(population)
        # Check data is intact
        # Might need to go more complicate and give different agents different values
        # They should remain in order for such a basic function, but can't guarntee
        assert len(population) == AGENT_COUNT / 2
        for instance in population:
            # Check neighbouring vars are correct
            assert instance.getVariableFloat("x") == 12.0
            assert instance.getVariableFloat("y") == 14.0
            j = instance.getVariableInt("id")
            # Check array sets are correct
            output_array = instance.getVariableArrayInt("array_var")
            assert output_array[0] == 2 + j
            assert output_array[1] == 4 + j
            assert output_array[2] == 8 + j
            assert output_array[3] == 16 + j
        


    def test_array_set(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableFloat("x")
        agent.newVariableArrayInt("array_var", 4)
        agent.newVariableFloat("y")
        agent.newVariableInt("id")
        # Do nothing, but ensure variables are made available on device
        func = agent.newRTCFunction("some_function", self.agent_fn_da_set)
        model.newLayer().addAgentFunction(func)
        # Init pop
        init_population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = init_population[i]
            instance.setVariableFloat("x", 12.0)
            instance.setVariableFloat("y", 14.0)
            instance.setVariableInt("id", i)
        
        # Setup Model
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.setPopulationData(init_population)
        # Run 1 step to ensure data is pushed to device
        cudaSimulation.step()
        # Recover data from device
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        cudaSimulation.getPopulationData(population)
        # Check data is intact
        # Might need to go more complicate and give different agents different values
        # They should remain in order for such a basic function, but can't guarntee
        assert len(population) == AGENT_COUNT
        for instance in population:
            # Check neighbouring vars are correct
            assert instance.getVariableFloat("x") == 12.0
            assert instance.getVariableFloat("y") == 14.0
            j = instance.getVariableInt("id")
            # Check array sets are correct
            output_array = instance.getVariableArrayInt("array_var")
            assert output_array[0] == 2 + j
            assert output_array[1] == 4 + j
            assert output_array[2] == 8 + j
            assert output_array[3] == 16 + j
        

    def test_array_get(self): 
        model = pyflamegpu.ModelDescription("test_array_get")
        agent = model.newAgent("agent_name")
        agent.newVariableFloat("x")
        agent.newVariableArrayInt("array_var", 4)
        agent.newVariableFloat("y")
        agent.newVariableInt("id")
        agent.newVariableInt("a1")
        agent.newVariableInt("a2")
        agent.newVariableInt("a3")
        agent.newVariableInt("a4")
        # Do nothing, but ensure variables are made available on device
        func = agent.newRTCFunction("some_function", self.agent_fn_da_get)
        model.newLayer().addAgentFunction(func)
        # Init pop
        init_population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = init_population[i]
            instance.setVariableFloat("x", 12.0)
            instance.setVariableArrayInt("array_var", (2 + i, 4 + i, 8 + i, 16 + i) )
            instance.setVariableFloat("y", 14.0)
            instance.setVariableInt("id", i)
        
        # Setup Model
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.setPopulationData(init_population)
        # Run 1 step to ensure data is pushed to device
        cudaSimulation.step()
        # Recover data from device
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        cudaSimulation.getPopulationData(population)
        # Check data is intact
        # Might need to go more complicate and give different agents different values
        # They should remain in order for such a basic function, but can't guarntee
        assert len(population) == AGENT_COUNT
        for instance in population:
            # Check neighbouring vars are correct
            assert instance.getVariableFloat("x") == 12.0
            assert instance.getVariableFloat("y") == 14.0
            j = instance.getVariableInt("id")
            # Check array sets are correct
            assert instance.getVariableInt("a1") == 2 + j
            assert instance.getVariableInt("a2") == 4 + j
            assert instance.getVariableInt("a3") == 8 + j
            assert instance.getVariableInt("a4") == 16 + j

