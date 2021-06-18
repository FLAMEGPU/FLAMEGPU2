import pytest
from unittest import TestCase
from pyflamegpu import *


# Agent arrays
AGENT_COUNT = 1024


class DeviceAgentCreationTest(TestCase):

    MandatoryOutput = """
    FLAMEGPU_AGENT_FUNCTION(MandatoryOutput, flamegpu::MsgNone, flamegpu::MsgNone) {
        unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
        FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0);
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
        return flamegpu::ALIVE;
    }"""
    
    OptionalOutput = """
    FLAMEGPU_AGENT_FUNCTION(OptionalOutput, flamegpu::MsgNone, flamegpu::MsgNone) {
        unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
        if (threadIdx.x % 2 == 0) {
            FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0);
            FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
        }
        return flamegpu::ALIVE;
    }"""
    
    MandatoryOutputWithDeath = """
    FLAMEGPU_AGENT_FUNCTION(MandatoryOutputWithDeath, flamegpu::MsgNone, flamegpu::MsgNone) {
        unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
        FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0);
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
        return flamegpu::DEAD;
    }"""
    
    OptionalOutputWithDeath = """
    FLAMEGPU_AGENT_FUNCTION(OptionalOutputWithDeath, flamegpu::MsgNone, flamegpu::MsgNone) {
        unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
        if (threadIdx.x % 2 == 0) {
            FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0);
            FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
        } else {
            return flamegpu::DEAD;
        }
        return flamegpu::ALIVE;
    }"""

    EvenThreadsOnlyCdn = """
    FLAMEGPU_AGENT_FUNCTION_CONDITION(EvenThreadsOnlyCdn) {
        return threadIdx.x % 2 == 0;
    }
    """

    ArrayVarDeviceBirth = """
    FLAMEGPU_AGENT_FUNCTION(ArrayVarDeviceBirth, flamegpu::MsgNone, flamegpu::MsgNone) {
        unsigned int i = FLAMEGPU->getVariable<unsigned int>("id") * 3;
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", i);
        FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 0, 3 + i);
        FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 1, 5 + i);
        FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 2, 9 + i);
        FLAMEGPU->agent_out.setVariable<int, 4>("array_var", 3, 17 + i);
        FLAMEGPU->agent_out.setVariable<float>("y", 14.0f + i);
        return flamegpu::DEAD;
    }"""
    
    ArrayVarDeviceBirth_DefaultWorks = """
    FLAMEGPU_AGENT_FUNCTION(ArrayVarDeviceBirth_DefaultWorks, flamegpu::MsgNone, flamegpu::MsgNone) {
        unsigned int i = FLAMEGPU->getVariable<unsigned int>("id") * 3;
        FLAMEGPU->agent_out.setVariable<unsigned int>("id", i);
        return flamegpu::DEAD;
    }
    """
    
    ArrayVarDeviceBirth_ArrayUnsuitable = """
    FLAMEGPU_AGENT_FUNCTION(ArrayVarDeviceBirth_ArrayUnsuitable, flamegpu::MsgNone, flamegpu::MsgNone) {
        FLAMEGPU->agent_out.setVariable<int>("array_var", 0);
        return flamegpu::DEAD;
    }"""
    
    def test_mandatory_output_same_state(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_same_state")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutput)
        function.setAgentOutput(agent)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        assert len(population) == AGENT_COUNT * 2
        is_1 = 0
        is_12 = 0
        for ai in population:
            id = ai.getVariableUInt("id")
            val = ai.getVariableFloat("x") - id
            if (val == 1.0) :
                is_1 += 1
            elif (val == 12.0):
                is_12 += 1
        
        assert is_1 == AGENT_COUNT
        assert is_12 == AGENT_COUNT

    def test_optional_output_same_state(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_same_state")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutput)
        function.setAgentOutput(agent)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        assert len(population) == int(AGENT_COUNT * 1.5)
        is_1 = 0
        is_12 = 0
        for ai in population:
            id = ai.getVariableUInt("id")
            val = ai.getVariableFloat("x") - id
            if (val == 1.0):
                is_1 += 1
            elif (val == 12.0):
                is_12 += 1
        
        assert is_1 == AGENT_COUNT
        assert is_12 == AGENT_COUNT/2

    def test_mandatory_output_different_state(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_different_state")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutput)
        function.setInitialState("a")
        function.setEndState("a")
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT
        assert len(population_b) == AGENT_COUNT
        for ai in population_a:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
        
        for ai in population_b:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0

    def test_optional_output_different_state(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_different_state")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutput)
        function.setInitialState("a")
        function.setEndState("a")
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT
        assert len(population_b) == AGENT_COUNT / 2
        for ai in population_a:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
        
        for ai in population_b: 
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0
        
    def test_mandatory_output_same_state_with_death(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_same_state_with_death")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutputWithDeath)
        function.setAgentOutput(agent)
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        assert len(population) == AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for ai in population:
            id = ai.getVariableUInt("id")
            val = ai.getVariableFloat("x") - id
            if (val == 1.0):
                is_1 += 1
            elif (val == 12.0):
                is_12 += 1
        
        assert is_1 == 0
        assert is_12 == AGENT_COUNT

    def test_optional_output_same_state_with_death(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_same_state_with_death")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutputWithDeath)
        function.setAgentOutput(agent)
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        AGENT_COUNT = 1024
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        assert len(population) == AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for ai in population:
            id = ai.getVariableUInt("id")
            val = ai.getVariableFloat("x") - id
            if (val == 1.0):
                is_1 += 1
            elif (val == 12.0):
                is_12 += 1
        
        assert is_1 == AGENT_COUNT / 2
        assert is_12 == AGENT_COUNT / 2

    def test_mandatory_output_different_state_with_death(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_different_state_with_death")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutputWithDeath)
        function.setInitialState("a")
        function.setEndState("a")
        function.setAgentOutput(agent, "b")
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == 0
        assert len(population_b) == AGENT_COUNT
        for ai in population_b:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0

    def test_optional_output_different_state_with_death(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_different_state_with_death")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutputWithDeath)
        function.setInitialState("a")
        function.setEndState("a")
        function.setAgentOutput(agent, "b")
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT / 2
        assert len(population_b) == AGENT_COUNT / 2
        for ai in population_a:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
        
        for ai in population_b:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0

    # Tests beyond here all also check id % 2 or id % 4
    def test_mandatory_output_different_agent(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_different_agent")
        agent = model.newAgent("agent")
        agent2 = model.newAgent("agent2")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        agent2.newVariableFloat("x")
        agent2.newVariableUInt("id")
        function = agent2.newRTCFunction("output", self.MandatoryOutput)
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent2"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        newPopulation = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population)
        cudaSimulation.getPopulationData(newPopulation, "b")
        # Validate each agent has same result
        assert len(population) == AGENT_COUNT
        assert len(newPopulation) == AGENT_COUNT
        is_1_mod2_0 = 0
        is_1_mod2_1 = 0
        for ai in population:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            if (ai.getVariableUInt("id") % 2 == 0) :
                is_1_mod2_0 += 1
            else :
                is_1_mod2_1 += 1

        assert is_1_mod2_0 == AGENT_COUNT / 2
        assert is_1_mod2_1 == AGENT_COUNT / 2
        is_12_mod2_0 = 0
        is_12_mod2_1 = 0
        for ai in newPopulation:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0
            if (ai.getVariableUInt("id") % 2 == 0):
                is_12_mod2_0 += 1
            else :
                is_12_mod2_1 += 1

        assert is_12_mod2_0 == AGENT_COUNT / 2
        assert is_12_mod2_1 == AGENT_COUNT / 2

    def test_optional_output_different_agent(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_different_agent")
        agent = model.newAgent("agent")
        agent2 = model.newAgent("agent2")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        agent2.newVariableFloat("x")
        agent2.newVariableUInt("id")
        function = agent2.newRTCFunction("output", self.OptionalOutput)
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent2"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        newPopulation = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population)
        cudaSimulation.getPopulationData(newPopulation, "b")
        # Validate each agent has same result
        assert len(population) == AGENT_COUNT
        assert len(newPopulation) == AGENT_COUNT / 2
        is_1_mod2_0 = 0
        is_1_mod2_1 = 0
        for ai in population:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            if (ai.getVariableUInt("id") % 2 == 0): 
                is_1_mod2_0 += 1
            else:
                is_1_mod2_1 += 1

        assert is_1_mod2_0 == AGENT_COUNT / 2
        assert is_1_mod2_1 == AGENT_COUNT / 2
        for ai in newPopulation:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0
            assert ai.getVariableUInt("id") % 2 == 1

    def test_mandatory_output_different_agent_with_death(self): 
        # 1024 initial agents (type 'agent2') with value 1
        # every agent outputs a new agent  (type 'agent') with value 12, and then dies
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_different_agent_with_death")
        agent = model.newAgent("agent")
        agent2 = model.newAgent("agent2")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        agent2.newVariableFloat("x")
        agent2.newVariableUInt("id")
        function = agent2.newRTCFunction("output", self.MandatoryOutputWithDeath)
        function.setAgentOutput(agent, "b")
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent2"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        newPopulation = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population)
        cudaSimulation.getPopulationData(newPopulation, "b")
        # Validate each agent has same result
        assert len(population) == 0
        assert len(newPopulation) == AGENT_COUNT
        is_1 = 0
        is_12 = 0
        is_12_mod2_0 = 0
        is_12_mod2_1 = 0
        for ai in newPopulation:
            id = ai.getVariableUInt("id")
            val = ai.getVariableFloat("x") - id
            if (val == 1.0): 
                is_1 += 1
            elif (val == 12.0):
                is_12 += 1
                if (ai.getVariableUInt("id") % 2 == 0): 
                    is_12_mod2_0 += 1
                else:
                    is_12_mod2_1 += 1
        
        assert is_1 == 0
        assert is_12 == AGENT_COUNT
        assert is_12_mod2_0 == AGENT_COUNT / 2
        assert is_12_mod2_1 == AGENT_COUNT / 2

    def test_optional_output_different_agent_with_death(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_different_agent_with_death")
        agent = model.newAgent("agent")
        agent2 = model.newAgent("agent2")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        agent2.newVariableFloat("x")
        agent2.newVariableUInt("id")
        function = agent2.newRTCFunction("output", self.OptionalOutputWithDeath)
        function.setAgentOutput(agent, "b")
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent2"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        newPopulation = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population)
        cudaSimulation.getPopulationData(newPopulation, "b")
        # Validate each agent has same result
        assert len(population) == AGENT_COUNT / 2
        assert len(newPopulation) == AGENT_COUNT / 2
        for ai in population:
            assert ai.getVariableUInt("id") % 2 == 0
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
        
        for ai in newPopulation:
            assert ai.getVariableUInt("id") % 2 == 1
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0

    def test_default_variable_value(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_default_variable_value")
        agent = model.newAgent("agent")
        agent2 = model.newAgent("agent2")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        agent.newVariableUInt("id")
        agent2.newVariableFloat("x")
        agent2.newVariableUInt("id")
        function = agent2.newRTCFunction("output", self.MandatoryOutput)
        function.setAgentOutput(agent)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        AGENT_COUNT = 1024
        population = pyflamegpu.AgentVector(model.Agent("agent2"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        newPopulation = pyflamegpu.AgentVector(model.Agent("agent"))
        # cudaSimulation.getPopulationData(population)
        cudaSimulation.getPopulationData(newPopulation)
        # Validate each new agent has default value
        assert len(newPopulation) == AGENT_COUNT
        is_15 = 0
        is_12_mod2_1 = 0
        is_12_mod2_3 = 0
        for ai in newPopulation:
            if (ai.getVariableFloat("default") == 15.0): 
                is_15 += 1
                if (ai.getVariableUInt("id") % 4 == 1): 
                    is_12_mod2_1 += 1
                elif (ai.getVariableUInt("id") % 4 == 3):
                    is_12_mod2_3 += 1

        assert is_15 == AGENT_COUNT
        assert is_12_mod2_1 == AGENT_COUNT / 4
        assert is_12_mod2_3 == AGENT_COUNT / 4

    def test_mandatory_output_same_state_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_same_state_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutput)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setInitialState("a")
        function.setEndState("b")
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT / 2
        assert len(population_b) == AGENT_COUNT
        for ai in population_a:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            assert ai.getVariableUInt("id") % 2 == 1
        
        is_1 = 0
        is_12 = 0
        is_12_mod2_1 = 0
        is_12_mod2_3 = 0
        for ai in population_b:
            if (ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0): 
                is_12 += 1
                if (ai.getVariableUInt("id") % 4 == 1):
                    is_12_mod2_1 += 1
                elif (ai.getVariableUInt("id") % 4 == 3):
                    is_12_mod2_3 += 1
                
            elif (ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0):
                is_1 += 1
                assert ai.getVariableUInt("id") % 2 == 0
            
        
        assert is_1 == AGENT_COUNT / 2
        assert is_12 == AGENT_COUNT / 2
        assert is_12_mod2_1 == AGENT_COUNT / 4
        assert is_12_mod2_3 == AGENT_COUNT / 4

    def test_optional_output_same_state_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_same_state_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutput)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setInitialState("a")
        function.setEndState("b")
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT / 2
        assert len(population_b) == AGENT_COUNT / 2 + AGENT_COUNT / 4
        is_1 = 0
        for ai in population_a:
            if (ai.getVariableFloat("x") - ai.getVariableUInt("id"), 1.0):
                is_1 += 1
                assert ai.getVariableUInt("id") % 2 == 1
            
        
        assert is_1 == len(population_a)
        is_1 = 0
        is_12 = 0
        for ai in population_b:
            if (ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0):
                is_12 += 1
                assert ai.getVariableUInt("id") % 4 == 1
            elif (ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0):
                is_1 += 1
                assert ai.getVariableUInt("id") % 2 == 0
            
        
        assert is_12 == AGENT_COUNT / 4
        assert is_1 == AGENT_COUNT / 2

    def test_mandatory_output_different_state_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_different_state_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newState("c")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutput)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setInitialState("a")
        function.setEndState("c")
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        population_c = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        cudaSimulation.getPopulationData(population_c, "c")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT / 2
        assert len(population_b) == AGENT_COUNT / 2
        assert len(population_c) == AGENT_COUNT / 2
        for ai in population_a:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            assert ai.getVariableUInt("id") % 2 == 1
        
        is_12_mod2_1 = 0
        is_12_mod2_3 = 0
        for ai in population_b:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0
            if (ai.getVariableUInt("id") % 4 == 1):
                is_12_mod2_1 += 1
            elif (ai.getVariableUInt("id") % 4 == 3):
                is_12_mod2_3 += 1

        assert is_12_mod2_1 == AGENT_COUNT / 4
        assert is_12_mod2_3 == AGENT_COUNT / 4
        for ai in population_c:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            assert ai.getVariableUInt("id") % 2 == 0

    def test_optional_output_different_state_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_different_state_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newState("c")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutput)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setInitialState("a")
        function.setEndState("c")
        function.setAgentOutput(agent, "b")
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        population_c = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        cudaSimulation.getPopulationData(population_c, "c")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT / 2
        assert len(population_b) == AGENT_COUNT / 4
        assert len(population_c) == AGENT_COUNT / 2
        for ai in population_a: 
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            assert ai.getVariableUInt("id") % 2 == 1
        
        for ai in population_b:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0
            assert ai.getVariableUInt("id") % 4 == 1
        
        for ai in population_c:
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            assert ai.getVariableUInt("id") % 2 == 0

    def test_mandatory_output_same_state_with_death_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_same_state_with_death_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutputWithDeath)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setAgentOutput(agent)
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        # 50% original agents output new agent and died, 50% original agents lived on disabled
        assert len(population) == AGENT_COUNT
        is_1 = 0
        is_12 = 0
        is_12_mod2_1 = 0
        is_12_mod2_3 = 0
        for ai in population:
            val = ai.getVariableFloat("x") - ai.getVariableUInt("id")
            if (val == 1.0):
                is_1 += 1
                assert ai.getVariableUInt("id") % 2 == 1
            elif (val == 12.0):
                is_12 += 1
                if (ai.getVariableUInt("id") % 4 == 1):
                    is_12_mod2_1 += 1
                elif (ai.getVariableUInt("id") % 4 == 3): 
                    is_12_mod2_3 += 1
                
            
        
        assert is_1 == AGENT_COUNT / 2
        assert is_12 == AGENT_COUNT / 2
        assert is_12_mod2_1 == AGENT_COUNT / 4
        assert is_12_mod2_3 == AGENT_COUNT / 4

    def test_optional_output_same_state_with_death_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_same_state_with_death_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutputWithDeath)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setAgentOutput(agent)
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        AGENT_COUNT = 1024
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population)
        # Execute model
        cudaSimulation.step()
        # Test output
        cudaSimulation.getPopulationData(population)
        # Validate each agent has same result
        # 50 % original agents did not execute so lived on = AGENT_COUNT / 2
        # 25 % original agents executed, output new agent and died
        assert len(population) == AGENT_COUNT
        is_1 = 0
        is_12 = 0
        is_1_mod2_0 = 0
        is_1_mod2_1 = 0
        for ai in population:
            val = ai.getVariableFloat("x") - ai.getVariableUInt("id")
            if (val == 1.0): 
                is_1 += 1
                if (ai.getVariableUInt("id") % 2 == 0):
                    is_1_mod2_0 += 1
                else:
                    is_1_mod2_1 += 1
                
            elif (val == 12.0):
                is_12 += 1
                assert ai.getVariableUInt("id") % 4 == 1
            #else:
            #    printf("i:%u, x:%f, id:%u\n", i, ai.getVariableFloat("x"), ai.getVariableUInt("id"))
            
        
        assert is_1 == AGENT_COUNT / 2 + AGENT_COUNT / 4
        assert is_1_mod2_0 == AGENT_COUNT / 4
        assert is_1_mod2_1 == AGENT_COUNT / 2
        assert is_12 == AGENT_COUNT / 4

    def test_mandatory_output_different_state_with_death_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_mandatory_output_different_state_with_death_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.MandatoryOutputWithDeath)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setInitialState("a")
        function.setEndState("a")
        function.setAgentOutput(agent, "b")
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT / 2
        assert len(population_b) == AGENT_COUNT / 2
        for ai in population_a:
            assert ai.getVariableUInt("id") % 2 == 1
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
        
        for ai in population_b:
            assert ai.getVariableUInt("id") % 2 == 1
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0

    def test_optional_output_different_state_with_death_with_agent_function_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_optional_output_different_state_with_death_with_agent_function_condition")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent.newVariableUInt("id")
        function = agent.newRTCFunction("output", self.OptionalOutputWithDeath)
        function.setRTCFunctionCondition(self.EvenThreadsOnlyCdn)
        function.setInitialState("a")
        function.setEndState("a")
        function.setAgentOutput(agent, "b")
        function.setAllowAgentDeath(True)
        layer1 = model.newLayer()
        layer1.addAgentFunction(function)
        # Init agent pop
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        AGENT_COUNT = 1024
        population = pyflamegpu.AgentVector(model.Agent("agent"), AGENT_COUNT)
        # Initialise agents
        for i in range(AGENT_COUNT):
            instance = population[i]
            instance.setVariableFloat("x", i + 1.0)
            instance.setVariableUInt("id", i)
        
        cudaSimulation.setPopulationData(population, "a")
        # Execute model
        cudaSimulation.step()
        # Test output
        population_a = pyflamegpu.AgentVector(model.Agent("agent"))
        population_b = pyflamegpu.AgentVector(model.Agent("agent"))
        cudaSimulation.getPopulationData(population_a, "a")
        cudaSimulation.getPopulationData(population_b, "b")
        # Validate each agent has same result
        assert len(population_a) == AGENT_COUNT / 2 + AGENT_COUNT / 4
        assert len(population_b) == AGENT_COUNT / 4
        is_1_mod2_0 = 0
        is_1_mod2_1 = 0
        for ai in population_a:
            if (ai.getVariableUInt("id") % 2 == 0):
                is_1_mod2_0 += 1
            else: 
                is_1_mod2_1 += 1
            
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 1.0
            #if (ai.getVariableFloat("x") - ai.getVariableUInt("id") != 1.0):
            #    printf("i:%u, x:%f, id:%u\n", i, ai.getVariableFloat("x"), ai.getVariableUInt("id"))
        
        assert is_1_mod2_0 == AGENT_COUNT / 4
        assert is_1_mod2_1 == AGENT_COUNT / 2
        for ai in population_b:
            assert ai.getVariableUInt("id") % 4 == 1
            assert ai.getVariableFloat("x") - ai.getVariableUInt("id") == 12.0

    def test_device_agent_birth_array_set(self):
        TEST_REFERENCE =  [3, 5, 9, 17] # Needs to be a list not tuple to allow list to be changed
        model = pyflamegpu.ModelDescription("test_device_agent_birth_array_set")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id")
        agent.newVariableArrayInt("array_var", 4)
        agent.newVariableFloat("y")
        fn = agent.newRTCFunction("out", self.ArrayVarDeviceBirth)
        fn.setAllowAgentDeath(True)
        fn.setAgentOutput(agent)
        model.newLayer().addAgentFunction(fn)
        # Run the init function
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = population[i]
            ai.setVariableUInt("id", i)
        
        sim = pyflamegpu.CUDASimulation(model)
        sim.setPopulationData(population)
        sim.step()
        sim.getPopulationData(population)
        # Check data is correct
        assert len(population) == AGENT_COUNT
        for instance in population:
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = list(instance.getVariableArrayInt("array_var"))
            for k in range(4): 
                array1[k] -= j
            
            assert j % 3 == 0
            assert array1 == TEST_REFERENCE
            assert instance.getVariableFloat("y") == 14.0 + j

    def test_device_agent_birth_default_works(self): 
        TEST_REFERENCE =  (3, 5, 9, 17) 
        model = pyflamegpu.ModelDescription("test_device_agent_birth_default_works")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id")
        agent.newVariableArrayInt("array_var", 4, TEST_REFERENCE)
        agent.newVariableFloat("y", 14.0)
        fn = agent.newRTCFunction("out", self.ArrayVarDeviceBirth_DefaultWorks)
        fn.setAllowAgentDeath(True)
        fn.setAgentOutput(agent)
        model.newLayer().addAgentFunction(fn)
        # Run the init function
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            ai = population[i]
            ai.setVariableUInt("id", i)
        
        sim = pyflamegpu.CUDASimulation(model)
        sim.setPopulationData(population)
        sim.step()
        sim.getPopulationData(population)
        # Check data is correct
        assert len(population) == AGENT_COUNT
        for instance in population:
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = instance.getVariableArrayInt("array_var")
            assert j % 3 == 0
            assert array1 == TEST_REFERENCE
            assert instance.getVariableFloat("y") == 14.0
        
