import pytest
from unittest import TestCase
from pyflamegpu import *

AGENT_COUNT = 1024
INIT_AGENT_COUNT = 512
NEW_AGENT_COUNT = 512

class BasicOutput(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT):
            FLAMEGPU.newAgent("agent").setVariableFloat("x", 1.0)
     
class BasicOutputCdn(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT):
            FLAMEGPU.newAgent("agent").setVariableFloat("x", 1.0)
        return pyflamegpu.CONTINUE  # New agents wont be created if EXIT is passed
    
class OutputState(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT):
            FLAMEGPU.newAgent("agent", "b").setVariableFloat("x", 1.0)

class OutputMultiAgent(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            FLAMEGPU.newAgent("agent", "b").setVariableFloat("x", 1.0)
            FLAMEGPU.newAgent("agent2").setVariableFloat("y", 2.0)
        
class BadVarName(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent").setVariableFloat("nope", 1.0)

class BadVarType(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent").setVariableInt64("x", 1.0)
        
class Getter(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            newAgt = FLAMEGPU.newAgent("agent")
            newAgt.setVariableFloat("x", newAgt.getVariableFloat("default"))
        
class GetBadVarName(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            newAgt = FLAMEGPU.newAgent("agent")
            FLAMEGPU.newAgent("agent").getVariableFloat("nope")
                
class GetBadVarType(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(NEW_AGENT_COUNT): 
            newAgt = FLAMEGPU.newAgent("agent")
            FLAMEGPU.newAgent("agent").getVariableInt64("x")
        
class ArrayVarHostBirth(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(AGENT_COUNT): 
            a = FLAMEGPU.newAgent("agent_name")
            a.setVariableUInt("id", i)
            a.setVariableIntArray4("array_var",  (2 + i, 4 + i, 8 + i, 16 + i) )
            a.setVariableInt("array_var2", 0, 3 + i)
            a.setVariableInt("array_var2", 1, 5 + i)
            a.setVariableInt("array_var2", 2, 9 + i)
            a.setVariableInt("array_var2", 3, 17 + i)
            a.setVariableFloat("y", 14.0 + i)

class ArrayVarHostBirthSetGet(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(AGENT_COUNT): 
            a = FLAMEGPU.newAgent("agent_name")
            a.setVariableUInt("id", i)
            # Set
            a.setVariableIntArray4("array_var",  (2 + i, 4 + i, 8 + i, 16 + i) )
            a.setVariableInt("array_var2", 0, 3 + i)
            a.setVariableInt("array_var2", 1, 5 + i)
            a.setVariableInt("array_var2", 2, 9 + i)
            a.setVariableInt("array_var2", 3, 17 + i)
            a.setVariableFloat("y", 14.0 + i)
            # GetSet
            a.setVariableIntArray4("array_var", a.getVariableIntArray4("array_var"))
            a.setVariableInt("array_var2", 0, a.getVariableInt("array_var2", 0))
            a.setVariableInt("array_var2", 1, a.getVariableInt("array_var2", 1))
            a.setVariableInt("array_var2", 2, a.getVariableInt("array_var2", 2))
            a.setVariableInt("array_var2", 3, a.getVariableInt("array_var2", 3))
            a.setVariableFloat("y", a.getVariableFloat("y"))
        
class ArrayVarHostBirth_DefaultWorks(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        for i in range(AGENT_COUNT): 
            FLAMEGPU.newAgent("agent_name")
               
class ArrayVarHostBirth_LenWrong(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableIntArray8("array_var", [0]*8)
        
class ArrayVarHostBirth_LenWrong2(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableInt("array_var", 5, 0)
        
class ArrayVarHostBirth_TypeWrong(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableFloatArray4("array_var", [0]*4)
        
class ArrayVarHostBirth_TypeWrong2(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableFloat("array_var", 4, 0.0)
        
class ArrayVarHostBirth_NameWrong(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableIntArray4("array_varAAAAAA", [0]*4)
        
class ArrayVarHostBirth_NameWrong2(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableInt("array_varAAAAAA", 4, 0)
        
class ArrayVarHostBirth_ArrayNotSuitableSet(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableInt("array_var", 12)
        
class ArrayVarHostBirth_ArrayNotSuitableGet(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").getVariableInt("array_var")
     
class reserved_name_step(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableInt("_", 0)
        
class reserved_name_step_array(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.newAgent("agent_name").setVariableIntArray3("_", [0]*3)

     
        
class HostAgentCreationTest(TestCase):


    def test_from_init(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.addInitFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_step(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_host_layer(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.newLayer().addHostFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_exit_condition(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutputCdn()
        model.addExitConditionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance()
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == INIT_AGENT_COUNT + NEW_AGENT_COUNT
        is_1 = 0
        is_12 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
            elif val == 12.0:
                is_12 += 1
        
        assert is_12 == INIT_AGENT_COUNT
        assert is_1 == NEW_AGENT_COUNT

    def test_from_step_empty_pop(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BasicOutput()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"))
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == NEW_AGENT_COUNT
        is_1 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 1.0:
                is_1 += 1
        
        assert is_1 == NEW_AGENT_COUNT

    def test_from_step_multi_state(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        func = OutputState()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance("a")
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize("a") == INIT_AGENT_COUNT
        assert population.getCurrentListSize("b") == NEW_AGENT_COUNT
        for i in range (population.getCurrentListSize("a")): 
            ai = population.getInstanceAt(i, "a")
            assert 12.0 == ai.getVariableFloat("x")
        
        for i in range (population.getCurrentListSize("b")): 
            ai = population.getInstanceAt(i, "b")
            assert 1.0 == ai.getVariableFloat("x")
        

    def test_from_step_multi_agent(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newState("a")
        agent.newState("b")
        agent.newVariableFloat("x")
        agent2 = model.newAgent("agent2")
        agent2.newVariableFloat("y")
        func = OutputMultiAgent()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range (INIT_AGENT_COUNT): 
            instance = population.getNextInstance("a")
            instance.setVariableFloat("x", 12.0)
        
        cuda_model.setPopulationData(population)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        cuda_model.getPopulationData(population)
        population2 = pyflamegpu.AgentPopulation(agent2)
        cuda_model.getPopulationData(population2)
        # Validate each agent has same result
        assert population.getCurrentListSize("a") == INIT_AGENT_COUNT
        assert population.getCurrentListSize("b") == NEW_AGENT_COUNT
        assert population2.getCurrentListSize() == NEW_AGENT_COUNT
        for i in range (population.getCurrentListSize("a")): 
            ai = population.getInstanceAt(i, "a")
            assert 12.0 == ai.getVariableFloat("x")
        
        for i in range (population.getCurrentListSize("b")):
            ai = population.getInstanceAt(i, "b")
            assert 1.0 == ai.getVariableFloat("x")
        
        for i in range (population2.getCurrentListSize()): 
            ai = population2.getInstanceAt(i)
            assert 2.0 == ai.getVariableFloat("y")
        

    def test_default_variable_value(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        func = BasicOutput()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), NEW_AGENT_COUNT)
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() ==  NEW_AGENT_COUNT
        is_15 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("default")
            if val == 15.0:
                is_15 += 1
        
        assert is_15 == NEW_AGENT_COUNT

    def test_bad_var_name(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BadVarName()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        # Execute model
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            cuda_model.step()
        assert e.value.type() == "InvalidAgentVar"

    def test_bad_var_type(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = BadVarType()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        # Execute model
        with pytest.raises (TypeError) as e: # Python raises TypeError rather than InvalidVarType when passing float as int
            cuda_model.step() 

    def test_GetterWorks(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        func = Getter()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        # Execute model
        cuda_model.SimulationConfig().steps = 1
        cuda_model.applyConfig()
        cuda_model.simulate()
        # Test output
        population = pyflamegpu.AgentPopulation(model.Agent("agent"), NEW_AGENT_COUNT)
        cuda_model.getPopulationData(population)
        # Validate each agent has same result
        assert population.getCurrentListSize() == NEW_AGENT_COUNT
        is_15 = 0
        for i in range(population.getCurrentListSize()):
            ai = population.getInstanceAt(i)
            val = ai.getVariableFloat("x")
            if val == 15.0:
                is_15 += 1
        
        # Every host created agent has had their default loaded from "default" and stored in "x"
        assert is_15 == NEW_AGENT_COUNT

    def test_getter_bad_var_name(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = GetBadVarName()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        # Execute model
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            cuda_model.step()
        assert e.value.type() == "InvalidAgentVar"

    def test_getter_bad_var_type(self): 
        # Define model
        model = pyflamegpu.ModelDescription("TestModel")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        func = GetBadVarType()
        model.addStepFunctionCallback(func)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        # Execute model
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            cuda_model.step()
        assert e.value.type() == "InvalidVarType"


    def test_host_agent_birth_array_set(self): 
        TEST_REFERENCE =  (2, 4, 8, 16) 
        TEST_REFERENCE2 =  (3, 5, 9, 17) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id", 0)
        agent.newVariableIntArray4("array_var")
        agent.newVariableIntArray4("array_var2")
        agent.newVariableFloat("y", 13.0)
        # Run the init function
        func = ArrayVarHostBirth()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        sim.step()
        population = pyflamegpu.AgentPopulation(agent)
        sim.getPopulationData(population)
        # Check data is correct
        assert population.getCurrentListSize() == AGENT_COUNT
        for i in range(population.getCurrentListSize()):
            instance = population.getInstanceAt(i)
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = list(instance.getVariableIntArray4("array_var"))
            array2 = list(instance.getVariableIntArray4("array_var2"))
            for k in range(4): 
                array1[k] -= j
                array2[k] -= j
            
            assert array1 == list(TEST_REFERENCE)
            assert array2 == list(TEST_REFERENCE2)
            assert instance.getVariableFloat("y") == 14 + j
        

    def test_host_agent_birth_array_set_get(self): 
        TEST_REFERENCE =  (2, 4, 8, 16) 
        TEST_REFERENCE2 =  (3, 5, 9, 17) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id", 0)
        agent.newVariableIntArray4("array_var")
        agent.newVariableIntArray4("array_var2")
        agent.newVariableFloat("y", 13.0)
        # Run the init function
        func = ArrayVarHostBirthSetGet()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        sim.step()
        population = pyflamegpu.AgentPopulation(agent)
        sim.getPopulationData(population)
        # Check data is correct
        assert population.getCurrentListSize() == AGENT_COUNT
        for i in range(population.getCurrentListSize()):
            instance = population.getInstanceAt(i)
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = list(instance.getVariableIntArray4("array_var"))
            array2 = list(instance.getVariableIntArray4("array_var2"))
            for k in range(4): 
                array1[k] -= j
                array2[k] -= j
            
            assert array1 == list(TEST_REFERENCE)
            assert array2 == list(TEST_REFERENCE2)
            assert instance.getVariableFloat("y") == 14 + j
        

    def test_host_agent_birth_array_default_works(self): 
        TEST_REFERENCE =  (2, 4, 8, 16) 
        TEST_REFERENCE2 =  (3, 5, 9, 17) 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableUInt("id", 0)
        agent.newVariableIntArray4("array_var", TEST_REFERENCE)
        agent.newVariableIntArray4("array_var2", TEST_REFERENCE2)
        agent.newVariableFloat("y", 13.0)
        # Run the init function
        func = ArrayVarHostBirth_DefaultWorks()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        sim.step()
        population = pyflamegpu.AgentPopulation(agent)
        sim.getPopulationData(population)
        # Check data is correct
        assert population.getCurrentListSize() == AGENT_COUNT
        for i in range(population.getCurrentListSize()):
            instance = population.getInstanceAt(i)
            j = instance.getVariableUInt("id")
            # Check array sets are correct
            array1 = instance.getVariableIntArray4("array_var")
            array2 = instance.getVariableIntArray4("array_var2")
            assert instance.getVariableUInt("id") == 0
            assert array1 == TEST_REFERENCE
            assert array2 == TEST_REFERENCE2
            assert instance.getVariableFloat("y") == 13.0
        

    def test_host_agent_birth_array_len_wrong(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_LenWrong()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidVarArrayLen"

    def test_host_agent_birth_array_len_wrong2(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_LenWrong2()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "OutOfRangeVarArray"

    def test_host_agent_birth_array_len_wrong3(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableInt("array_var")
        # Run the init function
        func = ArrayVarHostBirth_LenWrong()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidVarArrayLen"

    def test_host_agent_birth_array_len_wrong4(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableInt("array_var")
        # Run the init function
        func = ArrayVarHostBirth_LenWrong2()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "OutOfRangeVarArray"

    def test_host_agent_birth_array_type_wrong(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_TypeWrong()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidVarType"

    def test_host_agent_birth_array_type_wrong2(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_TypeWrong2()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidVarType"

    def test_host_agent_birth_array_name_wrong(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_NameWrong()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidAgentVar"

    def test_host_agent_birth_array_name_wrong2(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_NameWrong()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidAgentVar"

    def test_host_agent_birth_array_not_suitable_set(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_ArrayNotSuitableSet()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidAgentVar"

    def test_host_agent_birth_array_not_suitable_get(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Run the init function
        func = ArrayVarHostBirth_ArrayNotSuitableGet()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "InvalidAgentVar"


    def test_reserved_name(self): 
        model = pyflamegpu.ModelDescription("model")
        model.newAgent("agent_name")
        # Run the init function
        func = reserved_name_step()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "ReservedName"

    def test_reserved_name_array(self): 
        model = pyflamegpu.ModelDescription("model")
        model.newAgent("agent_name")
        func = reserved_name_step_array()
        model.addStepFunctionCallback(func)
        sim = pyflamegpu.CUDASimulation(model)
        with pytest.raises (pyflamegpu.FGPURuntimeException) as e:
            sim.step()
        assert e.value.type() == "ReservedName"

