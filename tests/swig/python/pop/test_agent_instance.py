import pytest
from unittest import TestCase
from pyflamegpu import *
 
INIT_AGENT_COUNT = 10
AGENT_COUNT = 1024

class AgentInstanceTest(TestCase):

    agent_fn_ap1 = """
    FLAMEGPU_AGENT_FUNCTION(agent_fn_ap1, MsgNone, MsgNone){
        // do nothing
        return ALIVE;
    }
    """

    def test_default_variable_value(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_default_variable_value")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range(INIT_AGENT_COUNT):
           instance = population.getNextInstance()
           assert instance.getVariableFloat("x") == float(0)
           assert instance.getVariableFloat("default") == float(15.0)
        

    def test_getter_bad_var_name(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_getter_bad_var_name")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range(INIT_AGENT_COUNT):
            instance = population.getNextInstance()
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.getVariableFloat("nope")
            assert e.value.type() == "InvalidAgentVar"
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.getVariableFloat("this is not valid")
            assert e.value.type() == "InvalidAgentVar"
        

    def test_getter_bad_var_type(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_getter_bad_var_type")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range(INIT_AGENT_COUNT):
            instance = population.getNextInstance()
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.getVariableInt64("x")
            assert e.value.type() == "InvalidVarType"
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.getVariableUInt("default")
            assert e.value.type() == "InvalidVarType"
        

    def test_setter_bad_var_name(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_getter_bad_var_type")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range(INIT_AGENT_COUNT):
            instance = population.getNextInstance()
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.setVariableFloat("nope", 1.0)
            assert e.value.type() == "InvalidAgentVar"
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.setVariableFloat("this is not valid", 1.0)
            assert e.value.type() == "InvalidAgentVar"
        

    def test_setter_bad_var_type(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_setter_bad_var_type")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableFloat("default", 15.0)
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range(INIT_AGENT_COUNT):
            instance = population.getNextInstance()
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.setVariableInt64("x", 1)
            assert e.value.type() == "InvalidVarType"
            with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
                instance.setVariableUInt("default", 1)
            assert e.value.type() == "InvalidVarType"
        

    def test_setter_and_getter_work(self): 
        # Define model
        model = pyflamegpu.ModelDescription("test_setter_and_getter_work")
        agent = model.newAgent("agent")
        agent.newVariableUInt("x")
        # Init agent pop
        cuda_model = pyflamegpu.CUDASimulation(model)
        population = pyflamegpu.AgentPopulation(agent, INIT_AGENT_COUNT)
        # Initialise agents
        for i in range(INIT_AGENT_COUNT):
            instance = population.getNextInstance()
            instance.setVariableUInt("x", i)
            assert instance.getVariableUInt("x") == i
        


    def test_set_via_agent_instance(self): 
        model = pyflamegpu.ModelDescription("test_set_via_agent_instance")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableIntArray4("array_var")
        agent.newVariableFloat("y")
        # Do nothing, but ensure variables are made available on device
        func = agent.newRTCFunction("some_function", self.agent_fn_ap1)
        model.newLayer().addAgentFunction(func)
        # Init pop
        init_population = pyflamegpu.AgentPopulation(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = init_population.getNextInstance("default")
            instance.setVariableFloat("x", 12.0)
            instance.setVariableIntArray4("array_var", [2, 4, 8, 16])
            instance.setVariableFloat("y", 14.0)
        
        # Setup Model
        cuda_model = pyflamegpu.CUDASimulation(model)
        cuda_model.setPopulationData(init_population)
        # Run 1 step to ensure data is pushed to device
        cuda_model.step()
        # Recover data from device
        population = pyflamegpu.AgentPopulation(agent, AGENT_COUNT)
        cuda_model.getPopulationData(population)
        # Check data is intact
        # Might need to go more complicate and give different agents different values
        # They should remain in order for such a basic function, but can't guarantee
        assert population.getCurrentListSize() == AGENT_COUNT
        for i in range(population.getCurrentListSize()):
            instance = population.getInstanceAt(i)
            assert instance.getVariableFloat("x") == 12.0
            output_array = instance.getVariableIntArray4("array_var")
            test_array = (2, 4, 8, 16)
            assert output_array == test_array
            assert instance.getVariableFloat("y") == 14.0
        
    def test_set_via_agent_instance2(self): 
        model = pyflamegpu.ModelDescription("test_set_via_agent_instance2")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x")
        agent.newVariableIntArray4("array_var")
        agent.newVariableFloat("y")
        # Do nothing, but ensure variables are made available on device
        func = agent.newRTCFunction("some_function", self.agent_fn_ap1)
        model.newLayer().addAgentFunction(func)
        # Init pop
        init_population = pyflamegpu.AgentPopulation(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = init_population.getNextInstance("default")
            instance.setVariableFloat("x", 12.0)
            instance.setVariableInt("array_var", 0, 2)
            instance.setVariableInt("array_var", 1, 4)
            instance.setVariableInt("array_var", 2, 8)
            instance.setVariableInt("array_var", 3, 16)
            instance.setVariableFloat("y", 14.0)
        
        # Setup Model
        cuda_model = pyflamegpu.CUDASimulation(model)
        cuda_model.setPopulationData(init_population)
        # Run 1 step to ensure data is pushed to device
        cuda_model.step()
        # Recover data from device
        population = pyflamegpu.AgentPopulation(agent, AGENT_COUNT)
        cuda_model.getPopulationData(population)
        # Check data is intact
        # Might need to go more complicate and give different agents different values
        # They should remain in order for such a basic function, but can't guarntee
        assert population.getCurrentListSize() == AGENT_COUNT
        for i in range(population.getCurrentListSize()):
            instance = population.getInstanceAt(i)
            assert instance.getVariableFloat("x") == 12.0
            test_array = ( 2, 4, 8, 16 )
            output_val = instance.getVariableInt("array_var", 0)
            assert output_val == test_array[0]
            output_val = instance.getVariableInt("array_var", 1)
            assert output_val == test_array[1]
            output_val = instance.getVariableInt("array_var", 2)
            assert output_val == test_array[2]
            output_val = instance.getVariableInt("array_var", 3)
            assert output_val == test_array[3]
            assert instance.getVariableFloat("y") == 14.0
        

    def test_agent_instance_array_default_works(self): 
        TEST_REFERENCE  =  (2, 4, 8, 16) 
        model = pyflamegpu.ModelDescription("test_agent_instance_array_default_works")
        agent = model.newAgent("agent")
        agent.newVariableFloat("x", 12.0)
        agent.newVariableIntArray4("array_var", TEST_REFERENCE)
        agent.newVariableFloat("y", 13.0)
        # Do nothing, but ensure variables are made available on device
        func = agent.newRTCFunction("some_function", self.agent_fn_ap1)
        model.newLayer().addAgentFunction(func)
        # Init pop
        init_population = pyflamegpu.AgentPopulation(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT):
            instance = init_population.getNextInstance("default")
            test = instance.getVariableIntArray4("array_var")
            assert test == TEST_REFERENCE
            assert instance.getVariableFloat("x") == 12.0
            assert instance.getVariableFloat("y") == 13.0
        

    def test_agent_instance_array_type_wrong(self) : 
        model = pyflamegpu.ModelDescription("test_agent_instance_array_type_wrong")
        agent = model.newAgent("agent")
        agent.newVariableIntArray4("array_var")
        # Init pop
        init_population = pyflamegpu.AgentPopulation(agent, 1)
        instance = init_population.getNextInstance("default")

        # Check for expected exceptions
        with pytest.raises(TypeError) as e: # Will be a type error in Python rather than InvalidVarType
            instance.setVariableFloatArray4("array_var", ())
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableFloatArray4("array_var")
        assert e.value.type() == "InvalidVarType"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.setVariableFloat("array_var", 0, 2)
        assert e.value.type() == "InvalidVarType"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableFloat("array_var", 0)
        assert e.value.type() == "InvalidVarType"

    def test_agent_instance_array_len_wrong(self): 
        model = pyflamegpu.ModelDescription("test_agent_instance_array_len_wrong")
        agent = model.newAgent("agent_name")
        agent.newVariableInt("x")
        agent.newVariableIntArray4("array_var")
        # Init pop
        init_population = pyflamegpu.AgentPopulation(agent, 1)
        instance = init_population.getNextInstance("default")
        # Check for expected exceptions
        with pytest.raises(TypeError) as e: # Will be a type error in Python rather than InvalidVarArrayLen
            instance.setVariableIntArray8("x", ())
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableIntArray8("x")
        assert e.value.type() == "InvalidVarArrayLen"
        with pytest.raises(TypeError) as e: # Will be a type error in Python rather than InvalidVarArrayLen
            instance.setVariableIntArray8("array_var", ())
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableIntArray8("array_var")
        assert e.value.type() == "InvalidVarArrayLen"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.setVariableInt("x", 10, 0)
        assert e.value.type() == "OutOfRangeVarArray"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableInt("x", 1)
        assert e.value.type() == "OutOfRangeVarArray"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.setVariableInt("array_var", 10, 0)
        assert e.value.type() == "OutOfRangeVarArray"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableInt("array_var", 10)
        assert e.value.type() == "OutOfRangeVarArray"

    def test_agent_instance_array_name_wrong(self): 
        model = pyflamegpu.ModelDescription("test_agent_instance_array_name_wrong")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Init pop
        init_population = pyflamegpu.AgentPopulation(agent, 1)
        instance = init_population.getNextInstance("default")

        # Check for expected exceptions
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.setVariableFloatArray4("array_varAAAAAA", (1,2,3,4))
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableFloatArray4("array_varAAAAAA")
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.setVariableInt("array_varAAAAAA", 0, 2)
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableInt("array_varAAAAAA", 0)
        assert e.value.type() == "InvalidAgentVar"

    def test_agent_instance_array_not_suitable(self): 
        model = pyflamegpu.ModelDescription("test_agent_instance_array_not_suitable")
        agent = model.newAgent("agent_name")
        agent.newVariableIntArray4("array_var")
        # Init pop
        init_population = pyflamegpu.AgentPopulation(agent, 1)
        instance = init_population.getNextInstance("default")
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.setVariableInt("array_var", 0)
        assert e.value.type() == "InvalidAgentVar"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            instance.getVariableInt("array_var")
        assert e.value.type() == "InvalidAgentVar"
