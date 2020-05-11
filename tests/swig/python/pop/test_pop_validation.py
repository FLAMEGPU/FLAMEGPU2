import pytest
from unittest import TestCase
from pyflamegpu import *

POPULATION_SIZE = 100
MODEL_NAME = "circles_model"

class PopTest(TestCase):

    def test_population_name_check(self): 
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        agent = model.newAgent("circle")
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")

        population = pyflamegpu.AgentPopulation(agent, POPULATION_SIZE)
        for i in range(POPULATION_SIZE): 
            instance = population.getNextInstance("default")
            instance.setVariableFloat("x", i*0.1)
        
        assert population.getAgentName() == "circle"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            population.getStateMemory("circe")
        assert e.value.type() == "InvalidStateName"

    # Instance Checks have been discarded from original C++ tests as these are covered by test_agent_instance
    def test_population_inst_var_check1(self):
        pass
    def test_population_inst_var_check2(self):
        pass
    def test_population_inst_var_check3(self):
        pass

    def test_population_size_check(self): 
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        agent = model.newAgent("circle")
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        population = pyflamegpu.AgentPopulation(agent)

        assert population.getMaximumStateListCapacity() == pyflamegpu.AgentPopulation.DEFAULT_POPULATION_SIZE


    def test_population_add_more_capacity(self) :
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        agent = model.newAgent("circle")
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        population = pyflamegpu.AgentPopulation(agent, POPULATION_SIZE)
        assert population.getMaximumStateListCapacity() == POPULATION_SIZE
        population.setStateListCapacity(POPULATION_SIZE*2)
        assert population.getMaximumStateListCapacity() == POPULATION_SIZE*2
        # Catch exception that fails on above call (can't reduce population capacity)
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            population.setStateListCapacity(POPULATION_SIZE)
        assert e.value.type() == "InvalidPopulationData"


    def test_population_overflow_capacity(self) :
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        agent = model.newAgent("circle")
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        population = pyflamegpu.AgentPopulation(agent, POPULATION_SIZE)
        assert population.getMaximumStateListCapacity() == POPULATION_SIZE
        # add POPULATION_SIZE instances (no problem)
        for i in range(POPULATION_SIZE):    
            instance = population.getNextInstance("default")
            instance.setVariableFloat("x", i*0.1)
        # getNextInstance fails if capacity is too small when for loop creates 101 agents
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            population.getNextInstance("default")
        assert e.value.type() == "InvalidMemoryCapacity"


    def test_population_check_get_instance_beyond_size(self) : 
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        agent = model.newAgent("circle")
        agent.newVariableFloat("x")
        agent.newVariableFloat("y")
        population = pyflamegpu.AgentPopulation(agent, POPULATION_SIZE)
        instance_s1 = population.getNextInstance("default")
        instance_s1.setVariableFloat("x", 0.1)
        # check that getInstanceAt should fail if index is less than size
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
           population.getInstanceAt(1, "default")
        assert e.value.type() == "InvalidMemoryCapacity"


    def test_population_data_values_multiple_states(self) : 
        model = pyflamegpu.ModelDescription(MODEL_NAME)
        agent = model.newAgent("circle")
        agent.newState("s1")
        agent.newState("s2")
        agent.newVariableInt("id")
        population = pyflamegpu.AgentPopulation(agent, POPULATION_SIZE)
        # add POPULATION_SIZE instances (no problem)
        for i in range(POPULATION_SIZE):
            instance_s1 = population.getNextInstance("s1")
            instance_s1.setVariableInt("id", i)
            instance_s2 = population.getNextInstance("s2")
            instance_s2.setVariableInt("id", i + 1000)
        # check values are correct
        for i in range(POPULATION_SIZE):
            instance_s1 = population.getInstanceAt(i, "s1")
            assert instance_s1.getVariableInt("id") == i
            instance_s2 = population.getInstanceAt(i, "s2")
            assert instance_s2.getVariableInt("id") == i + 1000

