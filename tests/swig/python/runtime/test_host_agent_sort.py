import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

"""
Dual agent variable sorting is not exposed in pyflamegpu and as such the tests have been removed
Note: Casting random floats to in in python causes an off by one error
"""

AGENT_COUNT = 1024

class sort_ascending_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.agent("agent").sortFloat("float", pyflamegpu.HostAgentAPI.Asc)
        
class sort_descending_float(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.agent("agent").sortFloat("float",  pyflamegpu.HostAgentAPI.Desc)
    
class sort_ascending_int(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.agent("agent").sortInt("int",  pyflamegpu.HostAgentAPI.Asc)
   
class sort_descending_int(pyflamegpu.HostFunctionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        FLAMEGPU.agent("agent").sortInt("int",  pyflamegpu.HostAgentAPI.Desc)

class HostAgentSort(TestCase):

    def test_ascending_float(self): 
        # Define model
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent")
        agent.newVariableFloat("float")
        agent.newVariableFloat("spare")
        func = sort_ascending_float()
        model.newLayer().addHostFunctionCallback(func)
        rand.seed(a=31313131)
        # Init pop
        pop = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for instance in pop:
            t = rand.uniform(1, 1000000)
            instance.setVariableFloat("float", t)
            instance.setVariableFloat("spare", t+12.0)
        
        # Setup Model
        cuda_model = pyflamegpu.CUDASimulation(model)
        cuda_model.setPopulationData(pop)
        # Execute step fn
        cuda_model.step()
        # Check results
        cuda_model.getPopulationData(pop)
        assert AGENT_COUNT == pop.size()
        prev = 1
        for instance in pop:
            f = instance.getVariableFloat("float")
            s = instance.getVariableFloat("spare")
            # Agent variables are still aligned
            assert f+12.0 == s
            # Agent variables are ordered
            assert f >= prev
            # Store prev
            prev = f
        
    def test_descending_float(self): 
        # Define model
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent")
        agent.newVariableFloat("float")
        agent.newVariableFloat("spare")
        func = sort_descending_float()
        model.newLayer().addHostFunctionCallback(func)
        rand.seed(a=31313131)

        # Init pop
        pop = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for instance in pop:
            t = rand.uniform(1, 1000000)
            instance.setVariableFloat("float", t)
            instance.setVariableFloat("spare", t+12.0)
        
        # Setup Model
        cuda_model = pyflamegpu.CUDASimulation(model)
        cuda_model.setPopulationData(pop)
        # Execute step fn
        cuda_model.step()
        # Check results
        cuda_model.getPopulationData(pop)
        assert AGENT_COUNT == pop.size()
        prev = 1000000
        for instance in pop:
            f = instance.getVariableFloat("float")
            s = instance.getVariableFloat("spare")
            # Agent variables are still aligned
            assert f+12.0 == s
            # Agent variables are ordered
            assert f <= prev
            # Store prev
            prev = f
        

    def test_ascending_int(self): 
        # Define model
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent")
        agent.newVariableInt("int")
        agent.newVariableInt("spare")
        func = sort_ascending_int()
        model.newLayer().addHostFunctionCallback(func)
        rand.seed(a=31313131)

        # Init pop
        pop = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for i in range(AGENT_COUNT): 
            instance = pop[i]
            if i == AGENT_COUNT/2 : # Ensure zero is output at least once
                t = 0 
            else:
                t = int(rand.uniform(1, 1000000)) 
            instance.setVariableInt("int", t)
            instance.setVariableInt("spare", t+12)
        
        # Setup Model
        cuda_model = pyflamegpu.CUDASimulation(model)
        cuda_model.setPopulationData(pop)
        # Execute step fn
        cuda_model.step()
        # Check results
        cuda_model.getPopulationData(pop)
        assert AGENT_COUNT == pop.size()
        prev = 0
        for instance in pop:
            f = instance.getVariableInt("int")
            s = instance.getVariableInt("spare")
            # Agent variables are still aligned
            assert s-f == 12
            # Agent variables are ordered
            assert f >= prev
            # Store prev
            prev = f   

    def test_descending_int(self): 
        # Define model
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent")
        agent.newVariableInt("int")
        agent.newVariableInt("spare")
        func = sort_descending_int()
        model.newLayer().addHostFunctionCallback(func)
        rand.seed(a=31313131)

        # Init pop
        pop = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        for instance in pop:
            t = int(rand.uniform(1, 1000000))
            instance.setVariableInt("int", t)
            instance.setVariableInt("spare", t+12)
        
        # Setup Model
        cuda_model = pyflamegpu.CUDASimulation(model)
        cuda_model.setPopulationData(pop)
        # Execute step fn
        cuda_model.step()
        # Check results
        cuda_model.getPopulationData(pop)
        assert AGENT_COUNT == pop.size()
        prev = 1000000
        for instance in pop:
            f = instance.getVariableInt("int")
            s = instance.getVariableInt("spare")
            # Agent variables are still aligned
            assert s-f == 12
            # Agent variables are ordered
            assert f <= prev
            # Store prev
            prev = f


