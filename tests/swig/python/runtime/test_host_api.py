import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand


TOTAL_STEPS = 4


class init_testGetStepCounter(pyflamegpu.HostFunctionCallback):
    # Init should always be 0th iteration/step
    def __init__(self):
        super().__init__()
        self.step_counter = -1

    def run(self, FLAMEGPU):
        self.step_counter = FLAMEGPU.getStepCounter()
        
    def apply_assertions(self):
        assert self.step_counter == 0


class host_testGetStepCounter(pyflamegpu.HostFunctionCallback):
    # host is during, so 0? - @todo dynamic
    def __init__(self):
        super().__init__()
        self.step_counter = -1

    def run(self, FLAMEGPU):
        self.step_counter = FLAMEGPU.getStepCounter()
        
    def apply_assertions(self, expected_step_counter):
        assert self.step_counter == expected_step_counter


class step_testGetStepCounter(pyflamegpu.HostFunctionCallback):
    # Step functions are at the end of the step
    def __init__(self):
        super().__init__()
        self.step_counter = -1

    def run(self, FLAMEGPU):
        self.step_counter = FLAMEGPU.getStepCounter()
        
    def apply_assertions(self, expected_step_counter):
        assert self.step_counter == expected_step_counter

class exit_testGetStepCounter(pyflamegpu.HostFunctionCallback):
    # Runs between steps - i.e. after step functions
    def __init__(self):
        super().__init__()
        self.step_counter = -1

    def run(self, FLAMEGPU):
        self.step_counter = FLAMEGPU.getStepCounter()
        
    def apply_assertions(self, expected_step_counter):
        assert self.step_counter == expected_step_counter


class HostAPITest(TestCase):

    def test_get_step_counter(self): 
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("agent")

        init = init_testGetStepCounter()
        model.addInitFunctionCallback(init)
        host = host_testGetStepCounter()
        model.newLayer().addHostFunctionCallback(host)
        step = step_testGetStepCounter()
        model.addStepFunctionCallback(step)
        exit = exit_testGetStepCounter()
        model.addExitFunctionCallback(exit)

        # Init pop
        agentCount = 1
        init_population = pyflamegpu.AgentPopulation(agent, agentCount)
        for i in range(agentCount):
            instance = init_population.getNextInstance("default")
        
        # Setup Model
        cuda_model = pyflamegpu.CUDAAgentModel(model)
        cuda_model.setPopulationData(init_population)
        cuda_model.SimulationConfig().steps = TOTAL_STEPS
        cuda_model.simulate()
        
        # assert
        init.apply_assertions()
        # getStepCount will return the number of completed steps so will be one less than total at end
        host.apply_assertions(TOTAL_STEPS-1)
        step.apply_assertions(TOTAL_STEPS-1)
        exit.apply_assertions(TOTAL_STEPS)
