import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand


TOTAL_STEPS = 4


class init_testGetStepCounter(pyflamegpu.HostFunction):
    # Init should always be 0th iteration/step
    def __init__(self):
        super().__init__()
        self.step_counter = -1

    def run(self, FLAMEGPU):
        self.step_counter = FLAMEGPU.getStepCounter()
        
    def apply_assertions(self):
        assert self.step_counter == 0


class host_testGetStepCounter(pyflamegpu.HostFunction):
    # host is during, so 0? - @todo dynamic
    def __init__(self):
        super().__init__()
        self.step_counter = -1

    def run(self, FLAMEGPU):
        self.step_counter = FLAMEGPU.getStepCounter()
        
    def apply_assertions(self, expected_step_counter):
        assert self.step_counter == expected_step_counter


class step_testGetStepCounter(pyflamegpu.HostFunction):
    # Step functions are at the end of the step
    def __init__(self):
        super().__init__()
        self.step_counter = -1

    def run(self, FLAMEGPU):
        self.step_counter = FLAMEGPU.getStepCounter()
        
    def apply_assertions(self, expected_step_counter):
        assert self.step_counter == expected_step_counter

class exit_testGetStepCounter(pyflamegpu.HostFunction):
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
        model.addInitFunction(init)
        host = host_testGetStepCounter()
        model.newLayer().addHostFunction(host)
        step = step_testGetStepCounter()
        model.addStepFunction(step)
        exit = exit_testGetStepCounter()
        model.addExitFunction(exit)

        # Init pop
        agentCount = 1
        init_population = pyflamegpu.AgentVector(agent, agentCount)
        
        # Setup Model
        cudaSimulation = pyflamegpu.CUDASimulation(model)
        cudaSimulation.setPopulationData(init_population)
        cudaSimulation.SimulationConfig().steps = TOTAL_STEPS
        cudaSimulation.simulate()
        
        # assert
        init.apply_assertions()
        # getStepCount will return the number of completed steps so will be one less than total at end
        host.apply_assertions(TOTAL_STEPS-1)
        step.apply_assertions(TOTAL_STEPS-1)
        exit.apply_assertions(TOTAL_STEPS)
