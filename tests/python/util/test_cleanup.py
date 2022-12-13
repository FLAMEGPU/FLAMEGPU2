import pytest
from unittest import TestCase
from pyflamegpu import *

class initfn(pyflamegpu.HostFunction):
    # Init should always be 0th iteration/step
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
        # Generate a basic pop
        AGENT_COUNT = 10
        agent = FLAMEGPU.agent("Agent")
        for i in range(AGENT_COUNT):
            agent.newAgent()

class CleanupTest(TestCase):
    """
        Test suite for the flamegpu::util::cleanup
    """

    AliveFN = """
    FLAMEGPU_AGENT_FUNCTION(AliveFn, flamegpu::MessageNone, flamegpu::MessageNone) {
        return flamegpu::ALIVE;
    }
    """

    def test_CUDASimulation(self):
        """
        Test that cleanup when called after a simulation is executed does not break on dtor use (which may actually be GC'd...)
        """
        AGENT_COUNT = 10
        # Define a model and population
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("Agent")
        aliveDesc = agent.newRTCFunction("AliveFN", self.AliveFN)
        agent.newVariableInt("test")
        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        model.addExecutionRoot(aliveDesc)
        model.generateLayers()
        
        # Add some scoping to encourage the Simulation to be destroyed sooner rather than later?
        if True:
            # Create cuda simulation
            simulation = pyflamegpu.CUDASimulation(model)
            simulation.SimulationConfig().steps = 1
            simulation.setPopulationData(population)
            # Run the simulation
            simulation.simulate()
            # Try to encourage early dtor
            del simulation
        # Create a new simulation, but this time calling cleanup between simulate() and the dtor
        if True:
            # Create cuda simulation
            simulation = pyflamegpu.CUDASimulation(model)
            simulation.SimulationConfig().steps = 1
            simulation.setPopulationData(population)
            # Run the simulation
            simulation.simulate()
            # Call cleanup
            pyflamegpu.cleanup()
            # Try to encourage early dtor
            del simulation

    def test_CUDAEnsemble(self):
        """
        Test that cleanup when called after an ensemble is executed does not break on dtor use (which may actually be GC'd...)
        """
        PLAN_COUNT = 2

        # Define a model and population
        model = pyflamegpu.ModelDescription("model")
        agent = model.newAgent("Agent")
        aliveDesc = agent.newRTCFunction("AliveFN", self.AliveFN)
        agent.newVariableInt("test")
        init = initfn()
        model.addInitFunction(init)
        model.addExecutionRoot(aliveDesc)
        model.generateLayers()

        plans = pyflamegpu.RunPlanVector(model, PLAN_COUNT)
        for idx in range(plans.size()):
            plan = plans[idx]
            plan.setSteps(1)
        
        # Add some scoping to encourage the Ensemble to be destroyed sooner rather than later?
        if True:
            # Create and run the ensemble
            ensemble = pyflamegpu.CUDAEnsemble(model)
            ensemble.Config().verbosity = pyflamegpu.Verbosity_Default
            ensemble.simulate(plans)
            # Try to encourage early dtor
            del ensemble
        # Create a new ensemble, but this time calling cleanup between simulate() and the dtor
        if True:
             # Create and run the ensemble
            ensemble = pyflamegpu.CUDAEnsemble(model)
            ensemble.Config().verbosity = pyflamegpu.Verbosity_Default
            ensemble.simulate(plans)
            # Call cleanup
            pyflamegpu.cleanup()
            # Try to encourage early dtor
            del ensemble