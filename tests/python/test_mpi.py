import pytest
from unittest import TestCase
from pyflamegpu import *
import time


class model_step(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
      counter = FLAMEGPU.environment.getPropertyInt("counter")
      counter+=1
      FLAMEGPU.environment.setPropertyInt("counter", counter)
      time.sleep(0.1)  # Sleep 100ms
    
class throw_exception(pyflamegpu.HostFunction):
    def run(self, FLAMEGPU):
      counter = FLAMEGPU.environment.getPropertyInt("counter");
      init_counter = FLAMEGPU.environment.getPropertyInt("init_counter");
      if FLAMEGPU.getStepCounter() == 1 and counter == 8:
          raise Exception("Exception thrown by host fn throw_exception()");


class TestMPIEnsemble(TestCase):

    def test_mpi(self):
        # init model
        model = pyflamegpu.ModelDescription("MPITest")
        model.Environment().newPropertyInt("counter", -1)
        model.Environment().newPropertyInt("init_counter", -1)
        model.newAgent("agent")
        model.newLayer().addHostFunction(model_step())
        model.newLayer().addHostFunction(throw_exception())
        # init plans
        RUNS_PER_RANK = 10
        world_size = 2  # cant probe mpi easily, so just default to 2
        plans = pyflamegpu.RunPlanVector(model, RUNS_PER_RANK * world_size);
        plans.setSteps(10)
        plans.setPropertyLerpRangeInt("counter", 0, (RUNS_PER_RANK * world_size) - 1)
        plans.setPropertyLerpRangeInt("init_counter", 0, (RUNS_PER_RANK * world_size) - 1)
        # init exit logging config
        exit_log_cfg = pyflamegpu.LoggingConfig(model)
        exit_log_cfg.logEnvironment("counter")
        # init ensemble
        ensemble = pyflamegpu.CUDAEnsemble(model)
        ensemble.Config().concurrent_runs = 1
        ensemble.Config().error_level = pyflamegpu.CUDAEnsembleConfig.Off
        ## Can't do anything fancy like splitting GPUs
        ensemble.setExitLog(exit_log_cfg)
        # Run the ensemble
        err_count = ensemble.simulate(plans)
        ## Check that 1 error occurred at rank 0
        ## err_count == 1
        # Validate logs
        ## @note Best we can currently do is check logs of each runner have correct results
        ## @note Ideally we'd validate between nodes to ensure all runs have been completed
        logs = ensemble.getLogs()
        for index, log in logs.items():
            exit_log = log.getExitLog()
            # Get a logged environment property
            counter = exit_log.getEnvironmentPropertyInt("counter")
            assert counter == index + 10
            
        # cleanup to trigger MPI finalize
        pyflamegpu.cleanup()
