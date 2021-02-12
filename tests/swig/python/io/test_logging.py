import pytest
from unittest import TestCase
from pyflamegpu import *
import os

MODEL_NAME = "Model";
AGENT_NAME1 = "Agent1";
FUNCTION_NAME1 = "Func1";
HOST_FUNCTION_NAME1 = "Func2";


class step_fn1(pyflamegpu.HostFunctionCallback):
    def run(self, FLAMEGPU):
        # increment all properties
        FLAMEGPU.environment.setPropertyFloat("float_prop", FLAMEGPU.environment.getPropertyFloat("float_prop") + 1.0);
        FLAMEGPU.environment.setPropertyInt("int_prop", FLAMEGPU.environment.getPropertyInt("int_prop") + 1);
        FLAMEGPU.environment.setPropertyUInt("uint_prop", FLAMEGPU.environment.getPropertyUInt("uint_prop") + 1);

        a = FLAMEGPU.environment.getPropertyArrayFloat("float_prop_array");
        b = FLAMEGPU.environment.getPropertyArrayInt("int_prop_array");
        c = FLAMEGPU.environment.getPropertyArrayUInt("uint_prop_array");
        FLAMEGPU.environment.setPropertyArrayFloat("float_prop_array", [a[0] + 1.0, a[1] + 1.0]);
        FLAMEGPU.environment.setPropertyArrayInt("int_prop_array", [b[0] + 1, b[1] + 1, b[2] + 1]);
        FLAMEGPU.environment.setPropertyArrayUInt("uint_prop_array", [c[0] + 1, c[1] + 1, c[2] + 1, c[3] + 1]);
    
class logging_ensemble_init(pyflamegpu.HostFunctionCallback):
    def run(self, FLAMEGPU):
        instance_id  = FLAMEGPU.environment.getPropertyInt("instance_id");
        for i in range(instance_id, instance_id + 101):
            instance = FLAMEGPU.newAgent(AGENT_NAME1);
            instance.setVariableFloat("float_var", i);
            instance.setVariableInt("int_var", i+1);
            instance.setVariableUInt("uint_var", i+2);

class LoggingTest(TestCase):

    agent_fn1 = """
        FLAMEGPU_AGENT_FUNCTION(agent_fn1, MsgNone, MsgNone) {
            // increment all variables
            FLAMEGPU->setVariable<float>("float_var", FLAMEGPU->getVariable<float>("float_var") + 1.0f);
            FLAMEGPU->setVariable<int>("int_var", FLAMEGPU->getVariable<int>("int_var") + 1);
            FLAMEGPU->setVariable<unsigned int>("uint_var", FLAMEGPU->getVariable<unsigned int>("uint_var") + 1);
            return ALIVE;
        }
        """

    def logAllAgent(self, alcfg, var_name, type):
        fn = getattr(alcfg, f"logMin{type}");
        fn(var_name);
        fn = getattr(alcfg, f"logMax{type}");
        fn(var_name);
        fn = getattr(alcfg, f"logMean{type}");
        fn(var_name);
        fn = getattr(alcfg, f"logStandardDev{type}");
        fn(var_name);
        fn = getattr(alcfg, f"logSum{type}");
        fn(var_name);

    def test_CUDASimulationStep(self):
        """
           Ensure the expected data is logged when CUDASimulation::step() is called
           Note: does not check files logged to disk
        """
        # Define model
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME1)
        a.newVariableFloat("float_var");
        a.newVariableInt("int_var");
        a.newVariableUInt("uint_var");
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1);
        m.newLayer().addAgentFunction(f1);
        sf1 = step_fn1();
        m.addStepFunctionCallback(sf1);
        m.Environment().newPropertyFloat("float_prop", 1.0);
        m.Environment().newPropertyInt("int_prop", 1);
        m.Environment().newPropertyUInt("uint_prop", 1);
        m.Environment().newPropertyArrayFloat("float_prop_array", 2, [1.0, 2.0]);
        m.Environment().newPropertyArrayInt("int_prop_array", 3, [2, 3, 4]);
        m.Environment().newPropertyArrayUInt("uint_prop_array", 4, [3, 4, 5, 6]);

        # Define logging configs
        lcfg = pyflamegpu.LoggingConfig(m);
        alcfg = lcfg.agent(AGENT_NAME1);
        alcfg.logCount();
        self.logAllAgent(alcfg, "float_var", "Float");
        self.logAllAgent(alcfg, "int_var", "Int");
        self.logAllAgent(alcfg, "uint_var", "UInt");
        lcfg.logEnvironment("float_prop");
        lcfg.logEnvironment("int_prop");
        lcfg.logEnvironment("uint_prop");
        lcfg.logEnvironment("float_prop_array");
        lcfg.logEnvironment("int_prop_array");
        lcfg.logEnvironment("uint_prop_array");

        slcfg = pyflamegpu.StepLoggingConfig(lcfg);
        slcfg.setFrequency(2);

        # Create agent population
        pop = pyflamegpu.AgentVector(a, 101);
        for i in range(101):
            instance = pop[i];
            instance.setVariableFloat("float_var", i);
            instance.setVariableInt("int_var", i+1);
            instance.setVariableUInt("uint_var", i+2);


        # Run model
        sim = pyflamegpu.CUDASimulation(m);
        sim.SimulationConfig().steps = 10;
        sim.setStepLog(slcfg);
        sim.setExitLog(lcfg);
        sim.setPopulationData(pop);

        # Step log 5 individual times, and check step logs match expectations
        for i in range(1, 11):
            sim.step();
            log = sim.getRunLog();
            steps = log.getStepLog();
            # Step log frequency works as intended
            assert steps.size() == int(i/2)  
        
        log = sim.getRunLog();
        steps = log.getStepLog();
        step_index = 2;
        for step in steps:
            assert step.getStepCount() == step_index
            # Agent step logging works
            # assert step.getAgents().size() == 1 # Not supported in Py, getAgents() returns a complex map of custom types
            agent_log = step.getAgent(AGENT_NAME1);
            assert 101 == agent_log.getCount()

            assert 100.0 + step_index == agent_log.getMaxFloat("float_var")
            assert 101 + step_index == agent_log.getMaxInt("int_var")
            assert 102.0 + step_index == agent_log.getMaxUInt("uint_var")

            assert 0.0 + step_index == agent_log.getMinFloat("float_var")
            assert 1 + step_index == agent_log.getMinInt("int_var")
            assert 2.0 + step_index == agent_log.getMinUInt("uint_var")

            assert 50.0 + step_index == agent_log.getMean("float_var")
            assert 51.0 + step_index == agent_log.getMean("int_var")
            assert 52.0 + step_index == agent_log.getMean("uint_var")

            # Test value calculated with excel
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("float_var")
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("int_var")
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("uint_var")

            assert (50.0 + step_index) * 101 == agent_log.getSumFloat("float_var")
            assert (51 + step_index) * 101 == agent_log.getSumInt("int_var")
            assert (52 + step_index) * 101 == agent_log.getSumUInt("uint_var")

            # Env step logging works
            assert step.getEnvironmentPropertyFloat("float_prop") == 1.0 + step_index
            assert step.getEnvironmentPropertyInt("int_prop") == 1 + step_index
            assert step.getEnvironmentPropertyUInt("uint_prop") == 1 + step_index

            f_a = step.getEnvironmentPropertyArrayFloat("float_prop_array");
            assert f_a[0] == 1.0 + step_index
            assert f_a[1] == 2.0 + step_index
            i_a = step.getEnvironmentPropertyArrayInt("int_prop_array");
            assert i_a[0] == 2 + step_index
            assert i_a[1] == 3 + step_index
            assert i_a[2] == 4 + step_index
            u_a = step.getEnvironmentPropertyArrayUInt("uint_prop_array");
            assert u_a[0] == 3 + step_index
            assert u_a[1] == 4 + step_index
            assert u_a[2] == 5 + step_index
            assert u_a[3] == 6 + step_index

            step_index+=2;   
    
    def test_CUDASimulationSimulate(self):
        """
           Ensure the expected data is logged when CUDASimulation::simulate() is called
           Note: does not check files logged to disk
        """
        # Define model
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME1)
        a.newVariableFloat("float_var");
        a.newVariableInt("int_var");
        a.newVariableUInt("uint_var");
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1);
        m.newLayer().addAgentFunction(f1);
        sf1 = step_fn1();
        m.addStepFunctionCallback(sf1);
        m.Environment().newPropertyFloat("float_prop", 1.0);
        m.Environment().newPropertyInt("int_prop", 1);
        m.Environment().newPropertyUInt("uint_prop", 1);
        m.Environment().newPropertyArrayFloat("float_prop_array", 2, [1.0, 2.0]);
        m.Environment().newPropertyArrayInt("int_prop_array", 3, [2, 3, 4]);
        m.Environment().newPropertyArrayUInt("uint_prop_array", 4, [3, 4, 5, 6]);

        # Define logging configs
        lcfg = pyflamegpu.LoggingConfig(m);
        alcfg = lcfg.agent(AGENT_NAME1);
        alcfg.logCount();
        self.logAllAgent(alcfg, "float_var", "Float");
        self.logAllAgent(alcfg, "int_var", "Int");
        self.logAllAgent(alcfg, "uint_var", "UInt");
        lcfg.logEnvironment("float_prop");
        lcfg.logEnvironment("int_prop");
        lcfg.logEnvironment("uint_prop");
        lcfg.logEnvironment("float_prop_array");
        lcfg.logEnvironment("int_prop_array");
        lcfg.logEnvironment("uint_prop_array");

        slcfg = pyflamegpu.StepLoggingConfig(lcfg);
        slcfg.setFrequency(2);

        # Create agent population
        pop = pyflamegpu.AgentVector(a, 101);
        for i in range(101):
            instance = pop[i];
            instance.setVariableFloat("float_var", i);
            instance.setVariableInt("int_var", i+1);
            instance.setVariableUInt("uint_var", i+2);

        # Run model
        sim = pyflamegpu.CUDASimulation(m);
        # sim.SimulationConfig().common_log_file = "commmon.json";
        # sim.SimulationConfig().step_log_file = "step.json";
        # sim.SimulationConfig().exit_log_file = "exit.json";
        sim.SimulationConfig().steps = 10;
        sim.setStepLog(slcfg);
        sim.setExitLog(lcfg);
        sim.setPopulationData(pop);
        
        # Call simulate(), and check step and exit logs match expectations
        sim.simulate();
        
        #Check step log
        log = sim.getRunLog();
        steps = log.getStepLog();
        # init log, + 5 logs from 10 steps
        assert steps.size() == 6
        step_index = 0;
        for step in steps:
            assert step.getStepCount() == step_index
            # Agent step logging works
            # assert step.getAgents().size() == 1 # Not supported in Py, getAgents() returns a complex map of custom types
            agent_log = step.getAgent(AGENT_NAME1);
            assert 101 == agent_log.getCount()
            
            assert 100.0 + step_index == agent_log.getMaxFloat("float_var")
            assert 101 + step_index == agent_log.getMaxInt("int_var")
            assert 102.0 + step_index == agent_log.getMaxUInt("uint_var")

            assert 0.0 + step_index == agent_log.getMinFloat("float_var")
            assert 1 + step_index == agent_log.getMinInt("int_var")
            assert 2.0 + step_index == agent_log.getMinUInt("uint_var")

            assert 50.0 + step_index == agent_log.getMean("float_var")
            assert 51.0 + step_index == agent_log.getMean("int_var")
            assert 52.0 + step_index == agent_log.getMean("uint_var")

            # Test value calculated with excel
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("float_var")
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("int_var")
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("uint_var")

            assert (50.0 + step_index) * 101 == agent_log.getSumFloat("float_var")
            assert (51 + step_index) * 101 == agent_log.getSumInt("int_var")
            assert (52 + step_index) * 101 == agent_log.getSumUInt("uint_var")

            # Env step logging works
            assert step.getEnvironmentPropertyFloat("float_prop") == 1.0 + step_index
            assert step.getEnvironmentPropertyInt("int_prop") == 1 + step_index
            assert step.getEnvironmentPropertyUInt("uint_prop") == 1 + step_index

            f_a = step.getEnvironmentPropertyArrayFloat("float_prop_array");
            assert f_a[0] == 1.0 + step_index
            assert f_a[1] == 2.0 + step_index
            i_a = step.getEnvironmentPropertyArrayInt("int_prop_array");
            assert i_a[0] == 2 + step_index
            assert i_a[1] == 3 + step_index
            assert i_a[2] == 4 + step_index
            u_a = step.getEnvironmentPropertyArrayUInt("uint_prop_array");
            assert u_a[0] == 3 + step_index
            assert u_a[1] == 4 + step_index
            assert u_a[2] == 5 + step_index
            assert u_a[3] == 6 + step_index

            step_index+=2;

      
        # Check exit log, should match final step log
        step_index = 10;
        exit = log.getExitLog();
        assert exit.getStepCount() == step_index
        # Agent step logging works
        # assert exit.getAgents().size() == 1 # Not supported in Py, getAgents() returns a complex map of custom types
        agent_log = exit.getAgent(AGENT_NAME1);
        assert 101 == agent_log.getCount()

        assert 100.0 + step_index == agent_log.getMaxFloat("float_var")
        assert 101 + step_index == agent_log.getMaxInt("int_var")
        assert 102.0 + step_index == agent_log.getMaxUInt("uint_var")

        assert 0.0 + step_index == agent_log.getMinFloat("float_var")
        assert 1 + step_index == agent_log.getMinInt("int_var")
        assert 2.0 + step_index == agent_log.getMinUInt("uint_var")

        assert 50.0 + step_index == agent_log.getMean("float_var")
        assert 51.0 + step_index == agent_log.getMean("int_var")
        assert 52.0 + step_index == agent_log.getMean("uint_var")

        # Test value calculated with excel
        assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("float_var")
        assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("int_var")
        assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("uint_var")

        assert (50.0 + step_index) * 101 == agent_log.getSumFloat("float_var")
        assert (51 + step_index) * 101 == agent_log.getSumInt("int_var")
        assert (52 + step_index) * 101 == agent_log.getSumUInt("uint_var")

        # Env step logging works
        assert step.getEnvironmentPropertyFloat("float_prop") == 1.0 + step_index
        assert step.getEnvironmentPropertyInt("int_prop") == 1 + step_index
        assert step.getEnvironmentPropertyUInt("uint_prop") == 1 + step_index

        f_a = step.getEnvironmentPropertyArrayFloat("float_prop_array");
        assert f_a[0] == 1.0 + step_index
        assert f_a[1] == 2.0 + step_index
        i_a = step.getEnvironmentPropertyArrayInt("int_prop_array");
        assert i_a[0] == 2 + step_index
        assert i_a[1] == 3 + step_index
        assert i_a[2] == 4 + step_index
        u_a = step.getEnvironmentPropertyArrayUInt("uint_prop_array");
        assert u_a[0] == 3 + step_index
        assert u_a[1] == 4 + step_index
        assert u_a[2] == 5 + step_index
        assert u_a[3] == 6 + step_index    

    def test_CUDAEnsembleSimulate(self):
        """
           Ensure the expected data is logged when CUDAEnsemble::simulate() is called
           Note: does not check files logged to disk
        """
        # Define model
        m = pyflamegpu.ModelDescription(MODEL_NAME)
        a = m.newAgent(AGENT_NAME1)
        a.newVariableFloat("float_var");
        a.newVariableInt("int_var");
        a.newVariableUInt("uint_var");
        f1 = a.newRTCFunction(FUNCTION_NAME1, self.agent_fn1);
        if1 = logging_ensemble_init();
        m.addInitFunctionCallback(if1);  # This causes an access violation?
        m.newLayer().addAgentFunction(f1);
        sf1 = step_fn1();
        m.addStepFunctionCallback(sf1);  # This causes an access violation?
        m.Environment().newPropertyInt("instance_id", 0); # This will act as the modifier for ensemble instances, only impacting the init fn
        m.Environment().newPropertyFloat("float_prop", 1.0);
        m.Environment().newPropertyInt("int_prop", 1);
        m.Environment().newPropertyUInt("uint_prop", 1);
        m.Environment().newPropertyArrayFloat("float_prop_array", 2, [1.0, 2.0]);
        m.Environment().newPropertyArrayInt("int_prop_array", 3, [2, 3, 4]);
        m.Environment().newPropertyArrayUInt("uint_prop_array", 4, [3, 4, 5, 6]);

        # Define logging configs
        lcfg = pyflamegpu.LoggingConfig(m);
        alcfg = lcfg.agent(AGENT_NAME1);
        alcfg.logCount();
        self.logAllAgent(alcfg, "float_var", "Float");
        self.logAllAgent(alcfg, "int_var", "Int");
        self.logAllAgent(alcfg, "uint_var", "UInt");
        lcfg.logEnvironment("float_prop");
        lcfg.logEnvironment("instance_id");
        lcfg.logEnvironment("int_prop");
        lcfg.logEnvironment("uint_prop");
        lcfg.logEnvironment("float_prop_array");
        lcfg.logEnvironment("int_prop_array");
        lcfg.logEnvironment("uint_prop_array");

        slcfg = pyflamegpu.StepLoggingConfig(lcfg);
        slcfg.setFrequency(2);

        # Set up the runplan
        plan = pyflamegpu.RunPlanVec(m, 10);
        i_id = 0;
        for p in plan:
            p.setSteps(10);
            p.setPropertyInt("instance_id", i_id);
            i_id += 1
            # p.setOutputSubdirectory(i_id%2 == 0 ? "a" : "b");
    

        # Run model
        sim = pyflamegpu.CUDAEnsemble(m);
        sim.Config().concurrent_runs = 5;
        sim.Config().silent = True;
        sim.Config().timing = False;
        # sim.Config().out_directory = "ensemble_out";
        # sim.Config().out_format = "json";
        sim.setStepLog(slcfg);
        sim.setExitLog(lcfg);
        # Call simulate(), and check step and exit logs match expectations
        sim.simulate(plan);
    
        # Check step log
        run_logs = sim.getLogs();
        i_id = 0;
        for log in run_logs:
            # Check step log
            steps = log.getStepLog();
            # init log, + 5 logs from 10 steps
            assert steps.size() == 6
            step_index = 0;
            for step in steps:
                # Log corresponds to the correct instance
                assert step.getEnvironmentPropertyInt("instance_id") == i_id
                assert step.getStepCount() == step_index
                # Agent step logging works
                # assert step.getAgents().size() == 1 # Not supported in Py, getAgents() returns a complex map of custom types
                agent_log = step.getAgent(AGENT_NAME1);
                assert 101 == agent_log.getCount()
                
                assert 100.0 + step_index + i_id == agent_log.getMaxFloat("float_var")
                assert 101 + step_index + i_id== agent_log.getMaxInt("int_var")
                assert 102.0 + step_index + i_id== agent_log.getMaxUInt("uint_var")

                assert 0.0 + step_index + i_id== agent_log.getMinFloat("float_var")
                assert 1 + step_index + i_id== agent_log.getMinInt("int_var")
                assert 2.0 + step_index + i_id== agent_log.getMinUInt("uint_var")

                assert 50.0 + step_index + i_id== agent_log.getMean("float_var")
                assert 51.0 + step_index + i_id== agent_log.getMean("int_var")
                assert 52.0 + step_index + i_id== agent_log.getMean("uint_var")

                # Test value calculated with excel
                assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("float_var")
                assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("int_var")
                assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("uint_var")

                assert (50.0 + step_index + i_id) * 101 == agent_log.getSumFloat("float_var")
                assert (51 + step_index + i_id) * 101 == agent_log.getSumInt("int_var")
                assert (52 + step_index + i_id) * 101 == agent_log.getSumUInt("uint_var")

                # Env step logging works
                assert step.getEnvironmentPropertyFloat("float_prop") == 1.0 + step_index
                assert step.getEnvironmentPropertyInt("int_prop") == 1 + step_index
                assert step.getEnvironmentPropertyUInt("uint_prop") == 1 + step_index

                f_a = step.getEnvironmentPropertyArrayFloat("float_prop_array");
                assert f_a[0] == 1.0 + step_index
                assert f_a[1] == 2.0 + step_index
                i_a = step.getEnvironmentPropertyArrayInt("int_prop_array");
                assert i_a[0] == 2 + step_index
                assert i_a[1] == 3 + step_index
                assert i_a[2] == 4 + step_index
                u_a = step.getEnvironmentPropertyArrayUInt("uint_prop_array");
                assert u_a[0] == 3 + step_index
                assert u_a[1] == 4 + step_index
                assert u_a[2] == 5 + step_index
                assert u_a[3] == 6 + step_index

                step_index+=2;
                
                
            ## Check exit log, should match final step log
            step_index = 10;
            exit = log.getExitLog();
            assert exit.getStepCount() == step_index
            # Agent step logging works
            # assert exit.getAgents().size() == 1 # Not supported in Py, getAgents() returns a complex map of custom types
            agent_log = exit.getAgent(AGENT_NAME1);
            assert 101 == agent_log.getCount()

            assert 100.0 + step_index + i_id == agent_log.getMaxFloat("float_var")
            assert 101 + step_index + i_id== agent_log.getMaxInt("int_var")
            assert 102.0 + step_index + i_id== agent_log.getMaxUInt("uint_var")

            assert 0.0 + step_index + i_id== agent_log.getMinFloat("float_var")
            assert 1 + step_index + i_id== agent_log.getMinInt("int_var")
            assert 2.0 + step_index + i_id== agent_log.getMinUInt("uint_var")

            assert 50.0 + step_index + i_id== agent_log.getMean("float_var")
            assert 51.0 + step_index + i_id== agent_log.getMean("int_var")
            assert 52.0 + step_index + i_id== agent_log.getMean("uint_var")

            # Test value calculated with excel
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("float_var")
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("int_var")
            assert pytest.approx(29.15476, 0.0001) == agent_log.getStandardDev("uint_var")

            assert (50.0 + step_index + i_id) * 101 == agent_log.getSumFloat("float_var")
            assert (51 + step_index + i_id) * 101 == agent_log.getSumInt("int_var")
            assert (52 + step_index + i_id) * 101 == agent_log.getSumUInt("uint_var")

            # Env step logging works
            assert step.getEnvironmentPropertyFloat("float_prop") == 1.0 + step_index
            assert step.getEnvironmentPropertyInt("int_prop") == 1 + step_index
            assert step.getEnvironmentPropertyUInt("uint_prop") == 1 + step_index

            f_a = step.getEnvironmentPropertyArrayFloat("float_prop_array");
            assert f_a[0] == 1.0 + step_index
            assert f_a[1] == 2.0 + step_index
            i_a = step.getEnvironmentPropertyArrayInt("int_prop_array");
            assert i_a[0] == 2 + step_index
            assert i_a[1] == 3 + step_index
            assert i_a[2] == 4 + step_index
            u_a = step.getEnvironmentPropertyArrayUInt("uint_prop_array");
            assert u_a[0] == 3 + step_index
            assert u_a[1] == 4 + step_index
            assert u_a[2] == 5 + step_index
            assert u_a[3] == 6 + step_index

            i_id+=1