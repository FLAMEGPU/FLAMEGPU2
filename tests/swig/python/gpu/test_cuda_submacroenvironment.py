"""
  Tests of submodel functionality of class: CUDAMacroEnvironment
  ReadTest: Test that HostMacroProperty can read data (written via agent fn's)
  WriteTest: Test that HostMacroProperty can write data (read via agent fn's)
  ZeroTest: Test that HostMacroProperty can zero data (read via agent fn's)
"""
import pytest
from unittest import TestCase
from pyflamegpu import *

class ExitAlways(pyflamegpu.HostFunctionConditionCallback):
    def run(self, FLAMEGPU):
      return pyflamegpu.EXIT;
      
class Host_Write_5(pyflamegpu.HostFunctionCallback):
    def run(self, FLAMEGPU):
        print("hostwrite5")
        a = FLAMEGPU.environment.getMacroPropertyUInt("a");
        a.set(5);
        b = FLAMEGPU.environment.getMacroPropertyUInt("b");
        b[0] = 21;

class Host_Read_5(pyflamegpu.HostFunctionCallback):
    def run(self, FLAMEGPU):
        print("hostread5")
        result = FLAMEGPU.environment.getMacroPropertyUInt("a");
        assert result == 5;
        result2 = FLAMEGPU.environment.getMacroPropertyUInt("b");
        assert result2 == 21;

Agent_Write_5 = """
FLAMEGPU_AGENT_FUNCTION(Agent_Write_5, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("a").exchange(5u);
    FLAMEGPU->environment.getMacroProperty<unsigned int>("b").exchange(21u);
    return flamegpu::ALIVE;
}
"""
Agent_Read_Write_5 = """
FLAMEGPU_AGENT_FUNCTION(Agent_Read_Write_5, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("a", FLAMEGPU->environment.getMacroProperty<unsigned int>("a").exchange(0u));
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("b").exchange(0u));
    return flamegpu::ALIVE;
}
"""
class Host_Agent_Read_5(pyflamegpu.HostFunctionCallback):
    def run(self, FLAMEGPU):
        agt = FLAMEGPU.agent("test");
        pop = agt.getPopulationData();
        result = pop[0].getVariableUInt("a");
        assert result == 5;
        result2 = pop[0].getVariableUInt("b");
        assert result2 == 21;
        
class SubCUDAMacroEnvironmentTest(TestCase):
    def test_SubWriteHostMasterRead(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exit_always = ExitAlways()
        m2.addExitConditionCallback(exit_always);
        m2.Environment().newMacroPropertyUInt("a");
        m2.Environment().newMacroPropertyUInt("b");
        hw5 = Host_Write_5();
        m2.addStepFunctionCallback(hw5);
        m2.newAgent("test");
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyUInt("a");
        m.Environment().newMacroPropertyUInt("b");
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.autoMap();
        m.newLayer().addSubModel(sm);
        hr5 = Host_Read_5();
        m.newLayer().addHostFunctionCallback(hr5);
        pop = pyflamegpu.AgentVector(m.newAgent("test"), 1);

        sim = pyflamegpu.CUDASimulation(m);
        sim.SimulationConfig().steps = 1;
        sim.setPopulationData(pop);
        sim.simulate();

    def test_SubWriteAgentMasterRead(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exit_always = ExitAlways()
        m2.addExitConditionCallback(exit_always);
        m2.Environment().newMacroPropertyUInt("a");
        m2.Environment().newMacroPropertyUInt("b");
        af = m2.newAgent("test").newRTCFunction("t", Agent_Write_5);
        m2.newLayer().addAgentFunction(af);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyUInt("a");
        m.Environment().newMacroPropertyUInt("b");
        agt = m.newAgent("test");
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.autoMapMacroProperties();
        sm.bindAgent("test", "test");
        m.newLayer().addSubModel(sm);
        hr5 = Host_Read_5();
        m.newLayer().addHostFunctionCallback(hr5);
        pop = pyflamegpu.AgentVector(agt, 1);

        sim = pyflamegpu.CUDASimulation(m);
        sim.SimulationConfig().steps = 1;
        sim.setPopulationData(pop);
        sim.simulate();

    def test_MasterWriteSubReadHost(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exit_always = ExitAlways()
        m2.addExitConditionCallback(exit_always);
        m2.Environment().newMacroPropertyUInt("a");
        m2.Environment().newMacroPropertyUInt("b");
        hr5 = Host_Read_5();
        m2.addStepFunctionCallback(hr5);
        m2.newAgent("test");
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyUInt("a");
        m.Environment().newMacroPropertyUInt("b");
        agt = m.newAgent("test");
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.autoMap();
        hw5 = Host_Write_5();
        m.newLayer().addHostFunctionCallback(hw5);
        m.newLayer().addSubModel(sm);
        pop = pyflamegpu.AgentVector(agt, 1);

        sim = pyflamegpu.CUDASimulation(m);
        sim.SimulationConfig().steps = 1;
        sim.setPopulationData(pop);
        sim.simulate();

    def test_MasterWriteSubReadAgent(self):
        m2 = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        exit_always = ExitAlways()
        m2.addExitConditionCallback(exit_always);
        m2.Environment().newMacroPropertyUInt("a");
        m2.Environment().newMacroPropertyUInt("b");
        agt = m2.newAgent("test");
        agt.newVariableUInt("a");
        agt.newVariableUInt("b");
        af = agt.newRTCFunction("arw", Agent_Read_Write_5);
        m2.newLayer().addAgentFunction(af);
        har5 = Host_Agent_Read_5();
        m2.addStepFunctionCallback(har5);
        m = pyflamegpu.ModelDescription("host");
        # Define Model
        m.Environment().newMacroPropertyUInt("a");
        m.Environment().newMacroPropertyUInt("b");
        agt = m.newAgent("test");
        sm = m.newSubModel("sub", m2);
        senv = sm.SubEnvironment();
        senv.autoMap();
        sm.bindAgent("test", "test");
        hw5 = Host_Write_5();
        m.newLayer().addHostFunctionCallback(hw5);
        m.newLayer().addSubModel(sm);
        pop = pyflamegpu.AgentVector(agt, 1);

        sim = pyflamegpu.CUDASimulation(m);
        sim.SimulationConfig().steps = 1;
        sim.setPopulationData(pop);
        sim.simulate();
