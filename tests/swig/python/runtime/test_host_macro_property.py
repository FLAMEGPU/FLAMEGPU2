import pytest
from unittest import TestCase
from pyflamegpu import *

"""
   Tests of class: HostMacroProperty
   ReadTest: Test that HostMacroProperty can read data (written via agent fn's)
   WriteTest: Test that HostMacroProperty can write data (read via agent fn's)
   ZeroTest: Test that HostMacroProperty can zero data (read via agent fn's)
"""

TEST_DIMS = [2, 3, 4, 5];

class HostRead(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    t = FLAMEGPU.environment.getMacroPropertyUInt("int");
    a = 0;
    for i in t:
      for j in i:
        for k in j:
          for w in k:
            assert w == a
            a+=1;
    # Test both iteration styles
    a = 0;
    for i in range(TEST_DIMS[0]):
      for j in range(TEST_DIMS[1]):
        for k in range(TEST_DIMS[2]):
          for w in range(TEST_DIMS[3]):
            assert t[i][j][k][w] == a
            a+=1;
            
AgentWrite = """
FLAMEGPU_AGENT_FUNCTION(AgentWrite, flamegpu::MessageNone, flamegpu::MessageNone) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    const unsigned int i = FLAMEGPU->getVariable<unsigned int>("i");
    const unsigned int j = FLAMEGPU->getVariable<unsigned int>("j");
    const unsigned int k = FLAMEGPU->getVariable<unsigned int>("k");
    const unsigned int w = FLAMEGPU->getVariable<unsigned int>("w");
    t[i][j][k][w].exchange(FLAMEGPU->getVariable<unsigned int>("a"));
    return flamegpu::ALIVE;
}
"""

class HostWrite(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    t = FLAMEGPU.environment.getMacroPropertyUInt("int");
    a = 0;
    for i in t:
      for j in i:
        for k in j:
          for w in k:
            w.set(a);
            a+=1;

AgentRead = """
FLAMEGPU_AGENT_FUNCTION(AgentRead, flamegpu::MessageNone, flamegpu::MessageNone) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    const unsigned int i = FLAMEGPU->getVariable<unsigned int>("i");
    const unsigned int j = FLAMEGPU->getVariable<unsigned int>("j");
    const unsigned int k = FLAMEGPU->getVariable<unsigned int>("k");
    const unsigned int w = FLAMEGPU->getVariable<unsigned int>("w");
    if (t[i][j][k][w] == FLAMEGPU->getVariable<unsigned int>("a")) {
        FLAMEGPU->setVariable<unsigned int>("a", 1);
    } else {
        FLAMEGPU->setVariable<unsigned int>("a", 0);
    }
    return flamegpu::ALIVE;
}
"""

class HostZero(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    t = FLAMEGPU.environment.getMacroPropertyUInt("int");
    t.zero();

AgentReadZero = """
FLAMEGPU_AGENT_FUNCTION(AgentReadZero, flamegpu::MessageNone, flamegpu::MessageNone) {
    auto t = FLAMEGPU->environment.getMacroProperty<unsigned int, 2, 3, 4, 5>("int");
    const unsigned int i = FLAMEGPU->getVariable<unsigned int>("i");
    const unsigned int j = FLAMEGPU->getVariable<unsigned int>("j");
    const unsigned int k = FLAMEGPU->getVariable<unsigned int>("k");
    const unsigned int w = FLAMEGPU->getVariable<unsigned int>("w");
    if (t[i][j][k][w] == 0) {
        FLAMEGPU->setVariable<unsigned int>("a", 1);
    } else {
        FLAMEGPU->setVariable<unsigned int>("a", 0);
    }
    return flamegpu::ALIVE;
}
"""

class HostArithmeticInit(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
    FLAMEGPU.environment.getMacroPropertyInt("int")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyUInt("uint")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyInt8("int8")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyUInt8("uint8")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyInt64("int64")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyUInt64("uint64")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyID("id")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyFloat("float")[0] = 10;
    FLAMEGPU.environment.getMacroPropertyDouble("double").set(10);  # alt

class HostArithmetic(pyflamegpu.HostFunctionCallback):
  def run(self,FLAMEGPU):
        # int
        t = FLAMEGPU.environment.getMacroPropertyInt("int");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10 / 5;
        assert t / 3 == 10 / 3;
        assert t / 3.0 == 10 / 3.0;
        assert t + 5 == 10 + 5;
        assert t + 3.0 == 10 + 3.0;
        assert t - 3 == 10 - 3;
        assert t - 3.0 == 10 - 3.0;
        assert t * 5 == 10 * 5;
        assert t * 3.0 == 10 * 3.0;
        assert t % 5 == 10 % 5;
        assert t % 3 == 10 % 3;
        assert t // 5 == 10 // 5;
        assert t // 3 == 10 // 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t %= 4;
        assert t == 2;
        try:
            t /= 2; # int does no support true div, as would convert to float
        except pyflamegpu.FLAMEGPURuntimeException:
          assert True;
        else:
          assert False;
        
        # uint
        t = FLAMEGPU.environment.getMacroPropertyUInt("uint");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10 / 5;
        assert t / 3 == 10 / 3;
        assert t / 3.0 == 10 / 3.0;
        assert t + 5 == 10 + 5;
        assert t + 3.0 == 10 + 3.0;
        assert t - 3 == 10 - 3;
        assert t - 3.0 == 10 - 3.0;
        assert t * 5 == 10 * 5;
        assert t * 3.0 == 10 * 3.0;
        assert t % 5 == 10 % 5;
        assert t % 3 == 10 % 3;
        assert t // 5 == 10 // 5;
        assert t // 3 == 10 // 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t %= 4;
        assert t == 2;
        try:
            t /= 2; # int does no support true div, as would convert to float
        except pyflamegpu.FLAMEGPURuntimeException:
          assert True;
        else:
          assert False;
        
        # int8
        t = FLAMEGPU.environment.getMacroPropertyInt8("int8");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10 / 5;
        assert t / 3 == 10 / 3;
        assert t / 3.0 == 10 / 3.0;
        assert t + 5 == 10 + 5;
        assert t + 3.0 == 10 + 3.0;
        assert t - 3 == 10 - 3;
        assert t - 3.0 == 10 - 3.0;
        assert t * 5 == 10 * 5;
        assert t * 3.0 == 10 * 3.0;
        assert t % 5 == 10 % 5;
        assert t % 3 == 10 % 3;
        assert t // 5 == 10 // 5;
        assert t // 3 == 10 // 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t %= 4;
        assert t == 2;
        try:
            t /= 2; # int does no support true div, as would convert to float
        except pyflamegpu.FLAMEGPURuntimeException:
          assert True;
        else:
          assert False;
        
        # uint8
        t = FLAMEGPU.environment.getMacroPropertyUInt8("uint8");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10 / 5;
        assert t / 3 == 10 / 3;
        assert t / 3.0 == 10 / 3.0;
        assert t + 5 == 10 + 5;
        assert t + 3.0 == 10 + 3.0;
        assert t - 3 == 10 - 3;
        assert t - 3.0 == 10 - 3.0;
        assert t * 5 == 10 * 5;
        assert t * 3.0 == 10 * 3.0;
        assert t % 5 == 10 % 5;
        assert t % 3 == 10 % 3;
        assert t // 5 == 10 // 5;
        assert t // 3 == 10 // 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t %= 4;
        assert t == 2;
        try:
            t /= 2; # int does no support true div, as would convert to float
        except pyflamegpu.FLAMEGPURuntimeException:
          assert True;
        else:
          assert False;
        
        # int64
        t = FLAMEGPU.environment.getMacroPropertyInt64("int64");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10 / 5;
        assert t / 3 == 10 / 3;
        assert t / 3.0 == 10 / 3.0;
        assert t + 5 == 10 + 5;
        assert t + 3.0 == 10 + 3.0;
        assert t - 3 == 10 - 3;
        assert t - 3.0 == 10 - 3.0;
        assert t * 5 == 10 * 5;
        assert t * 3.0 == 10 * 3.0;
        assert t % 5 == 10 % 5;
        assert t % 3 == 10 % 3;
        assert t // 5 == 10 // 5;
        assert t // 3 == 10 // 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t %= 4;
        assert t == 2;
        try:
            t /= 2; # int does no support true div, as would convert to float
        except pyflamegpu.FLAMEGPURuntimeException:
          assert True;
        else:
          assert False;
        
        # uint64
        t = FLAMEGPU.environment.getMacroPropertyUInt64("uint64");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10 / 5;
        assert t / 3 == 10 / 3;
        assert t / 3.0 == 10 / 3.0;
        assert t + 5 == 10 + 5;
        assert t + 3.0 == 10 + 3.0;
        assert t - 3 == 10 - 3;
        assert t - 3.0 == 10 - 3.0;
        assert t * 5 == 10 * 5;
        assert t * 3.0 == 10 * 3.0;
        assert t % 5 == 10 % 5;
        assert t % 3 == 10 % 3;
        assert t // 5 == 10 // 5;
        assert t // 3 == 10 // 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t %= 4;
        assert t == 2;
        try:
            t /= 2; # int does no support true div, as would convert to float
        except pyflamegpu.FLAMEGPURuntimeException:
          assert True;
        else:
          assert False;
        
        # id
        t = FLAMEGPU.environment.getMacroPropertyID("id");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10 / 5;
        assert t / 3 == 10 / 3;
        assert t / 3.0 == 10 / 3.0;
        assert t + 5 == 10 + 5;
        assert t + 3.0 == 10 + 3.0;
        assert t - 3 == 10 - 3;
        assert t - 3.0 == 10 - 3.0;
        assert t * 5 == 10 * 5;
        assert t * 3.0 == 10 * 3.0;
        assert t % 5 == 10 % 5;
        assert t % 3 == 10 % 3;
        assert t // 5 == 10 // 5;
        assert t // 3 == 10 // 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t %= 4;
        assert t == 2;
        try:
            t /= 2; # int does no support true div, as would convert to float
        except pyflamegpu.FLAMEGPURuntimeException:
          assert True;
        else:
          assert False;
        
        # float
        t = FLAMEGPU.environment.getMacroPropertyFloat("float");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10.0 / 5;
        assert t / 3 == 10.0 / 3;
        assert t + 5 == 10.0 + 5;
        assert t - 3 == 10.0 - 3;
        assert t * 5 == 10.0 * 5;
        assert t // 5 == 10.0 // 5;
        assert t // 3 == 10.0 // 3;
        assert t % 5 == 10.0 % 5;
        assert t % 3 == 10.0 % 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t /= 2;
        assert t == 5;
        t.set(10);
        t %= 4.0;
        assert t == 2.0;
        
        # double
        t = FLAMEGPU.environment.getMacroPropertyDouble("double");
        # assert t++ == 10;  # Python does not support overloading increment operator
        # assert ++t == 12;  # Python does not support overloading increment operator
        # assert t-- == 12;  # Python does not support overloading increment operator
        # assert --t == 10;  # Python does not support overloading increment operator
        assert t / 5 == 10.0 / 5;
        assert t / 3 == 10.0 / 3;
        assert t + 5 == 10.0 + 5;
        assert t - 3 == 10.0 - 3;
        assert t * 5 == 10.0 * 5;
        assert t // 5 == 10.0 // 5;
        assert t // 3 == 10.0 // 3;
        assert t % 5 == 10.0 % 5;
        assert t % 3 == 10.0 % 3;
        assert t == 10;
        t += 10;
        assert t == 20;
        t -= 10;
        assert t == 10;
        t *= 2;
        assert t == 20;
        t //= 2;
        assert t == 10;
        t /= 2;
        assert t == 5;
        t.set(10);
        t %= 4.0;
        assert t == 2.0;

class HostMacroPropertyTest(TestCase):
    def test_ReadTest(self):
        # Fill MacroProperty with DeviceAPI
        # Test values match expected value with HostAPI and test in-place
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int", 2, 3, 4, 5);
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("i");
        agent.newVariableUInt("j");
        agent.newVariableUInt("k");
        agent.newVariableUInt("w");
        agent.newVariableUInt("a");
        t = agent.newRTCFunction("agentwrite", AgentWrite);
        model.newLayer().addAgentFunction(t);
        model.newLayer().addHostFunctionCallback(HostRead());
        total_agents = TEST_DIMS[0] * TEST_DIMS[1] * TEST_DIMS[2] * TEST_DIMS[3];
        population = pyflamegpu.AgentVector(agent, total_agents);
        a = 0;
        for i in range (TEST_DIMS[0]):
          for j in range (TEST_DIMS[1]):
            for k in range (TEST_DIMS[2]):
              for w in range (TEST_DIMS[3]):
                    p = population[a];
                    p.setVariableUInt("a", a);
                    p.setVariableUInt("i", i);
                    p.setVariableUInt("j", j);
                    p.setVariableUInt("k", k);
                    p.setVariableUInt("w", w);
                    a += 1;
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();

    def test_WriteTest(self):
        # Fill MacroProperty with HostAPI
        # Test values match expected value with DeviceAPI
        # Write results back to agent variable, and check at the end
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int", 2, 3, 4, 5);
        model.Environment().newMacroPropertyUInt("plusequal");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("i");
        agent.newVariableUInt("j");
        agent.newVariableUInt("k");
        agent.newVariableUInt("w");
        agent.newVariableUInt("a");
        t = agent.newRTCFunction("agentread", AgentRead);
        model.newLayer().addHostFunctionCallback(HostWrite());
        model.newLayer().addAgentFunction(t);
        total_agents = TEST_DIMS[0] * TEST_DIMS[1] * TEST_DIMS[2] * TEST_DIMS[3];
        population = pyflamegpu.AgentVector(agent, total_agents);
        a = 0;
        for i in range (TEST_DIMS[0]):
          for j in range (TEST_DIMS[1]):
            for k in range (TEST_DIMS[2]):
              for w in range (TEST_DIMS[3]):
                    p = population[a];
                    p.setVariableUInt("a", a);
                    p.setVariableUInt("i", i);
                    p.setVariableUInt("j", j);
                    p.setVariableUInt("k", k);
                    p.setVariableUInt("w", w);
                    a += 1;
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        # Check results
        correct = 0;
        for p in population:
            correct += 1 if p.getVariableUInt("a") == 1 else 0;
        
        assert correct == total_agents;

    def test_ZeroTest(self):
        # Fill MacroProperty with HostAPI
        # Test values match expected value with DeviceAPI
        # Write results back to agent variable, and check at the end
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int", 2, 3, 4, 5);
        model.Environment().newMacroPropertyUInt("plusequal");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("i");
        agent.newVariableUInt("j");
        agent.newVariableUInt("k");
        agent.newVariableUInt("w");
        agent.newVariableUInt("a");
        t1 = agent.newRTCFunction("agentwrite", AgentWrite);
        t2 = agent.newRTCFunction("agentread", AgentReadZero);
        model.newLayer().addAgentFunction(t1);
        model.newLayer().addHostFunctionCallback(HostZero());
        model.newLayer().addAgentFunction(t2);
        total_agents = TEST_DIMS[0] * TEST_DIMS[1] * TEST_DIMS[2] * TEST_DIMS[3];
        population = pyflamegpu.AgentVector(agent, total_agents);
        a = 0;
        for i in range (TEST_DIMS[0]):
          for j in range (TEST_DIMS[1]):
            for k in range (TEST_DIMS[2]):
              for w in range (TEST_DIMS[3]):
                    p = population[a];
                    p.setVariableUInt("a", a);
                    p.setVariableUInt("i", i);
                    p.setVariableUInt("j", j);
                    p.setVariableUInt("k", k);
                    p.setVariableUInt("w", w);
                    a += 1;
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        # Check results
        correct = 0;
        for p in population:
            correct += 1 if p.getVariableUInt("a") == 1 else 0;

        assert correct == total_agents;

    def test_ArithmeticTest(self):
        # Create single macro property for each type
        # Fill MacroProperties with HostAPI to a known value
        # Use all airthmetic ops, and test values match expected value with DeviceAPI
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyInt("int");
        model.Environment().newMacroPropertyUInt("uint");
        model.Environment().newMacroPropertyInt8("int8");
        model.Environment().newMacroPropertyUInt8("uint8");
        model.Environment().newMacroPropertyInt64("int64");
        model.Environment().newMacroPropertyUInt64("uint64");
        model.Environment().newMacroPropertyID("id");
        model.Environment().newMacroPropertyFloat("float");
        model.Environment().newMacroPropertyDouble("double");
        # Setup agent fn
        model.newAgent("agent");
        model.newLayer().addHostFunctionCallback(HostArithmeticInit());
        model.newLayer().addHostFunctionCallback(HostArithmetic());
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.simulate();
        