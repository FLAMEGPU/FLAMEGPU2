"""
 Tests of class: DeviceMacroProperty
 WriteRead: Check that SEATBELTS catches a read after write in same agent fn
 ReadWrite: Check that SEATBELTS catches a write after read in same agent fn
 add: Use DeviceAPI operator+=, then read the value back in a subsequent agent function
 add2: Use DeviceAPI operator+, and read the returned result
 sub: Use DeviceAPI operator-=, then read the value back in a subsequent agent function
 sub2: Use DeviceAPI operator-, and read the returned result
 preincrement: Use DeviceAPI operator++ (pre), check the results, then read the value back in a subsequent agent function
 predecrement: Use DeviceAPI operator-- (pre), check the results, then read the value back in a subsequent agent function
 postincrement: Use DeviceAPI operator++ (post), check the results, then read the value back in a subsequent agent function
 postdecrement: Use DeviceAPI operator-- (post), check the results, then read the value back in a subsequent agent function
 min: Use DeviceAPI min with a value that succeeds, check the results, then read the value back in a subsequent agent function
 min2: Use DeviceAPI min with a value that fails, check the results, then read the value back in a subsequent agent function
 max: Use DeviceAPI max with a value that succeeds, check the results, then read the value back in a subsequent agent function
 max2: Use DeviceAPI max with a value that fails, check the results, then read the value back in a subsequent agent function
 cas: Use DeviceAPI cas with a value that succeeds, check the results, then read the value back in a subsequent agent function
 cas2: Use DeviceAPI cas with a value that fails, check the results, then read the value back in a subsequent agent function
 exchange_int64: Atomic exchange int64, then read the value back in a subsequent agent function
 exchange_uint64: Atomic exchange uint64, then read the value back in a subsequent agent function
 exchange_int32: Atomic exchange int32, then read the value back in a subsequent agent function
 exchange_uint32: Atomic exchange uint32, then read the value back in a subsequent agent function
 exchange_double: Atomic exchange double, then read the value back in a subsequent agent function
 exchange_float: Atomic exchange float, then read the value back in a subsequent agent function
"""
import pytest
from unittest import TestCase
from pyflamegpu import *

WriteRead = """
FLAMEGPU_AGENT_FUNCTION(WriteRead, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
"""
ReadWrite = """
FLAMEGPU_AGENT_FUNCTION(ReadWrite, flamegpu::MessageNone, flamegpu::MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int");
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return flamegpu::ALIVE;
}
"""
Init_add = """
FLAMEGPU_AGENT_FUNCTION(Init_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(1);
    return flamegpu::ALIVE;
}
"""
Write_add = """
FLAMEGPU_AGENT_FUNCTION(Write_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    return flamegpu::ALIVE;
}
"""
Read_add = """
FLAMEGPU_AGENT_FUNCTION(Read_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
"""
Write_add2 = """
FLAMEGPU_AGENT_FUNCTION(Write_add2, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int") + FLAMEGPU->getVariable<unsigned int>("a"));
    return flamegpu::ALIVE;
}
"""
Init_sub = """
FLAMEGPU_AGENT_FUNCTION(Init_sub, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(25);
    return flamegpu::ALIVE;
}
"""
Write_sub = """
FLAMEGPU_AGENT_FUNCTION(Write_sub, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") -= FLAMEGPU->getVariable<unsigned int>("a");
    return flamegpu::ALIVE;
}
"""
Read_sub = """
FLAMEGPU_AGENT_FUNCTION(Read_sub, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
"""
Write_postincrement = """
FLAMEGPU_AGENT_FUNCTION(Write_postincrement, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int")++);
    return flamegpu::ALIVE;
}
"""
Read_increment = """
FLAMEGPU_AGENT_FUNCTION(Read_increment, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("c", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
"""
Write_sub2 = """
FLAMEGPU_AGENT_FUNCTION(Write_sub2, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int") - FLAMEGPU->getVariable<unsigned int>("a"));
    return flamegpu::ALIVE;
}
"""
Write_preincrement = """
FLAMEGPU_AGENT_FUNCTION(Write_preincrement, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", ++FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
"""
Write_postdecrement = """
FLAMEGPU_AGENT_FUNCTION(Write_postdecrement, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int")--);
    return flamegpu::ALIVE;
}
"""
Write_predecrement = """
FLAMEGPU_AGENT_FUNCTION(Write_predecrement, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", --FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
"""
Write_min = """
FLAMEGPU_AGENT_FUNCTION(Write_min, flamegpu::MessageNone, flamegpu::MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int").min(FLAMEGPU->getVariable<unsigned int>("a"));
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return flamegpu::ALIVE;
}
"""
Write_max = """
FLAMEGPU_AGENT_FUNCTION(Write_max, flamegpu::MessageNone, flamegpu::MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int").max(FLAMEGPU->getVariable<unsigned int>("a"));
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return flamegpu::ALIVE;
}
"""
Write_cas = """
FLAMEGPU_AGENT_FUNCTION(Write_cas, flamegpu::MessageNone, flamegpu::MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int").CAS(FLAMEGPU->getVariable<unsigned int>("a"), 100);
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return flamegpu::ALIVE;
}
"""
Init_int64 = """
FLAMEGPU_AGENT_FUNCTION(Init_int64, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<int64_t>("int").exchange(15);
    return flamegpu::ALIVE;
}
"""
Read_int64 = """
FLAMEGPU_AGENT_FUNCTION(Read_int64, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int64_t>("b", FLAMEGPU->environment.getMacroProperty<int64_t>("int"));
    return flamegpu::ALIVE;
}
"""
Init_uint64 = """
FLAMEGPU_AGENT_FUNCTION(Init_uint64, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<uint64_t>("int").exchange(15u);
    return flamegpu::ALIVE;
}
"""
Read_uint64 = """
FLAMEGPU_AGENT_FUNCTION(Read_uint64, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<uint64_t>("b", FLAMEGPU->environment.getMacroProperty<uint64_t>("int"));
    return flamegpu::ALIVE;
}
"""
Init_int32 = """
FLAMEGPU_AGENT_FUNCTION(Init_int32, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<int32_t>("int").exchange(15);
    return flamegpu::ALIVE;
}
"""
Read_int32 = """
FLAMEGPU_AGENT_FUNCTION(Read_int32, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int32_t>("b", FLAMEGPU->environment.getMacroProperty<int32_t>("int"));
    return flamegpu::ALIVE;
}
"""
Init_uint32 = """
FLAMEGPU_AGENT_FUNCTION(Init_uint32, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<uint32_t>("int").exchange(15u);
    return flamegpu::ALIVE;
}
"""
Read_uint32 = """
FLAMEGPU_AGENT_FUNCTION(Read_uint32, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<uint32_t>("b", FLAMEGPU->environment.getMacroProperty<uint32_t>("int"));
    return flamegpu::ALIVE;
}
"""
Init_double = """
FLAMEGPU_AGENT_FUNCTION(Init_double, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<double>("int").exchange(15);
    return flamegpu::ALIVE;
}
"""
Read_double = """
FLAMEGPU_AGENT_FUNCTION(Read_double, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<double>("b", FLAMEGPU->environment.getMacroProperty<double>("int"));
    return flamegpu::ALIVE;
}
"""
Init_float = """
FLAMEGPU_AGENT_FUNCTION(Init_float, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<float>("int").exchange(15);
    return flamegpu::ALIVE;
}
"""
Read_float = """
FLAMEGPU_AGENT_FUNCTION(Read_float, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<float>("b", FLAMEGPU->environment.getMacroProperty<float>("int"));
    return flamegpu::ALIVE;
}
"""

class DeviceMacroPropertyTest(TestCase):
    def test_add(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        initFn = agent.newRTCFunction("init", Init_add);
        writeFn = agent.newRTCFunction("write", Write_add);
        readFn = agent.newRTCFunction("read", Read_add);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 13 == t_out;
        
    def test_add2(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        initFn = agent.newRTCFunction("init", Init_add);
        writeFn = agent.newRTCFunction("write", Write_add2);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 13 == t_out;

    def test_sub(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_sub);
        readFn = agent.newRTCFunction("read", Read_sub);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 13 == t_out;

    def test_sub2(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_sub2);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 13 == t_out;
        
    def test_postincrement(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_add);
        writeFn = agent.newRTCFunction("write", Write_postincrement);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 1 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 2 == t_out2;
        
    def test_preincrement(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_add);
        writeFn = agent.newRTCFunction("write", Write_preincrement);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 2 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 2 == t_out2;
        
    def test_postdecrement(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_postdecrement);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 25 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 24 == t_out2;
        
    def test_predecrement(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_predecrement);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 24 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 24 == t_out2;

    def test_predecrement(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_min);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 12 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 12 == t_out2;

    def test_min2(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_min);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 45);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 25 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 25 == t_out2;

    def test_max(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_max);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 45);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 45 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 45 == t_out2;

    def test_max2(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_max);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 25 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 25 == t_out2;

    def test_cas(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_cas);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 25);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 25 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 100 == t_out2;

    def test_cas2(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        agent.newVariableUInt("c");
        initFn = agent.newRTCFunction("init", Init_sub);
        writeFn = agent.newRTCFunction("write", Write_cas);
        readFn = agent.newRTCFunction("read", Read_increment);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(writeFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt("b");
        assert 25 == t_out;
        t_out2 = population.at(0).getVariableUInt("c");
        assert 25 == t_out2;

    def test_exchange_int64(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyInt64("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableInt64("b");
        initFn = agent.newRTCFunction("init", Init_int64);
        readFn = agent.newRTCFunction("read", Read_int64);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableInt64("b");
        assert 15 == t_out;

    def test_exchange_uint64(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt64("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt64("b");
        initFn = agent.newRTCFunction("init", Init_uint64);
        readFn = agent.newRTCFunction("read", Read_uint64);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt64("b");
        assert 15 == t_out;

    def test_exchange_int32(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyInt32("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableInt32("b");
        initFn = agent.newRTCFunction("init", Init_int32);
        readFn = agent.newRTCFunction("read", Read_int32);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableInt32("b");
        assert 15 == t_out;

    def test_exchange_uint32(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt32("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt32("b");
        initFn = agent.newRTCFunction("init", Init_uint32);
        readFn = agent.newRTCFunction("read", Read_uint32);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableUInt32("b");
        assert 15 == t_out;

    def test_exchange_double(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyDouble("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableDouble("b");
        initFn = agent.newRTCFunction("init", Init_double);
        readFn = agent.newRTCFunction("read", Read_double);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableDouble("b");
        assert 15.0 == t_out;

    def test_exchange_float(self):
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyFloat("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableFloat("b");
        initFn = agent.newRTCFunction("init", Init_float);
        readFn = agent.newRTCFunction("read", Read_float);
        model.newLayer().addAgentFunction(initFn);
        model.newLayer().addAgentFunction(readFn);
        population = pyflamegpu.AgentVector(agent, 1);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        cudaSimulation.simulate();
        cudaSimulation.getPopulationData(population);
        t_out = population.at(0).getVariableFloat("b");
        assert 15.0 == t_out;

    def test_WriteRead(self):
        if not pyflamegpu.SEATBELTS:
            pytest.skip("Test requires SEATBELTS to be ON")
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        WriteReadFn = agent.newRTCFunction("WriteRead", WriteRead);
        model.newLayer().addAgentFunction(WriteReadFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        try:
            cudaSimulation.simulate(); # DeviceError
        except pyflamegpu.FLAMEGPURuntimeException:
            assert True;
        else:
            assert False;

    def test_ReadWrite(self):
        if not pyflamegpu.SEATBELTS:
            pytest.skip("Test requires SEATBELTS to be ON")
        model = pyflamegpu.ModelDescription("device_env_test");
        # Setup environment
        model.Environment().newMacroPropertyUInt("int");
        # Setup agent fn
        agent = model.newAgent("agent");
        agent.newVariableUInt("a");
        agent.newVariableUInt("b");
        ReadWriteFn = agent.newRTCFunction("ReadWrite", ReadWrite);
        model.newLayer().addAgentFunction(ReadWriteFn);
        population = pyflamegpu.AgentVector(agent, 1);
        population[0].setVariableUInt("a", 12);
        # Do Sim
        cudaSimulation = pyflamegpu.CUDASimulation(model);
        cudaSimulation.SimulationConfig().steps = 1;
        cudaSimulation.setPopulationData(population);
        try:
            cudaSimulation.simulate(); # DeviceError
        except pyflamegpu.FLAMEGPURuntimeException:
            assert True;
        else:
            assert False;
