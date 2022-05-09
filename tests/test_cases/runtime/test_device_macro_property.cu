/**
 * Tests of class: DeviceMacroProperty
 * WriteRead: Check that SEATBELTS catches a read after write in same agent fn
 * ReadWrite: Check that SEATBELTS catches a write after read in same agent fn
 * add: Use DeviceAPI operator+=, then read the value back in a subsequent agent function
 * add2: Use DeviceAPI operator+, and read the returned result
 * sub: Use DeviceAPI operator-=, then read the value back in a subsequent agent function
 * sub2: Use DeviceAPI operator-, and read the returned result
 * preincrement: Use DeviceAPI operator++ (pre), check the results, then read the value back in a subsequent agent function
 * predecrement: Use DeviceAPI operator-- (pre), check the results, then read the value back in a subsequent agent function
 * postincrement: Use DeviceAPI operator++ (post), check the results, then read the value back in a subsequent agent function
 * postdecrement: Use DeviceAPI operator-- (post), check the results, then read the value back in a subsequent agent function
 * min: Use DeviceAPI min with a value that succeeds, check the results, then read the value back in a subsequent agent function
 * min2: Use DeviceAPI min with a value that fails, check the results, then read the value back in a subsequent agent function
 * max: Use DeviceAPI max with a value that succeeds, check the results, then read the value back in a subsequent agent function
 * max2: Use DeviceAPI max with a value that fails, check the results, then read the value back in a subsequent agent function
 * cas: Use DeviceAPI cas with a value that succeeds, check the results, then read the value back in a subsequent agent function
 * cas2: Use DeviceAPI cas with a value that fails, check the results, then read the value back in a subsequent agent function
 * exchange_int64: Atomic exchange int64, then read the value back in a subsequent agent function
 * exchange_uint64: Atomic exchange uint64, then read the value back in a subsequent agent function
 * exchange_int32: Atomic exchange int32, then read the value back in a subsequent agent function
 * exchange_uint32: Atomic exchange uint32, then read the value back in a subsequent agent function
 * exchange_double: Atomic exchange double, then read the value back in a subsequent agent function
 * exchange_float: Atomic exchange float, then read the value back in a subsequent agent function
 */
#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace {
FLAMEGPU_AGENT_FUNCTION(WriteRead, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(DeviceMacroPropertyTest, WriteRead) {
#else
TEST(DeviceMacroPropertyTest, DISABLED_WriteRead) {
#endif
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    AgentFunctionDescription& WriteReadFn = agent.newFunction("WriteRead", WriteRead);
    model.newLayer().addAgentFunction(WriteReadFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    EXPECT_THROW(cudaSimulation.simulate(), flamegpu::exception::DeviceError);
}
FLAMEGPU_AGENT_FUNCTION(ReadWrite, flamegpu::MessageNone, flamegpu::MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int");
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return flamegpu::ALIVE;
}
#if !defined(SEATBELTS) || SEATBELTS
TEST(DeviceMacroPropertyTest, ReadWrite) {
#else
TEST(DeviceMacroPropertyTest, DISABLED_ReadWrite) {
#endif
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    AgentFunctionDescription& ReadWriteFn = agent.newFunction("ReadWrite", ReadWrite);
    model.newLayer().addAgentFunction(ReadWriteFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    EXPECT_THROW(cudaSimulation.simulate(), flamegpu::exception::DeviceError);
}

FLAMEGPU_AGENT_FUNCTION(Init_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(1);
    FLAMEGPU->environment.getMacroProperty<double>("double").exchange(1);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Write_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    FLAMEGPU->environment.getMacroProperty<double>("double") += static_cast<double>(FLAMEGPU->getVariable<unsigned int>("a"));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    FLAMEGPU->setVariable<double>("c", FLAMEGPU->environment.getMacroProperty<double>("double"));
    return flamegpu::ALIVE;
}
TEST(DeviceMacroPropertyTest, add) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    model.Environment().newMacroProperty<double>("double");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<double>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_add);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_add);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_add);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int b_out = population.at(0).getVariable<unsigned int>("b");
    EXPECT_EQ(13u, b_out);
    const double c_out = population.at(0).getVariable<double>("c");
    EXPECT_DOUBLE_EQ(13.0, c_out);
}
FLAMEGPU_AGENT_FUNCTION(Init_add2, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(1);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Write_add2, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int") + FLAMEGPU->getVariable<unsigned int>("a"));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Write_2check, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("c", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
TEST(DeviceMacroPropertyTest, add2) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_add2);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_add2);
    AgentFunctionDescription& checkFn = agent.newFunction("check", Write_2check);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(checkFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(13u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(1u, t_out2);
}
FLAMEGPU_AGENT_FUNCTION(Init_sub, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(25);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Write_sub, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") -= FLAMEGPU->getVariable<unsigned int>("a");
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_sub, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
TEST(DeviceMacroPropertyTest, sub) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_sub);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_sub);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(13u, t_out);
}
FLAMEGPU_AGENT_FUNCTION(Write_sub2, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int") - FLAMEGPU->getVariable<unsigned int>("a"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, sub2) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_sub2);
    AgentFunctionDescription& checkFn = agent.newFunction("check", Write_2check);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(checkFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(13u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(25u, t_out2);
}
FLAMEGPU_AGENT_FUNCTION(Init_postincrement, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(1);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Write_postincrement, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int")++);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_increment, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("c", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, postincrement) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_postincrement);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_postincrement);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(1u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(2u, t_out2);
}
FLAMEGPU_AGENT_FUNCTION(Init_preincrement, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(1);
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Write_preincrement, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", ++FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, preincrement) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_preincrement);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_preincrement);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(2u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(2u, t_out2);
}
FLAMEGPU_AGENT_FUNCTION(Write_postdecrement, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int")--);
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, postdecrement) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_postdecrement);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(25u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(24u, t_out2);
}
FLAMEGPU_AGENT_FUNCTION(Write_predecrement, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", --FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, predecrement) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_predecrement);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(24u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(24u, t_out2);
}

FLAMEGPU_AGENT_FUNCTION(Write_min, MessageNone, MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int").min(FLAMEGPU->getVariable<unsigned int>("a"));
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, min) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_min);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(12u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(12u, t_out2);
}
TEST(DeviceMacroPropertyTest, min2) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_min);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 45u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(25u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(25u, t_out2);
}
FLAMEGPU_AGENT_FUNCTION(Write_max, MessageNone, MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int").max(FLAMEGPU->getVariable<unsigned int>("a"));
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, max) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_max);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 45u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(45u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(45u, t_out2);
}
TEST(DeviceMacroPropertyTest, max2) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_max);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(25u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(25u, t_out2);
}
FLAMEGPU_AGENT_FUNCTION(Write_cas, MessageNone, MessageNone) {
    unsigned int t = FLAMEGPU->environment.getMacroProperty<unsigned int>("int").CAS(FLAMEGPU->getVariable<unsigned int>("a"), 100);
    FLAMEGPU->setVariable<unsigned int>("b", t);
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, cas) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_cas);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 25u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(25u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(100u, t_out2);
}
TEST(DeviceMacroPropertyTest, cas2) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    agent.newVariable<unsigned int>("c");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_sub);
    AgentFunctionDescription& writeFn = agent.newFunction("write", Write_cas);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_increment);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(25u, t_out);
    const unsigned int t_out2 = population.at(0).getVariable<unsigned int>("c");
    ASSERT_EQ(25u, t_out2);
}

FLAMEGPU_AGENT_FUNCTION(Init_int64, MessageNone, MessageNone) {
    FLAMEGPU->environment.getMacroProperty<int64_t>("int").exchange(15);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_int64, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<int64_t>("b", FLAMEGPU->environment.getMacroProperty<int64_t>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, exchange_int64) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<int64_t>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int64_t>("b");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_int64);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_int64);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const int64_t t_out = population.at(0).getVariable<int64_t>("b");
    ASSERT_EQ(15, t_out);
}
FLAMEGPU_AGENT_FUNCTION(Init_uint64, MessageNone, MessageNone) {
    FLAMEGPU->environment.getMacroProperty<uint64_t>("int").exchange(15u);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_uint64, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<uint64_t>("b", FLAMEGPU->environment.getMacroProperty<uint64_t>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, exchange_uint64) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<uint64_t>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<uint64_t>("b");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_uint64);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_uint64);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const uint64_t t_out = population.at(0).getVariable<uint64_t>("b");
    ASSERT_EQ(15u, t_out);
}
FLAMEGPU_AGENT_FUNCTION(Init_int32, MessageNone, MessageNone) {
    FLAMEGPU->environment.getMacroProperty<int32_t>("int").exchange(15);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_int32, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<int32_t>("b", FLAMEGPU->environment.getMacroProperty<int32_t>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, exchange_int32) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<int32_t>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int32_t>("b");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_int32);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_int32);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const int32_t t_out = population.at(0).getVariable<int32_t>("b");
    ASSERT_EQ(15, t_out);
}
FLAMEGPU_AGENT_FUNCTION(Init_uint32, MessageNone, MessageNone) {
    FLAMEGPU->environment.getMacroProperty<uint32_t>("int").exchange(15u);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_uint32, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<uint32_t>("b", FLAMEGPU->environment.getMacroProperty<uint32_t>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, exchange_uint32) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<uint32_t>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<uint32_t>("b");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_uint32);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_uint32);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const uint32_t t_out = population.at(0).getVariable<uint32_t>("b");
    ASSERT_EQ(15u, t_out);
}
FLAMEGPU_AGENT_FUNCTION(Init_double, MessageNone, MessageNone) {
    FLAMEGPU->environment.getMacroProperty<double>("int").exchange(15);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_double, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<double>("b", FLAMEGPU->environment.getMacroProperty<double>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, exchange_double) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<double>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<double>("b");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_double);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_double);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const double t_out = population.at(0).getVariable<double>("b");
    ASSERT_EQ(15.0, t_out);
}
FLAMEGPU_AGENT_FUNCTION(Init_float, MessageNone, MessageNone) {
    FLAMEGPU->environment.getMacroProperty<float>("int").exchange(15);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(Read_float, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<float>("b", FLAMEGPU->environment.getMacroProperty<float>("int"));
    return ALIVE;
}
TEST(DeviceMacroPropertyTest, exchange_float) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<float>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<float>("b");
    AgentFunctionDescription& initFn = agent.newFunction("init", Init_float);
    AgentFunctionDescription& readFn = agent.newFunction("read", Read_float);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const float t_out = population.at(0).getVariable<float>("b");
    ASSERT_EQ(15.0f, t_out);
}

const char* WriteRead_func = R"###(
FLAMEGPU_AGENT_FUNCTION(WriteRead, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
)###";
#if !defined(SEATBELTS) || SEATBELTS
TEST(DeviceMacroPropertyTest, RTC_WriteRead) {
#else
TEST(DeviceMacroPropertyTest, DISABLED_RTC_WriteRead) {
#endif
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    AgentFunctionDescription& WriteReadFn = agent.newRTCFunction("WriteRead", WriteRead_func);
    model.newLayer().addAgentFunction(WriteReadFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    EXPECT_THROW(cudaSimulation.simulate(), flamegpu::exception::DeviceError);
}

const char* Init_add_func = R"###(
FLAMEGPU_AGENT_FUNCTION(Init_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int").exchange(1);
    return flamegpu::ALIVE;
}
)###";
const char* Write_add_func = R"###(
FLAMEGPU_AGENT_FUNCTION(Write_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getMacroProperty<unsigned int>("int") += FLAMEGPU->getVariable<unsigned int>("a");
    return flamegpu::ALIVE;
}
)###";
const char* Read_add_func = R"###(
FLAMEGPU_AGENT_FUNCTION(Read_add, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("b", FLAMEGPU->environment.getMacroProperty<unsigned int>("int"));
    return flamegpu::ALIVE;
}
)###";

TEST(DeviceMacroPropertyTest, RTC_add) {
    ModelDescription model("device_env_test");
    // Setup environment
    model.Environment().newMacroProperty<unsigned int>("int");
    // Setup agent fn
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("a");
    agent.newVariable<unsigned int>("b");
    AgentFunctionDescription& initFn = agent.newRTCFunction("init", Init_add_func);
    AgentFunctionDescription& writeFn = agent.newRTCFunction("write", Write_add_func);
    AgentFunctionDescription& readFn = agent.newRTCFunction("read", Read_add_func);
    model.newLayer().addAgentFunction(initFn);
    model.newLayer().addAgentFunction(writeFn);
    model.newLayer().addAgentFunction(readFn);
    AgentVector population(agent, 1);
    population[0].setVariable<unsigned int>("a", 12u);
    // Do Sim
    CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 1;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const unsigned int t_out = population.at(0).getVariable<unsigned int>("b");
    ASSERT_EQ(13u, t_out);
}

}  // namespace
}  // namespace flamegpu
