#include "gtest/gtest.h"

#include "flamegpu/flamegpu.h"

namespace flamegpu {

// These tests wont work if built with SEATBELTS=OFF, so mark them all as disabled instead
#if defined(SEATBELTS) && !SEATBELTS
#undef TEST
#define TEST(test_suite_name, test_name) GTEST_TEST(test_suite_name, DISABLED_ ## test_name)
#endif

namespace test_rtc_device_exception {
const unsigned int AGENT_COUNT = 64;

/**
 * Test that exceptions on getVariable() work
 */
const char* rtc_dthrow_agent_func_getAgentVar = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->getVariable<int>("nope");
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getAgentVar_name) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getAgentVar);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    bool did_except = false;
    // Special case, catch the exception and test it's string
    try {
        cudaSimulation.simulate();
    } catch (exception::DeviceError &err) {
        did_except = true;
        const std::string s1 = err.what();
        EXPECT_TRUE(s1.find("nope") != std::string::npos);
    }
    // The appropriate exception was thrown?
    ASSERT_TRUE(did_except);
}
const char* rtc_dthrow_agent_func_getAgentVarType = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->getVariable<double>("id");
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getAgentVar_typesize) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getAgentVarType);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
/**
 * Test that exceptions on getArrayVariable() works
 */
const char* rtc_dthrow_agent_func_getAgentArrayVar = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->getVariable<int, 2>("nope", 0);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getAgentArrayVar_name) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getAgentArrayVar);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    bool did_except = false;
    // Special case, catch the exception and test it's string
    try {
        cudaSimulation.simulate();
    } catch (exception::DeviceError &err) {
        did_except = true;
        const std::string s1 = err.what();
        EXPECT_TRUE(s1.find("nope") != std::string::npos);
    }
    // The appropriate exception was thrown?
    ASSERT_TRUE(did_except);
}
const char* rtc_dthrow_agent_func_getAgentArrayVar1 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->getVariable<double, 2>("id", 0);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getAgentArrayVar_typesize) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getAgentArrayVar1);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
const char* rtc_dthrow_agent_func_getAgentArrayVar2 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->getVariable<int, 3>("id", 0);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getAgentArrayVar_length) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getAgentArrayVar2);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
const char* rtc_dthrow_agent_func_getAgentArrayVar3 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->getVariable<int, 2>("id", 2);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getAgentArrayVar_bounds) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getAgentArrayVar3);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
/**
 * Test that exceptions on setVariable() works
 */
const char* rtc_dthrow_agent_func_setAgentVar = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("nope", 12);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, setAgentVar_name) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_setAgentVar);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    bool did_except = false;
    // Special case, catch the exception and test it's string
    try {
        cudaSimulation.simulate();
    } catch (exception::DeviceError &err) {
        did_except = true;
        const std::string s1 = err.what();
        EXPECT_TRUE(s1.find("nope") != std::string::npos);
    }
    // The appropriate exception was thrown?
    ASSERT_TRUE(did_except);
}
const char* rtc_dthrow_agent_func_setAgentVar2 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<double>("id", 12);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, setAgentVar_typesize) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_setAgentVar2);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
/**
 * Test that exceptions on setVariable() work
 */
const char* rtc_dthrow_agent_func_setAgentArrayVar = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int, 2>("nope", 0, 12);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, setAgentArrayVar_name) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_setAgentArrayVar);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    bool did_except = false;
    // Special case, catch the exception and test it's string
    try {
        cudaSimulation.simulate();
    } catch (exception::DeviceError &err) {
        did_except = true;
        const std::string s1 = err.what();
        EXPECT_TRUE(s1.find("nope") != std::string::npos);
    }
    // The appropriate exception was thrown?
    ASSERT_TRUE(did_except);
}
const char* rtc_dthrow_agent_func_setAgentArrayVar1 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<double, 2>("id", 0, 12);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, setAgentArrayVar_typesize) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_setAgentArrayVar1);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
const char* rtc_dthrow_agent_func_setAgentArrayVar2 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int, 3>("id", 0, 12);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, setAgentArrayVar_length) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_setAgentArrayVar2);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
const char* rtc_dthrow_agent_func_setAgentArrayVar3 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int, 2>("id", 2, 12);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, setAgentArrayVar_bounds) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int, 2>("id");
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_setAgentArrayVar3);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
/**
 * Test that exceptions on environment.getProperty() work
 */
const char* rtc_dthrow_agent_func_getEnvironmentProp = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getProperty<int>("nope");
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getEnvironmentProp_name) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    model.Environment().newProperty<int>("test", 12);
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getEnvironmentProp);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    bool did_except = false;
    // Special case, catch the exception and test it's string
    try {
        cudaSimulation.simulate();
    } catch (exception::DeviceError &err) {
        did_except = true;
        const std::string s1 = err.what();
        EXPECT_TRUE(s1.find("nope") != std::string::npos);
    }
    // The appropriate exception was thrown?
    ASSERT_TRUE(did_except);
}
const char* rtc_dthrow_agent_func_getEnvironmentProp1 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getProperty<double>("test");
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getEnvironmentProp_typesize) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    model.Environment().newProperty<int>("test", 12);
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getEnvironmentProp1);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
/**
 * Test that exceptions on environment.getArrayProperty() work
 */
const char* rtc_dthrow_agent_func_getEnvironmentArrayProp = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getProperty<int>("nope", 0);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getEnvironmentArrayProp_name) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    model.Environment().newProperty<int>("test", 12);
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getEnvironmentArrayProp);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    bool did_except = false;
    // Special case, catch the exception and test it's string
    try {
        cudaSimulation.simulate();
    } catch (exception::DeviceError &err) {
        did_except = true;
        const std::string s1 = err.what();
        EXPECT_TRUE(s1.find("nope") != std::string::npos);
    }
    // The appropriate exception was thrown?
    ASSERT_TRUE(did_except);
}
const char* rtc_dthrow_agent_func_getEnvironmentArrayProp1 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getProperty<double>("test", 0);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getEnvironmentArrayProp_typesize) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    model.Environment().newProperty<int, 2>("test", {11, 12});
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getEnvironmentArrayProp1);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}
// Device environment does not currently require user to specify length of array
// const char* rtc_dthrow_agent_func_getEnvironmentArrayProp2 = R"###(
// FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
//     FLAMEGPU->environment.getProperty<int>("test", 0);
//     return flamegpu::ALIVE;
// }
// )###";
// TEST(RTCDeviceExceptionTest, getEnvironmentArrayProp_length) {
//     ModelDescription model("model");
//     AgentDescription& agent = model.newAgent("agent_name");
//     agent.newVariable<int>("id");
//     model.Environment().newProperty<int, 2>("test", {11, 12});
//     // add RTC agent function
//     AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getEnvironmentArrayProp2);
//     model.newLayer().addAgentFunction(func);
//     // Init pop
//     AgentVector init_population(agent, AGENT_COUNT);
//     for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
//         AgentVector::Agent instance = init_population[i];
//         instance.setVariable<int>("id", i);
//     }
//     // Setup Model
//     CUDASimulation cudaSimulation(model);
//     cudaSimulation.setPopulationData(init_population);
//     // Run 1 step to ensure data is pushed to device
//     EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
// }
const char* rtc_dthrow_agent_func_getEnvironmentArrayProp3 = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->environment.getProperty<int>("test", 2);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceExceptionTest, getEnvironmentArrayProp_bounds) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent_name");
    agent.newVariable<int>("id");
    model.Environment().newProperty<int, 2>("test", {11, 12});
    // add RTC agent function
    AgentFunctionDescription& func = agent.newRTCFunction("rtc_test_func", rtc_dthrow_agent_func_getEnvironmentArrayProp3);
    model.newLayer().addAgentFunction(func);
    // Init pop
    AgentVector init_population(agent, AGENT_COUNT);
    for (int i = 0; i < static_cast<int>(AGENT_COUNT); i++) {
        AgentVector::Agent instance = init_population[i];
        instance.setVariable<int>("id", i);
    }
    // Setup Model
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(init_population);
    // Run 1 step to ensure data is pushed to device
    EXPECT_THROW(cudaSimulation.step(), exception::DeviceError);
}

}  // namespace test_rtc_device_exception
}  // namespace flamegpu
