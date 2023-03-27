/**
 * Tests of class: DeviceEnvironment
 * 
 * Tests cover:
 * > get() [per supported type, individual/array/element]
 */
#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace {

__device__ float float_out;
__device__ double double_out;
__device__ uint8_t uint8_t_out;
__device__ int8_t int8_t_out;
__device__ uint16_t uint16_t_out;
__device__ int16_t int16_t_out;
__device__ uint32_t uint32_t_out;
__device__ int32_t int32_t_out;
__device__ uint64_t uint64_t_out;
__device__ int64_t int64_t_out;

FLAMEGPU_AGENT_FUNCTION(get_float, MessageNone, MessageNone) {
    float_out = FLAMEGPU->environment.getProperty<float>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_double, MessageNone, MessageNone) {
    double_out = FLAMEGPU->environment.getProperty<double>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint8_t, MessageNone, MessageNone) {
    uint8_t_out = FLAMEGPU->environment.getProperty<uint8_t>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int8_t, MessageNone, MessageNone) {
    int8_t_out = FLAMEGPU->environment.getProperty<int8_t>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint16_t, MessageNone, MessageNone) {
    uint16_t_out = FLAMEGPU->environment.getProperty<uint16_t>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int16_t, MessageNone, MessageNone) {
    int16_t_out = FLAMEGPU->environment.getProperty<int16_t>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint32_t, MessageNone, MessageNone) {
    uint32_t_out = FLAMEGPU->environment.getProperty<uint32_t>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int32_t, MessageNone, MessageNone) {
    int32_t_out = FLAMEGPU->environment.getProperty<int32_t>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint64_t, MessageNone, MessageNone) {
    uint64_t_out = FLAMEGPU->environment.getProperty<uint64_t>("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int64_t, MessageNone, MessageNone) {
    int64_t_out = FLAMEGPU->environment.getProperty<int64_t>("a");
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(get_arrayElement_float, MessageNone, MessageNone) {
    float_out = FLAMEGPU->environment.getProperty<float, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_double, MessageNone, MessageNone) {
    double_out = FLAMEGPU->environment.getProperty<double, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint8_t, MessageNone, MessageNone) {
    uint8_t_out = FLAMEGPU->environment.getProperty<uint8_t, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int8_t, MessageNone, MessageNone) {
    int8_t_out = FLAMEGPU->environment.getProperty<int8_t, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint16_t, MessageNone, MessageNone) {
    uint16_t_out = FLAMEGPU->environment.getProperty<uint16_t, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int16_t, MessageNone, MessageNone) {
    int16_t_out = FLAMEGPU->environment.getProperty<int16_t, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint32_t, MessageNone, MessageNone) {
    uint32_t_out = FLAMEGPU->environment.getProperty<uint32_t, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int32_t, MessageNone, MessageNone) {
    int32_t_out = FLAMEGPU->environment.getProperty<int32_t, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint64_t, MessageNone, MessageNone) {
    uint64_t_out = FLAMEGPU->environment.getProperty<uint64_t, 5>("a", 1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int64_t, MessageNone, MessageNone) {
    int64_t_out = FLAMEGPU->environment.getProperty<int64_t, 5>("a", 1);
    return ALIVE;
}

class MiniSim {
 public:
    MiniSim() :
      model("model"),
      agent(model.newAgent("agent")),
      population(nullptr),
      env(model.Environment()),
      cudaSimulation(nullptr) {
        agent.newVariable<float>("x");  // Redundant
    }
    ~MiniSim() {
        delete cudaSimulation;
        delete population;
    }
    void run() {
        population = new AgentVector(agent, 1);
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        cudaSimulation = new CUDASimulation(model);
        cudaSimulation->SimulationConfig().steps = 2;
        // This fails as agentMap is empty
        cudaSimulation->setPopulationData(*population);
        ASSERT_NO_THROW(cudaSimulation->simulate());
        // The negative of this, is that cudaSimulation is inaccessible within the test!
        // So copy across population data here
        ASSERT_NO_THROW(cudaSimulation->getPopulationData(*population));
    }
    ModelDescription model;
    AgentDescription agent;
    AgentVector *population;
    EnvironmentDescription env;
    CUDASimulation *cudaSimulation;

    template <typename T>
    T Get_test() {
        // Setup environment
        T a = static_cast<T>(12.0f);
        env.newProperty<T>("a", a);
        run();
        return a;
    }
    template <typename T>
    T Get_arrayElement_test() {
        // Setup environment
        std::array<T, 5> a;
        for (int i = 0; i < 5; ++i) {
            a[i] = static_cast<T>(1);
        }
        env.newProperty<T, 5>("a", a);
        run();
        return a[1];
    }
};
/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class DeviceEnvironmentTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
        float _float_out = 0;
        double _double_out = 0;
        uint8_t _uint8_t_out = 0;
        int8_t _int8_t_out = 0;
        uint16_t _uint16_t_out = 0;
        int16_t _int16_t_out = 0;
        uint32_t _uint32_t_out = 0;
        int32_t _int32_t_out = 0;
        uint64_t _uint64_t_out = 0;
        int64_t _int64_t_out = 0;
        cudaMemcpyToSymbol(float_out, &_float_out, sizeof(float));
        cudaMemcpyToSymbol(double_out, &_double_out, sizeof(uint8_t));
        cudaMemcpyToSymbol(uint8_t_out, &_uint8_t_out, sizeof(uint8_t));
        cudaMemcpyToSymbol(int8_t_out, &_int8_t_out, sizeof(int8_t));
        cudaMemcpyToSymbol(uint16_t_out, &_uint16_t_out, sizeof(uint16_t));
        cudaMemcpyToSymbol(int16_t_out, &_int16_t_out, sizeof(int16_t));
        cudaMemcpyToSymbol(uint32_t_out, &_uint32_t_out, sizeof(uint32_t));
        cudaMemcpyToSymbol(int32_t_out, &_int32_t_out, sizeof(int32_t));
        cudaMemcpyToSymbol(uint64_t_out, &_uint64_t_out, sizeof(uint64_t));
        cudaMemcpyToSymbol(int64_t_out, &_int64_t_out, sizeof(int64_t));
    }

    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};


}  // namespace


TEST_F(DeviceEnvironmentTest, Get_float) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_float);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto float_check = ms->Get_test<float>();
    float _float_out = 0;
    cudaMemcpyFromSymbol(&_float_out, float_out, sizeof(float));
    EXPECT_EQ(float_check, _float_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_double) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_double);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto double_check = ms->Get_test<double>();
    double _double_out = 0;
    cudaMemcpyFromSymbol(&_double_out, double_out, sizeof(double));
    EXPECT_EQ(double_check, _double_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_int8_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_int8_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int8_t_check = ms->Get_test<int8_t>();
    int8_t _int8_t_out = 0;
    cudaMemcpyFromSymbol(&_int8_t_out, int8_t_out, sizeof(int8_t));
    EXPECT_EQ(int8_t_check, _int8_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_uint8_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_uint8_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint8_t_check = ms->Get_test<uint8_t>();
    uint8_t _uint8_t_out = 0;
    cudaMemcpyFromSymbol(&_uint8_t_out, uint8_t_out, sizeof(uint8_t));
    EXPECT_EQ(uint8_t_check, _uint8_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_int16_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_int16_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int16_t_check = ms->Get_test<int16_t>();
    int16_t _int16_t_out = 0;
    cudaMemcpyFromSymbol(&_int16_t_out, int16_t_out, sizeof(int16_t));
    EXPECT_EQ(int16_t_check, _int16_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_uint16_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_uint16_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint16_t_check = ms->Get_test<uint16_t>();
    uint16_t _uint16_t_out = 0;
    cudaMemcpyFromSymbol(&_uint16_t_out, uint16_t_out, sizeof(uint16_t));
    EXPECT_EQ(uint16_t_check, _uint16_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_int32_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_int32_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int32_t_check = ms->Get_test<int32_t>();
    int32_t _int32_t_out = 0;
    cudaMemcpyFromSymbol(&_int32_t_out, int32_t_out, sizeof(int32_t));
    EXPECT_EQ(int32_t_check, _int32_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_uint32_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_uint32_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint32_t_check = ms->Get_test<uint32_t>();
    uint32_t _uint32_t_out = 0;
    cudaMemcpyFromSymbol(&_uint32_t_out, uint32_t_out, sizeof(uint32_t));
    EXPECT_EQ(uint32_t_check, _uint32_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_int64_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_int64_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int64_t_check = ms->Get_test<int64_t>();
    int64_t _int64_t_out = 0;
    cudaMemcpyFromSymbol(&_int64_t_out, int64_t_out, sizeof(int64_t));
    EXPECT_EQ(int64_t_check, _int64_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_uint64_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_uint64_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint64_t_check = ms->Get_test<uint64_t>();
    uint64_t _uint64_t_out = 0;
    cudaMemcpyFromSymbol(&_uint64_t_out, uint64_t_out, sizeof(uint64_t));
    EXPECT_EQ(uint64_t_check, _uint64_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}

TEST_F(DeviceEnvironmentTest, Get_arrayElement_float) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_float);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto float_check = ms->Get_arrayElement_test<float>();
    float _float_out = 0;
    cudaMemcpyFromSymbol(&_float_out, float_out, sizeof(float));
    EXPECT_EQ(float_check, _float_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_double) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_double);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto double_check = ms->Get_arrayElement_test<double>();
    double _double_out = 0;
    cudaMemcpyFromSymbol(&_double_out, double_out, sizeof(double));
    EXPECT_EQ(double_check, _double_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_int8_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int8_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int8_t_check = ms->Get_arrayElement_test<int8_t>();
    int8_t _int8_t_out = 0;
    cudaMemcpyFromSymbol(&_int8_t_out, int8_t_out, sizeof(int8_t));
    EXPECT_EQ(int8_t_check, _int8_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_uint8_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint8_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint8_t_check = ms->Get_arrayElement_test<uint8_t>();
    uint8_t _uint8_t_out = 0;
    cudaMemcpyFromSymbol(&_uint8_t_out, uint8_t_out, sizeof(uint8_t));
    EXPECT_EQ(uint8_t_check, _uint8_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_int16_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int16_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int16_t_check = ms->Get_arrayElement_test<int16_t>();
    int16_t _int16_t_out = 0;
    cudaMemcpyFromSymbol(&_int16_t_out, int16_t_out, sizeof(int16_t));
    EXPECT_EQ(int16_t_check, _int16_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_uint16_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint16_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint16_t_check = ms->Get_arrayElement_test<uint16_t>();
    uint16_t _uint16_t_out = 0;
    cudaMemcpyFromSymbol(&_uint16_t_out, uint16_t_out, sizeof(uint16_t));
    EXPECT_EQ(uint16_t_check, _uint16_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_uint32_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint32_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint32_t_check = ms->Get_arrayElement_test<uint32_t>();
    uint32_t _uint32_t_out = 0;
    cudaMemcpyFromSymbol(&_uint32_t_out, uint32_t_out, sizeof(uint32_t));
    EXPECT_EQ(uint32_t_check, _uint32_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_int32_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int32_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int32_t_check = ms->Get_arrayElement_test<int32_t>();
    int32_t _int32_t_out = 0;
    cudaMemcpyFromSymbol(&_int32_t_out, int32_t_out, sizeof(int32_t));
    EXPECT_EQ(int32_t_check, _int32_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_uint64_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint64_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint64_t_check = ms->Get_arrayElement_test<uint64_t>();
    uint64_t _uint64_t_out = 0;
    cudaMemcpyFromSymbol(&_uint64_t_out, uint64_t_out, sizeof(uint64_t));
    EXPECT_EQ(uint64_t_check, _uint64_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(DeviceEnvironmentTest, Get_arrayElement_int64_t) {
    // Setup agent fn
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int64_t);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int64_t_check = ms->Get_arrayElement_test<int64_t>();
    int64_t _int64_t_out = 0;
    cudaMemcpyFromSymbol(&_int64_t_out, int64_t_out, sizeof(int64_t));
    EXPECT_EQ(int64_t_check, _int64_t_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
FLAMEGPU_AGENT_FUNCTION(get_array_shorthand, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<float, 3>("k", 0, FLAMEGPU->environment.getProperty<float>("k", 0));
    FLAMEGPU->setVariable<float, 3>("k", 1, FLAMEGPU->environment.getProperty<float>("k", 1));
    FLAMEGPU->setVariable<float, 3>("k", 2, FLAMEGPU->environment.getProperty<float>("k", 2));
    return ALIVE;
}
TEST_F(DeviceEnvironmentTest, Get_array_shorthand) {
    // It's no longer necessary to specify env property array length in agent functions when retrieiving them
    // This test is better ran with FLAMEGPU_SEATBELTS=ON, to catch the seatbelts checking
    // Setup agent fn
    ms->agent.newVariable<float, 3>("k");
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_array_shorthand);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    const std::array<float, 3> t_in = { 12.0f, -12.5f, 13.0f };
    ms->env.newProperty<float, 3>("k", t_in);
    ms->run();
    const std::array<float, 3> t_out = ms->population->at(0).getVariable<float, 3>("k");
    ASSERT_EQ(t_in, t_out);
}
const char* rtc_get_array_shorthand = R"###(
FLAMEGPU_AGENT_FUNCTION(get_array_shorthand, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<float, 3>("k", 0, FLAMEGPU->environment.getProperty<float>("k", 0));
    FLAMEGPU->setVariable<float, 3>("k", 1, FLAMEGPU->environment.getProperty<float>("k", 1));
    FLAMEGPU->setVariable<float, 3>("k", 2, FLAMEGPU->environment.getProperty<float>("k", 2));
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceEnvironmentTest, get_array_shorthand) {
    ModelDescription model("device_env_test");
    // Setup environment
    const std::array<float, 3> t_in = { 12.0f, -12.5f, 13.0f };
    model.Environment().newProperty<float, 3>("k", t_in);
    // Setup agent fn
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<float, 3>("k");
    AgentFunctionDescription deviceFn = agent.newRTCFunction("device_function", rtc_get_array_shorthand);
    LayerDescription devicefn_layer = model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation = CUDASimulation(model);
    cudaSimulation.SimulationConfig().steps = 2;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const std::array<float, 3> t_out = population.at(0).getVariable<float, 3>("k");
    ASSERT_EQ(t_in, t_out);
}
#ifdef FLAMEGPU_USE_GLM
FLAMEGPU_AGENT_FUNCTION(get_array_glm, MessageNone, MessageNone) {
    glm::vec3 t = FLAMEGPU->environment.getProperty<glm::vec3>("k");
    FLAMEGPU->setVariable<float, 3>("k", 0, t[0]);
    FLAMEGPU->setVariable<float, 3>("k", 1, t[1]);
    FLAMEGPU->setVariable<float, 3>("k", 2, t[2]);
    return ALIVE;
}
TEST_F(DeviceEnvironmentTest, Get_array_glm) {
    // Setup agent fn
    ms->agent.newVariable<glm::vec3>("k");
    AgentFunctionDescription deviceFn = ms->agent.newFunction("device_function", get_array_glm);
    LayerDescription devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    const glm::vec3 t_in = { 12.0f, -12.5f, 13.0f };
    ms->env.newProperty<glm::vec3>("k", t_in);
    ms->run();
    const glm::vec3 t_out = ms->population->at(0).getVariable<glm::vec3>("k");
    ASSERT_EQ(t_in, t_out);
}
const char* rtc_get_array_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(get_array_glm, flamegpu::MessageNone, flamegpu::MessageNone) {
    glm::vec3 t = FLAMEGPU->environment.getProperty<glm::vec3>("k");
    FLAMEGPU->setVariable<float, 3>("k", 0, t[0]);
    FLAMEGPU->setVariable<float, 3>("k", 1, t[1]);
    FLAMEGPU->setVariable<float, 3>("k", 2, t[2]);
    return flamegpu::ALIVE;
}
)###";
TEST(RTCDeviceEnvironmentTest, Get_array_glm) {
    ModelDescription model("device_env_test");
    // Setup environment
    const glm::vec3 t_in = { 12.0f, -12.5f, 13.0f };
    model.Environment().newProperty<glm::vec3>("k", t_in);
    // Setup agent fn
    AgentDescription agent = model.newAgent("agent");
    agent.newVariable<glm::vec3>("k");
    AgentFunctionDescription deviceFn = agent.newRTCFunction("device_function", rtc_get_array_glm);
    LayerDescription devicefn_layer = model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    AgentVector population(agent, 1);
    // Do Sim
    CUDASimulation cudaSimulation = CUDASimulation(model);
    cudaSimulation.SimulationConfig().steps = 2;
    cudaSimulation.setPopulationData(population);
    ASSERT_NO_THROW(cudaSimulation.simulate());
    ASSERT_NO_THROW(cudaSimulation.getPopulationData(population));
    const glm::vec3 t_out = population.at(0).getVariable<glm::vec3>("k");
    ASSERT_EQ(t_in, t_out);
}
#else
TEST(DeviceEnvironmentTest, DISABLED_Get_array_glm) {}
TEST(RTCDeviceEnvironmentTest, DISABLED_Get_array_glm) {}
#endif
}  // namespace flamegpu
