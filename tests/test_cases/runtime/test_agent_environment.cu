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

__device__ bool bool_out;
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

FLAMEGPU_AGENT_FUNCTION(get_float, MsgNone, MsgNone) {
    float_out = FLAMEGPU->environment.getProperty<float>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_double, MsgNone, MsgNone) {
    double_out = FLAMEGPU->environment.getProperty<double>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint8_t, MsgNone, MsgNone) {
    uint8_t_out = FLAMEGPU->environment.getProperty<uint8_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int8_t, MsgNone, MsgNone) {
    int8_t_out = FLAMEGPU->environment.getProperty<int8_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint16_t, MsgNone, MsgNone) {
    uint16_t_out = FLAMEGPU->environment.getProperty<uint16_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int16_t, MsgNone, MsgNone) {
    int16_t_out = FLAMEGPU->environment.getProperty<int16_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint32_t, MsgNone, MsgNone) {
    uint32_t_out = FLAMEGPU->environment.getProperty<uint32_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int32_t, MsgNone, MsgNone) {
    int32_t_out = FLAMEGPU->environment.getProperty<int32_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_uint64_t, MsgNone, MsgNone) {
    uint64_t_out = FLAMEGPU->environment.getProperty<uint64_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_int64_t, MsgNone, MsgNone) {
    int64_t_out = FLAMEGPU->environment.getProperty<int64_t>("a");
    bool_out = FLAMEGPU->environment.containsProperty("a");
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(get_arrayElement_float, MsgNone, MsgNone) {
    float_out = FLAMEGPU->environment.getProperty<float>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_double, MsgNone, MsgNone) {
    double_out = FLAMEGPU->environment.getProperty<double>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint8_t, MsgNone, MsgNone) {
    uint8_t_out = FLAMEGPU->environment.getProperty<uint8_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int8_t, MsgNone, MsgNone) {
    int8_t_out = FLAMEGPU->environment.getProperty<int8_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint16_t, MsgNone, MsgNone) {
    uint16_t_out = FLAMEGPU->environment.getProperty<uint16_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int16_t, MsgNone, MsgNone) {
    int16_t_out = FLAMEGPU->environment.getProperty<int16_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint32_t, MsgNone, MsgNone) {
    uint32_t_out = FLAMEGPU->environment.getProperty<uint32_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int32_t, MsgNone, MsgNone) {
    int32_t_out = FLAMEGPU->environment.getProperty<int32_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_uint64_t, MsgNone, MsgNone) {
    uint64_t_out = FLAMEGPU->environment.getProperty<uint64_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(get_arrayElement_int64_t, MsgNone, MsgNone) {
    int64_t_out = FLAMEGPU->environment.getProperty<int64_t>("a", 1);
    bool_out = FLAMEGPU->environment.containsProperty("b");
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
    AgentDescription &agent;
    AgentVector *population;
    EnvironmentDescription &env;
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
class AgentEnvironmentTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
        bool _bool_out = false;
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
        cudaMemcpyToSymbol(bool_out, &_bool_out, sizeof(bool));
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


TEST_F(AgentEnvironmentTest, Get_float) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_float);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto float_check = ms->Get_test<float>();
    float _float_out = 0;
    cudaMemcpyFromSymbol(&_float_out, float_out, sizeof(float));
    EXPECT_EQ(float_check, _float_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_double) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_double);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto double_check = ms->Get_test<double>();
    double _double_out = 0;
    cudaMemcpyFromSymbol(&_double_out, double_out, sizeof(double));
    EXPECT_EQ(double_check, _double_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_int8_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_int8_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int8_t_check = ms->Get_test<int8_t>();
    int8_t _int8_t_out = 0;
    cudaMemcpyFromSymbol(&_int8_t_out, int8_t_out, sizeof(int8_t));
    EXPECT_EQ(int8_t_check, _int8_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_uint8_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_uint8_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint8_t_check = ms->Get_test<uint8_t>();
    uint8_t _uint8_t_out = 0;
    cudaMemcpyFromSymbol(&_uint8_t_out, uint8_t_out, sizeof(uint8_t));
    EXPECT_EQ(uint8_t_check, _uint8_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_int16_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_int16_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int16_t_check = ms->Get_test<int16_t>();
    int16_t _int16_t_out = 0;
    cudaMemcpyFromSymbol(&_int16_t_out, int16_t_out, sizeof(int16_t));
    EXPECT_EQ(int16_t_check, _int16_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_uint16_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_uint16_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint16_t_check = ms->Get_test<uint16_t>();
    uint16_t _uint16_t_out = 0;
    cudaMemcpyFromSymbol(&_uint16_t_out, uint16_t_out, sizeof(uint16_t));
    EXPECT_EQ(uint16_t_check, _uint16_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_int32_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_int32_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int32_t_check = ms->Get_test<int32_t>();
    int32_t _int32_t_out = 0;
    cudaMemcpyFromSymbol(&_int32_t_out, int32_t_out, sizeof(int32_t));
    EXPECT_EQ(int32_t_check, _int32_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_uint32_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_uint32_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint32_t_check = ms->Get_test<uint32_t>();
    uint32_t _uint32_t_out = 0;
    cudaMemcpyFromSymbol(&_uint32_t_out, uint32_t_out, sizeof(uint32_t));
    EXPECT_EQ(uint32_t_check, _uint32_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_int64_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_int64_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int64_t_check = ms->Get_test<int64_t>();
    int64_t _int64_t_out = 0;
    cudaMemcpyFromSymbol(&_int64_t_out, int64_t_out, sizeof(int64_t));
    EXPECT_EQ(int64_t_check, _int64_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_uint64_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_uint64_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint64_t_check = ms->Get_test<uint64_t>();
    uint64_t _uint64_t_out = 0;
    cudaMemcpyFromSymbol(&_uint64_t_out, uint64_t_out, sizeof(uint64_t));
    EXPECT_EQ(uint64_t_check, _uint64_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(true, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}

TEST_F(AgentEnvironmentTest, Get_arrayElement_float) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_float);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto float_check = ms->Get_arrayElement_test<float>();
    float _float_out = 0;
    cudaMemcpyFromSymbol(&_float_out, float_out, sizeof(float));
    EXPECT_EQ(float_check, _float_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_double) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_double);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto double_check = ms->Get_arrayElement_test<double>();
    double _double_out = 0;
    cudaMemcpyFromSymbol(&_double_out, double_out, sizeof(double));
    EXPECT_EQ(double_check, _double_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_int8_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int8_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int8_t_check = ms->Get_arrayElement_test<int8_t>();
    int8_t _int8_t_out = 0;
    cudaMemcpyFromSymbol(&_int8_t_out, int8_t_out, sizeof(int8_t));
    EXPECT_EQ(int8_t_check, _int8_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_uint8_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint8_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint8_t_check = ms->Get_arrayElement_test<uint8_t>();
    uint8_t _uint8_t_out = 0;
    cudaMemcpyFromSymbol(&_uint8_t_out, uint8_t_out, sizeof(uint8_t));
    EXPECT_EQ(uint8_t_check, _uint8_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_int16_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int16_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int16_t_check = ms->Get_arrayElement_test<int16_t>();
    int16_t _int16_t_out = 0;
    cudaMemcpyFromSymbol(&_int16_t_out, int16_t_out, sizeof(int16_t));
    EXPECT_EQ(int16_t_check, _int16_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_uint16_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint16_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint16_t_check = ms->Get_arrayElement_test<uint16_t>();
    uint16_t _uint16_t_out = 0;
    cudaMemcpyFromSymbol(&_uint16_t_out, uint16_t_out, sizeof(uint16_t));
    EXPECT_EQ(uint16_t_check, _uint16_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_uint32_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint32_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint32_t_check = ms->Get_arrayElement_test<uint32_t>();
    uint32_t _uint32_t_out = 0;
    cudaMemcpyFromSymbol(&_uint32_t_out, uint32_t_out, sizeof(uint32_t));
    EXPECT_EQ(uint32_t_check, _uint32_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_int32_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int32_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int32_t_check = ms->Get_arrayElement_test<int32_t>();
    int32_t _int32_t_out = 0;
    cudaMemcpyFromSymbol(&_int32_t_out, int32_t_out, sizeof(int32_t));
    EXPECT_EQ(int32_t_check, _int32_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_uint64_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_uint64_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto uint64_t_check = ms->Get_arrayElement_test<uint64_t>();
    uint64_t _uint64_t_out = 0;
    cudaMemcpyFromSymbol(&_uint64_t_out, uint64_t_out, sizeof(uint64_t));
    EXPECT_EQ(uint64_t_check, _uint64_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
TEST_F(AgentEnvironmentTest, Get_arrayElement_int64_t) {
    // Setup agent fn
    AgentFunctionDescription &deviceFn = ms->agent.newFunction("device_function", get_arrayElement_int64_t);
    LayerDescription &devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    auto int64_t_check = ms->Get_arrayElement_test<int64_t>();
    int64_t _int64_t_out = 0;
    cudaMemcpyFromSymbol(&_int64_t_out, int64_t_out, sizeof(int64_t));
    EXPECT_EQ(int64_t_check, _int64_t_out);
    bool _bool_out = false;
    cudaMemcpyFromSymbol(&_bool_out, bool_out, sizeof(bool));
    EXPECT_EQ(false, _bool_out);
    EXPECT_EQ(cudaGetLastError(), CUDA_SUCCESS);
}
#ifdef USE_GLM
FLAMEGPU_AGENT_FUNCTION(get_array_glm, MsgNone, MsgNone) {
    glm::vec3 t = FLAMEGPU->environment.getProperty<glm::vec3>("k");
    FLAMEGPU->setVariable<float, 3>("k", 0, t[0]);
    FLAMEGPU->setVariable<float, 3>("k", 1, t[1]);
    FLAMEGPU->setVariable<float, 3>("k", 2, t[2]);
    return ALIVE;
}
TEST_F(AgentEnvironmentTest, Get_array_glm) {
    // Setup agent fn
    ms->agent.newVariable<float, 3>("k");
    AgentFunctionDescription& deviceFn = ms->agent.newFunction("device_function", get_array_glm);
    LayerDescription& devicefn_layer = ms->model.newLayer("devicefn_layer");
    devicefn_layer.addAgentFunction(deviceFn);
    // Setup environment
    const std::array<float, 3> t_in = { 12.0f, -12.5f, 13.0f };
    ms->env.newProperty<float, 3>("k", t_in);
    ms->run();
    const std::array<float, 3> t_out = ms->population->at(0).getVariable<float, 3>("k");
    ASSERT_EQ(t_in, t_out);
}
const char* rtc_get_array_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(get_array_glm, flamegpu::MsgNone, flamegpu::MsgNone) {
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
    const std::array<float, 3> t_in = { 12.0f, -12.5f, 13.0f };
    model.Environment().newProperty<float, 3>("k", t_in);
    // Setup agent fn
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<float, 3>("k");
    AgentFunctionDescription& deviceFn = agent.newRTCFunction("device_function", rtc_get_array_glm);
    LayerDescription& devicefn_layer = model.newLayer("devicefn_layer");
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
#else

#endif
}  // namespace flamegpu
