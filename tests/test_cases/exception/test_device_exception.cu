
#include "gtest/gtest.h"

#include "flamegpu/flamegpu.h"

namespace flamegpu {

// These tests wont work if built with SEATBELTS=OFF, so mark them all as disabled instead
#if defined(SEATBELTS) && !SEATBELTS
#undef TEST_F
#define TEST_F(test_fixture, test_name)\
  GTEST_TEST_(test_fixture, DISABLED_ ## test_name, test_fixture, \
              ::testing::internal::GetTypeId<test_fixture>())
#endif

namespace test_device_exception {

namespace {

class MiniSim {
 public:
    MiniSim() :
        model("model"),
        agent(model.newAgent("agent")) {
        agent.newVariable<int>("int");
        agent.newVariable<int, 2>("array");
        agent.newVariable<id_t>("id_other", ID_NOT_SET);
        model.Environment().newProperty<int>("int", 12);
        model.Environment().newProperty<int, 2>("array", {12, 13});
    }
    void run(int steps = 2) {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        CUDASimulation cudaSimulation(model);
        cudaSimulation.SimulationConfig().steps = steps;
        // This fails as agentMap is empty
        AgentVector population(agent, 1);
        cudaSimulation.setPopulationData(population);
        EXPECT_THROW(cudaSimulation.simulate(), DeviceError);
    }
    template<typename T>
    void addFunc(T func) {
        agent.newFunction("name", func);
        model.newLayer().addAgentFunction(func);
    }
    template<typename T, typename C>
    void addFuncCdn(T func, C cdn) {
        auto &fn = agent.newFunction("name", func);
        fn.setFunctionCondition(cdn);
        model.newLayer().addAgentFunction(fn);
    }
    template<typename T>
    void addAgentOutFunc(T func) {
        auto &fn = agent.newFunction("name", func);
        fn.setAgentOutput("agent");
        model.newLayer().addAgentFunction(func);
    }
    template<typename Msg, typename T>
    void addMsgOutFunc(T func) {
        typename Msg::Description &msg = model.newMessage<Msg>("message");
        msg.template newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMsgS2DOutFunc(T func) {
        MsgSpatial2D::Description &msg = model.newMessage<MsgSpatial2D>("message");
        msg.setMin(-1, -1);
        msg.setMax(1, 1);
        msg.setRadius(1);
        msg.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMsgS3DOutFunc(T func) {
        MsgSpatial3D::Description &msg = model.newMessage<MsgSpatial3D>("message");
        msg.setMin(-1, -1, -2);
        msg.setMax(1, 1, 1);
        msg.setRadius(1);
        msg.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMsgA1DOutFunc(T func) {
        MsgArray::Description &msg = model.newMessage<MsgArray>("message");
        msg.setLength(10);
        msg.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMsgA2DOutFunc(T func) {
        MsgArray2D::Description &msg = model.newMessage<MsgArray2D>("message");
        msg.setDimensions(10, 10);
        msg.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMsgA3DOutFunc(T func) {
        MsgArray3D::Description &msg = model.newMessage<MsgArray3D>("message");
        msg.setDimensions(10, 10, 10);
        msg.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMsgBucketOutFunc(T func) {
        MsgBucket::Description &msg = model.newMessage<MsgBucket>("message");
        msg.setBounds(0, 1023);
        msg.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }

    template<typename Msg, typename T1, typename T2>
    void addMsgInFunc(T1 out_func, T2 in_func) {
        auto &msg = model.newMessage<Msg>("message");
        msg.template newVariable<int>("int");
        {
            auto &fn = agent.newFunction("out_name", out_func);
            fn.setMessageOutput("message");
            model.newLayer().addAgentFunction(out_func);
        }
        {
            auto &fn = agent.newFunction("in_name", in_func);
            fn.setMessageInput("message");
            model.newLayer().addAgentFunction(in_func);
        }
    }
    template<typename T1, typename T2>
    void addMsgS2DInFunc(T1 out_func, T2 in_func) {
        MsgSpatial2D::Description &msg = model.newMessage<MsgSpatial2D>("message");
        msg.setMin(-1, -1);
        msg.setMax(1, 1);
        msg.setRadius(1);
        msg.template newVariable<int>("int");
        {
            auto &fn = agent.newFunction("out_name", out_func);
            fn.setMessageOutput("message");
            model.newLayer().addAgentFunction(out_func);
        }
        {
            auto &fn = agent.newFunction("in_name", in_func);
            fn.setMessageInput("message");
            model.newLayer().addAgentFunction(in_func);
        }
    }
    template<typename T1, typename T2>
    void addMsgS3DInFunc(T1 out_func, T2 in_func) {
        MsgSpatial3D::Description &msg = model.newMessage<MsgSpatial3D>("message");
        msg.setMin(-1, -1, -1);
        msg.setMax(1, 1, 1);
        msg.setRadius(1);
        msg.template newVariable<int>("int");
        {
            auto &fn = agent.newFunction("out_name", out_func);
            fn.setMessageOutput("message");
            model.newLayer().addAgentFunction(out_func);
        }
        {
            auto &fn = agent.newFunction("in_name", in_func);
            fn.setMessageInput("message");
            model.newLayer().addAgentFunction(in_func);
        }
    }
    template<typename T1, typename T2>
    void addMsgA1DInFunc(T1 out_func, T2 in_func) {
        MsgArray::Description &msg = model.newMessage<MsgArray>("message");
        msg.setLength(10);
        msg.template newVariable<int>("int");
        {
            auto &fn = agent.newFunction("out_name", out_func);
            fn.setMessageOutput("message");
            model.newLayer().addAgentFunction(out_func);
        }
        {
            auto &fn = agent.newFunction("in_name", in_func);
            fn.setMessageInput("message");
            model.newLayer().addAgentFunction(in_func);
        }
    }
    template<typename T1, typename T2>
    void addMsgA2DInFunc(T1 out_func, T2 in_func) {
        MsgArray2D::Description &msg = model.newMessage<MsgArray2D>("message");
        msg.setDimensions(10, 10);
        msg.template newVariable<int>("int");
        {
            auto &fn = agent.newFunction("out_name", out_func);
            fn.setMessageOutput("message");
            model.newLayer().addAgentFunction(out_func);
        }
        {
            auto &fn = agent.newFunction("in_name", in_func);
            fn.setMessageInput("message");
            model.newLayer().addAgentFunction(in_func);
        }
    }
    template<typename T1, typename T2>
    void addMsgA3DInFunc(T1 out_func, T2 in_func) {
        MsgArray3D::Description &msg = model.newMessage<MsgArray3D>("message");
        msg.setDimensions(10, 10, 10);
        msg.template newVariable<int>("int");
        {
            auto &fn = agent.newFunction("out_name", out_func);
            fn.setMessageOutput("message");
            model.newLayer().addAgentFunction(out_func);
        }
        {
            auto &fn = agent.newFunction("in_name", in_func);
            fn.setMessageInput("message");
            model.newLayer().addAgentFunction(in_func);
        }
    }
    template<typename T1, typename T2>
    void addMsgBucketInFunc(T1 out_func, T2 in_func) {
        MsgBucket::Description &msg = model.newMessage<MsgBucket>("message");
        msg.setBounds(0, 1023);
        msg.template newVariable<int>("int");
        {
            auto &fn = agent.newFunction("out_name", out_func);
            fn.setMessageOutput("message");
            model.newLayer().addAgentFunction(out_func);
        }
        {
            auto &fn = agent.newFunction("in_name", in_func);
            fn.setMessageInput("message");
            model.newLayer().addAgentFunction(in_func);
        }
    }
    ModelDescription model;
    AgentDescription &agent;
};
/**
* This defines a common fixture used as a base for all test cases in the file
* @see https://github.com/google/googletest/blob/master/googletest/samples/sample5_unittest.cc
*/
class DeviceExceptionTest : public testing::Test {
 protected:
    void SetUp() override {
        ms = new MiniSim();
    }
    void TearDown() override {
        delete ms;
    }

    MiniSim *ms = nullptr;
};
}  // namespace
//
// ReadOnlyDeviceAPI::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(GetUnknownAgentVariable, MsgNone, MsgNone) {
    FLAMEGPU->getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetUnknownAgentVariable) {
    // Add required agent function
    ms->addFunc(GetUnknownAgentVariable);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(GetAgentVariableBadType, MsgNone, MsgNone) {
    FLAMEGPU->getVariable<double>("int");   // Note type checking only confirms size currently
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetAgentVariableBadType) {
    // Add required agent function
    ms->addFunc(GetAgentVariableBadType);
    // Test Something
    ms->run(1);
}

// ReadOnlyDeviceAPI::getVariable<T, N, M>()
FLAMEGPU_AGENT_FUNCTION(GetUnknownAgentVariableArray, MsgNone, MsgNone) {
    FLAMEGPU->getVariable<int, 3>("nope", 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetUnknownAgentVariableArray) {
    // Add required agent function
    ms->addFunc(GetUnknownAgentVariableArray);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(GetAgentVariableArrayBadType, MsgNone, MsgNone) {
    FLAMEGPU->getVariable<double, 3>("array", 0);   // Note type checking only confirms size currently
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetAgentVariableArrayBadType) {
    // Add required agent function
    ms->addFunc(GetAgentVariableArrayBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(GetAgentVariableArrayOutOfRange, MsgNone, MsgNone) {
    FLAMEGPU->getVariable<int, 2>("array", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetAgentVariableArrayOutOfRange) {
    // Add required agent function
    ms->addFunc(GetAgentVariableArrayOutOfRange);
    // Test Something
    ms->run(1);
}

// DeviceAPI::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(SetUnknownAgentVariable, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetUnknownAgentVariable) {
    // Add required agent function
    ms->addFunc(SetUnknownAgentVariable);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(SetAgentVariableBadType, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<double>("int", 12.0);  // Note type checking only confirms size currently
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetAgentVariableBadType) {
    // Add required agent function
    ms->addFunc(SetAgentVariableBadType);
    // Test Something
    ms->run(1);
}

// DeviceAPI::setVariable<T, N, M>()
FLAMEGPU_AGENT_FUNCTION(SetUnknownAgentVariableArray, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<int, 3>("nope", 0, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetUnknownAgentVariableArray) {
    // Add required agent function
    ms->addFunc(SetUnknownAgentVariableArray);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(SetAgentVariableArrayBadType, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<double, 3>("array", 0, 12.0);  // Note type checking only confirms size currently
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetAgentVariableArrayBadType) {
    // Add required agent function
    ms->addFunc(SetAgentVariableArrayBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(SetAgentVariableArrayOutOfRange, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<int, 2>("array", 2, 12.0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetAgentVariableArrayOutOfRange) {
    // Add required agent function
    ms->addFunc(SetAgentVariableArrayOutOfRange);
    // Test Something
    ms->run(1);
}

// AgentRandom::uniform<T>(T, T)
FLAMEGPU_AGENT_FUNCTION(AgentRandomUniformInvalidRange1, MsgNone, MsgNone) {
    FLAMEGPU->random.uniform<int>(5, 4);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentRandomUniformInvalidRange1) {
    // Add required agent function
    ms->addFunc(AgentRandomUniformInvalidRange1);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentRandomUniformInvalidRange2, MsgNone, MsgNone) {
    FLAMEGPU->random.uniform<int64_t>(5, 4);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentRandomUniformInvalidRange2) {
    // Add required agent function
    ms->addFunc(AgentRandomUniformInvalidRange2);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentRandomUniformInvalidRange3, MsgNone, MsgNone) {
    FLAMEGPU->random.uniform<uint64_t>(5, 4);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentRandomUniformInvalidRange3) {
    // Add required agent function
    ms->addFunc(AgentRandomUniformInvalidRange3);
    // Test Something
    ms->run(1);
}

// DeviceEnvironment::get<T, N>()
FLAMEGPU_AGENT_FUNCTION(DeviceEnvironmentGetUnknownProperty, MsgNone, MsgNone) {
    FLAMEGPU->environment.getProperty<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, DeviceEnvironmentGetUnknownProperty) {
    // Add required agent function
    ms->addFunc(DeviceEnvironmentGetUnknownProperty);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(DeviceEnvironmentGetBadType, MsgNone, MsgNone) {
    FLAMEGPU->environment.getProperty<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, DeviceEnvironmentGetBadType) {
    // Add required agent function
    ms->addFunc(DeviceEnvironmentGetBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(DeviceEnvironmentGetOutOfRange, MsgNone, MsgNone) {
    FLAMEGPU->environment.getProperty<int>("array", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, DeviceEnvironmentGetOutOfRange) {
    // Add required agent function
    ms->addFunc(DeviceEnvironmentGetOutOfRange);
    // Test Something
    ms->run(1);
}

// AgentOut::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableUnknown, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableUnknown) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableBadType, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<double>("int", 2.0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableBadType) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableBadType);
    // Test Something
    ms->run(1);
}

// AgentOut::setVariable<T, N, M>()
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayUnknown, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<int, 2>("nope", 0, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayUnknown) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayBadType, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<double, 2>("array", 0, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayBadType) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayOutOfRange1, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<int, 2>("array", 2, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayOutOfRange1) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayOutOfRange1);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayOutOfRange2, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<int, 3>("array", 2, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayOutOfRange2) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayOutOfRange2);
    // Test Something
    ms->run(1);
}

// MsgBruteForce::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgBruteForceOutVariableUnknown, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBruteForceOutVariableUnknown) {
    // Add required agent function
    ms->addMsgOutFunc<MsgBruteForce>(MsgBruteForceOutVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgBruteForceOutVariableBadType, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable<double>("int", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBruteForceOutVariableBadType) {
    // Add required agent function
    ms->addMsgOutFunc<MsgBruteForce>(MsgBruteForceOutVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgBruteForce::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgBruteForceDefaultOut, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgBruteForceInVariableUnknown, MsgBruteForce, MsgNone) {
    for (auto m : FLAMEGPU->message_in) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBruteForceInVariableUnknown) {
    // Add required agent function
    ms->addMsgInFunc<MsgBruteForce>(MsgBruteForceDefaultOut, MsgBruteForceInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgBruteForceInVariableBadType, MsgBruteForce, MsgNone) {
    for (auto m : FLAMEGPU->message_in) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBruteForceInVariableBadType) {
    // Add required agent function
    ms->addMsgInFunc<MsgBruteForce>(MsgBruteForceDefaultOut, MsgBruteForceInVariableBadType);
    // Test Something
    ms->run(1);
}


// MsgSpatial2D::Out::setVariable<T, N>() (These should be identical to MsgBruteForce due to the object being inherited)
FLAMEGPU_AGENT_FUNCTION(MsgSpatial2DOutVariableUnknown, MsgNone, MsgSpatial2D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial2DOutVariableUnknown) {
    // Add required agent function
    ms->addMsgS2DOutFunc(MsgSpatial2DOutVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgSpatial2DOutVariableBadType, MsgNone, MsgSpatial2D) {
    FLAMEGPU->message_out.setVariable<double>("int", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial2DOutVariableBadType) {
    // Add required agent function
    ms->addMsgS2DOutFunc(MsgSpatial2DOutVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgSpatial2D::In::Filter::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgSpatial2DDefaultOut, MsgNone, MsgSpatial2D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setLocation(0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgSpatial2DInVariableUnknown, MsgSpatial2D, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0)) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial2DInVariableUnknown) {
    // Add required agent function
    ms->addMsgS2DInFunc(MsgSpatial2DDefaultOut, MsgSpatial2DInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgSpatial2DInVariableBadType, MsgSpatial2D, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0)) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial2DInVariableBadType) {
    // Add required agent function
    ms->addMsgS2DInFunc(MsgSpatial2DDefaultOut, MsgSpatial2DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgSpatial3D::Out::setVariable<T, N>() (These should be identical to MsgBruteForce due to the object being inherited)
FLAMEGPU_AGENT_FUNCTION(MsgSpatial3DOutVariableUnknown, MsgNone, MsgSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial3DOutVariableUnknown) {
    // Add required agent function
    ms->addMsgS3DOutFunc(MsgSpatial3DOutVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgSpatial3DOutVariableBadType, MsgNone, MsgSpatial3D) {
    FLAMEGPU->message_out.setVariable<double>("int", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial3DOutVariableBadType) {
    // Add required agent function
    ms->addMsgS3DOutFunc(MsgSpatial3DOutVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgSpatial3D::In::Filter::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgSpatial3DDefaultOut, MsgNone, MsgSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setLocation(0, 0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgSpatial3DInVariableUnknown, MsgSpatial3D, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0, 0)) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial3DInVariableUnknown) {
    // Add required agent function
    ms->addMsgS3DInFunc(MsgSpatial3DDefaultOut, MsgSpatial3DInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgSpatial3DInVariableBadType, MsgSpatial3D, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0, 0)) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgSpatial3DInVariableBadType) {
    // Add required agent function
    ms->addMsgS3DInFunc(MsgSpatial3DDefaultOut, MsgSpatial3DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgArray::Out::setIndex()
FLAMEGPU_AGENT_FUNCTION(MsgArrayOutIndexOutOfRange, MsgNone, MsgArray) {
    FLAMEGPU->message_out.setIndex(10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArrayOutIndexOutOfRange) {
    // Add required agent function
    ms->addMsgA1DOutFunc(MsgArrayOutIndexOutOfRange);
    // Test Something
    ms->run(1);
}

// MsgArray::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgArrayOutUnknownVariable, MsgNone, MsgArray) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArrayOutUnknownVariable) {
    // Add required agent function
    ms->addMsgA1DOutFunc(MsgArrayOutUnknownVariable);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArrayVariableBadType, MsgNone, MsgArray) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArrayVariableBadType) {
    // Add required agent function
    ms->addMsgA1DOutFunc(MsgArrayVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgArray::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgArrayDefaultOut, MsgNone, MsgArray) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setIndex(0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgArrayInVariableUnknown, MsgArray, MsgNone) {
    FLAMEGPU->message_in.at(0).getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArrayInVariableUnknown) {
    // Add required agent function
    ms->addMsgA1DInFunc(MsgArrayDefaultOut, MsgArrayInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArrayInVariableBadType, MsgArray, MsgNone) {
    FLAMEGPU->message_in.at(0).getVariable<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArrayInVariableBadType) {
    // Add required agent function
    ms->addMsgA1DInFunc(MsgArrayDefaultOut, MsgArrayInVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgArray::In::operator()()
FLAMEGPU_AGENT_FUNCTION(MsgArrayInVariableBadRadius, MsgArray, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0, 0)) {
        m.getVariable<int>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArrayInVariableBadRadius) {
    // Add required agent function
    ms->addMsgA1DInFunc(MsgArrayDefaultOut, MsgArrayInVariableBadRadius);
    // Test Something
    ms->run(1);
}

// MsgArray::In::at()
FLAMEGPU_AGENT_FUNCTION(MsgArrayInVariableIndexOutOfBounds, MsgArray, MsgNone) {
    FLAMEGPU->message_in.at(10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArrayInVariableIndexOutOfBounds) {
    // Add required agent function
    ms->addMsgA1DInFunc(MsgArrayDefaultOut, MsgArrayInVariableIndexOutOfBounds);
    // Test Something
    ms->run(1);
}

// MsgArray2D::Out::setIndex()
FLAMEGPU_AGENT_FUNCTION(MsgArray2DOutIndexOutOfRange, MsgNone, MsgArray2D) {
    FLAMEGPU->message_out.setIndex(10, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DOutIndexOutOfRange) {
    // Add required agent function
    ms->addMsgA2DOutFunc(MsgArray2DOutIndexOutOfRange);
    // Test Something
    ms->run(1);
}

// MsgArray2D::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgArray2DOutUnknownVariable, MsgNone, MsgArray2D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DOutUnknownVariable) {
    // Add required agent function
    ms->addMsgA2DOutFunc(MsgArray2DOutUnknownVariable);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArray2DVariableBadType, MsgNone, MsgArray2D) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DVariableBadType) {
    // Add required agent function
    ms->addMsgA2DOutFunc(MsgArray2DVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgArray2D::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgArray2DDefaultOut, MsgNone, MsgArray2D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setIndex(0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgArray2DInVariableUnknown, MsgArray2D, MsgNone) {
    FLAMEGPU->message_in.at(0, 0).getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DInVariableUnknown) {
    // Add required agent function
    ms->addMsgA2DInFunc(MsgArray2DDefaultOut, MsgArray2DInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArray2DInVariableBadType, MsgArray2D, MsgNone) {
    FLAMEGPU->message_in.at(0, 0).getVariable<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DInVariableBadType) {
    // Add required agent function
    ms->addMsgA2DInFunc(MsgArray2DDefaultOut, MsgArray2DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgArray2D::In::operator()()
FLAMEGPU_AGENT_FUNCTION(MsgArray2DInVariableBadRadius, MsgArray2D, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0, 0, 0)) {
        m.getVariable<int>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DInVariableBadRadius) {
    // Add required agent function
    ms->addMsgA2DInFunc(MsgArray2DDefaultOut, MsgArray2DInVariableBadRadius);
    // Test Something
    ms->run(1);
}

// MsgArray2D::In::at()
FLAMEGPU_AGENT_FUNCTION(MsgArray2DInVariableIndexOutOfBoundsX, MsgArray2D, MsgNone) {
    FLAMEGPU->message_in.at(10, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DInVariableIndexOutOfBoundsX) {
    // Add required agent function
    ms->addMsgA2DInFunc(MsgArray2DDefaultOut, MsgArray2DInVariableIndexOutOfBoundsX);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArray2DInVariableIndexOutOfBoundsY, MsgArray2D, MsgNone) {
    FLAMEGPU->message_in.at(0, 10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray2DInVariableIndexOutOfBoundsY) {
    // Add required agent function
    ms->addMsgA2DInFunc(MsgArray2DDefaultOut, MsgArray2DInVariableIndexOutOfBoundsY);
    // Test Something
    ms->run(1);
}

// MsgArray3D::Out::setIndex()
FLAMEGPU_AGENT_FUNCTION(MsgArray3DOutIndexOutOfRange, MsgNone, MsgArray3D) {
    FLAMEGPU->message_out.setIndex(10, 0, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DOutIndexOutOfRange) {
    // Add required agent function
    ms->addMsgA3DOutFunc(MsgArray3DOutIndexOutOfRange);
    // Test Something
    ms->run(1);
}

// MsgArray3D::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgArray3DOutUnknownVariable, MsgNone, MsgArray3D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DOutUnknownVariable) {
    // Add required agent function
    ms->addMsgA3DOutFunc(MsgArray3DOutUnknownVariable);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArray3DVariableBadType, MsgNone, MsgArray3D) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DVariableBadType) {
    // Add required agent function
    ms->addMsgA3DOutFunc(MsgArray3DVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgArray3D::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgArray3DDefaultOut, MsgNone, MsgArray3D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setIndex(0, 0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgArray3DInVariableUnknown, MsgArray3D, MsgNone) {
    FLAMEGPU->message_in.at(0, 0, 0).getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DInVariableUnknown) {
    // Add required agent function
    ms->addMsgA3DInFunc(MsgArray3DDefaultOut, MsgArray3DInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArray3DInVariableBadType, MsgArray3D, MsgNone) {
    FLAMEGPU->message_in.at(0, 0, 0).getVariable<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DInVariableBadType) {
    // Add required agent function
    ms->addMsgA3DInFunc(MsgArray3DDefaultOut, MsgArray3DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgArray3D::In::operator()()
FLAMEGPU_AGENT_FUNCTION(MsgArray3DInVariableBadRadius, MsgArray3D, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0, 0, 0, 0)) {
        m.getVariable<int>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DInVariableBadRadius) {
    // Add required agent function
    ms->addMsgA3DInFunc(MsgArray3DDefaultOut, MsgArray3DInVariableBadRadius);
    // Test Something
    ms->run(1);
}

// MsgArray3D::In::at()
FLAMEGPU_AGENT_FUNCTION(MsgArray3DInVariableIndexOutOfBoundsX, MsgArray3D, MsgNone) {
    FLAMEGPU->message_in.at(10, 0, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DInVariableIndexOutOfBoundsX) {
    // Add required agent function
    ms->addMsgA3DInFunc(MsgArray3DDefaultOut, MsgArray3DInVariableIndexOutOfBoundsX);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArray3DInVariableIndexOutOfBoundsY, MsgArray3D, MsgNone) {
    FLAMEGPU->message_in.at(0, 10, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DInVariableIndexOutOfBoundsY) {
    // Add required agent function
    ms->addMsgA3DInFunc(MsgArray3DDefaultOut, MsgArray3DInVariableIndexOutOfBoundsY);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgArray3DInVariableIndexOutOfBoundsZ, MsgArray3D, MsgNone) {
    FLAMEGPU->message_in.at(0, 0, 10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgArray3DInVariableIndexOutOfBoundsZ) {
    // Add required agent function
    ms->addMsgA3DInFunc(MsgArray3DDefaultOut, MsgArray3DInVariableIndexOutOfBoundsZ);
    // Test Something
    ms->run(1);
}

// MsgBucket::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgBucketOutUnknownVariable, MsgNone, MsgBucket) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBucketOutUnknownVariable) {
    // Add required agent function
    ms->addMsgBucketOutFunc(MsgBucketOutUnknownVariable);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketVariableBadType, MsgNone, MsgBucket) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBucketVariableBadType) {
    // Add required agent function
    ms->addMsgBucketOutFunc(MsgBucketVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgBucket::Out::setKey()
FLAMEGPU_AGENT_FUNCTION(MsgBucketOutBadKey1, MsgNone, MsgBucket) {
    FLAMEGPU->message_out.setKey(-1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketOutBadKey2, MsgNone, MsgBucket) {
    FLAMEGPU->message_out.setKey(1024);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBucketOutBadKey1) {
    // Add required agent function
    ms->addMsgBucketOutFunc(MsgBucketOutBadKey1);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MsgBucketOutBadKey2) {
    // Add required agent function
    ms->addMsgBucketOutFunc(MsgBucketOutBadKey2);
    // Test Something
    ms->run(1);
}

// MsgBucket::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MsgBucketDefaultOut, MsgNone, MsgBucket) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setKey(0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketInVariableUnknown, MsgBucket, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0)) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBucketInVariableUnknown) {
    // Add required agent function
    ms->addMsgBucketInFunc(MsgBucketDefaultOut, MsgBucketInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketInVariableBadType, MsgBucket, MsgNone) {
    for (auto m : FLAMEGPU->message_in(0)) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBucketInVariableBadType) {
    // Add required agent function
    ms->addMsgBucketInFunc(MsgBucketDefaultOut, MsgBucketInVariableBadType);
    // Test Something
    ms->run(1);
}

// MsgBucket::In::operator()(key)
// MsgBucket::In::operator()(beginKey, endKey)
FLAMEGPU_AGENT_FUNCTION(MsgBucketInBadKey1, MsgBucket, MsgNone) {
    FLAMEGPU->message_in(-1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketInBadKey2, MsgBucket, MsgNone) {
    FLAMEGPU->message_in(1024);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketInBadKey3, MsgBucket, MsgNone) {
    FLAMEGPU->message_in(-1, 1023);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketInBadKey4, MsgBucket, MsgNone) {
    FLAMEGPU->message_in(0, 1025);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MsgBucketInBadKey5, MsgBucket, MsgNone) {
    FLAMEGPU->message_in(100, 5);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MsgBucketInBadKey1) {
    // Add required agent function
    ms->addMsgBucketInFunc(MsgBucketDefaultOut, MsgBucketInBadKey1);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MsgBucketInBadKey2) {
    // Add required agent function
    ms->addMsgBucketInFunc(MsgBucketDefaultOut, MsgBucketInBadKey2);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MsgBucketInBadKey3) {
    // Add required agent function
    ms->addMsgBucketInFunc(MsgBucketDefaultOut, MsgBucketInBadKey3);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MsgBucketInBadKey4) {
    // Add required agent function
    ms->addMsgBucketInFunc(MsgBucketDefaultOut, MsgBucketInBadKey4);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MsgBucketInBadKey5) {
    // Add required agent function
    ms->addMsgBucketInFunc(MsgBucketDefaultOut, MsgBucketInBadKey5);
    // Test Something
    ms->run(1);
}

// Test error in agent function condition
FLAMEGPU_AGENT_FUNCTION_CONDITION(AgentFunctionConditionError1) {
    FLAMEGPU->getVariable<int>("nope");
    return true;
}FLAMEGPU_AGENT_FUNCTION_CONDITION(AgentFunctionConditionError2) {
    FLAMEGPU->getVariable<int>("nope");
    return false;
}
FLAMEGPU_AGENT_FUNCTION(GetAgentVariable, MsgNone, MsgNone) {
    FLAMEGPU->getVariable<int>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentFunctionConditionError1) {
    // Add required agent function
    ms->addFuncCdn(GetAgentVariable, AgentFunctionConditionError1);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, AgentFunctionConditionError2) {
    // Add required agent function
    ms->addFuncCdn(GetAgentVariable, AgentFunctionConditionError2);
    // Test Something
    ms->run(1);
}

// Test error if agent birth/death not enabled
FLAMEGPU_AGENT_FUNCTION(AgentBirthMock1, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<int>("int", 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentBirthMock2, MsgNone, MsgNone) {
    FLAMEGPU->agent_out.setVariable<int, 2>("int", 0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentDeathMock, MsgNone, MsgNone) {
    return DEAD;
}
TEST_F(DeviceExceptionTest, AgentBirthDisabled1) {
    // Add required agent function
    ms->addFunc(AgentBirthMock1);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, AgentBirthDisabled2) {
    // Add required agent function
    ms->addFunc(AgentBirthMock2);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, AgentDeathDisabled) {
    // Add required agent function
    ms->addFunc(AgentDeathMock);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutGetID, MsgNone, MsgNone) {
    FLAMEGPU->setVariable<id_t>("id_other", FLAMEGPU->agent_out.getID());
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentID_DeviceBirth) {
    // Attempt to get ID of device agent, whilst agent birth is not enabled
    ms->addFunc(AgentOutGetID);
    // Test Something
    ms->run(1);
}

}  // namespace test_device_exception
}  // namespace flamegpu
