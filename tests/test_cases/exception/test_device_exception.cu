
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
    void run(int steps = 2, bool test_error_string = false) {
        // CudaModel must be declared here
        // As the initial call to constructor fixes the agent population
        // This means if we haven't called model.newAgent(agent) first
        CUDASimulation cudaSimulation(model);
        cudaSimulation.SimulationConfig().steps = steps;
        // This fails as agentMap is empty
        AgentVector population(agent, 1);
        cudaSimulation.setPopulationData(population);
        if (test_error_string) {
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
        } else {
            EXPECT_THROW(cudaSimulation.simulate(), exception::DeviceError);
        }
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
    template<typename Message, typename T>
    void addMessageOutFunc(T func) {
        typename Message::Description &message = model.newMessage<Message>("message");
        message.template newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMessageS2DOutFunc(T func) {
        MessageSpatial2D::Description &message = model.newMessage<MessageSpatial2D>("message");
        message.setMin(-1, -1);
        message.setMax(1, 1);
        message.setRadius(1);
        message.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMessageS3DOutFunc(T func) {
        MessageSpatial3D::Description &message = model.newMessage<MessageSpatial3D>("message");
        message.setMin(-1, -1, -2);
        message.setMax(1, 1, 1);
        message.setRadius(1);
        message.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMessageA1DOutFunc(T func) {
        MessageArray::Description &message = model.newMessage<MessageArray>("message");
        message.setLength(10);
        message.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMessageA2DOutFunc(T func) {
        MessageArray2D::Description &message = model.newMessage<MessageArray2D>("message");
        message.setDimensions(10, 10);
        message.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMessageA3DOutFunc(T func) {
        MessageArray3D::Description &message = model.newMessage<MessageArray3D>("message");
        message.setDimensions(10, 10, 10);
        message.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }
    template<typename T>
    void addMessageBucketOutFunc(T func) {
        MessageBucket::Description &message = model.newMessage<MessageBucket>("message");
        message.setBounds(0, 1023);
        message.newVariable<int>("int");
        auto &fn = agent.newFunction("name", func);
        fn.setMessageOutput("message");
        model.newLayer().addAgentFunction(func);
    }

    template<typename Message, typename T1, typename T2>
    void addMessageInFunc(T1 out_func, T2 in_func) {
        auto &message = model.newMessage<Message>("message");
        message.template newVariable<int>("int");
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
    void addMessageS2DInFunc(T1 out_func, T2 in_func) {
        MessageSpatial2D::Description &message = model.newMessage<MessageSpatial2D>("message");
        message.setMin(-1, -1);
        message.setMax(1, 1);
        message.setRadius(1);
        message.template newVariable<int>("int");
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
    void addMessageS3DInFunc(T1 out_func, T2 in_func) {
        MessageSpatial3D::Description &message = model.newMessage<MessageSpatial3D>("message");
        message.setMin(-1, -1, -1);
        message.setMax(1, 1, 1);
        message.setRadius(1);
        message.template newVariable<int>("int");
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
    void addMessageA1DInFunc(T1 out_func, T2 in_func) {
        MessageArray::Description &message = model.newMessage<MessageArray>("message");
        message.setLength(10);
        message.template newVariable<int>("int");
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
    void addMessageA2DInFunc(T1 out_func, T2 in_func) {
        MessageArray2D::Description &message = model.newMessage<MessageArray2D>("message");
        message.setDimensions(10, 10);
        message.template newVariable<int>("int");
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
    void addMessageA3DInFunc(T1 out_func, T2 in_func) {
        MessageArray3D::Description &message = model.newMessage<MessageArray3D>("message");
        message.setDimensions(10, 10, 10);
        message.template newVariable<int>("int");
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
    void addMessageBucketInFunc(T1 out_func, T2 in_func) {
        MessageBucket::Description &message = model.newMessage<MessageBucket>("message");
        message.setBounds(0, 1023);
        message.template newVariable<int>("int");
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
    AgentDescription agent;
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
FLAMEGPU_AGENT_FUNCTION(GetUnknownAgentVariable, MessageNone, MessageNone) {
    FLAMEGPU->getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetUnknownAgentVariable) {
    // Add required agent function
    ms->addFunc(GetUnknownAgentVariable);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(GetAgentVariableBadType, MessageNone, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(GetUnknownAgentVariableArray, MessageNone, MessageNone) {
    FLAMEGPU->getVariable<int, 3>("nope", 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetUnknownAgentVariableArray) {
    // Add required agent function
    ms->addFunc(GetUnknownAgentVariableArray);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(GetAgentVariableArrayBadType, MessageNone, MessageNone) {
    FLAMEGPU->getVariable<double, 3>("array", 0);   // Note type checking only confirms size currently
    return ALIVE;
}
TEST_F(DeviceExceptionTest, GetAgentVariableArrayBadType) {
    // Add required agent function
    ms->addFunc(GetAgentVariableArrayBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(GetAgentVariableArrayOutOfRange, MessageNone, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(SetUnknownAgentVariable, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetUnknownAgentVariable) {
    // Add required agent function
    ms->addFunc(SetUnknownAgentVariable);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(SetAgentVariableBadType, MessageNone, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(SetUnknownAgentVariableArray, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<int, 3>("nope", 0, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetUnknownAgentVariableArray) {
    // Add required agent function
    ms->addFunc(SetUnknownAgentVariableArray);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(SetAgentVariableArrayBadType, MessageNone, MessageNone) {
    FLAMEGPU->setVariable<double, 3>("array", 0, 12.0);  // Note type checking only confirms size currently
    return ALIVE;
}
TEST_F(DeviceExceptionTest, SetAgentVariableArrayBadType) {
    // Add required agent function
    ms->addFunc(SetAgentVariableArrayBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(SetAgentVariableArrayOutOfRange, MessageNone, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(AgentRandomUniformInvalidRange1, MessageNone, MessageNone) {
    FLAMEGPU->random.uniform<int>(5, 4);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentRandomUniformInvalidRange1) {
    // Add required agent function
    ms->addFunc(AgentRandomUniformInvalidRange1);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentRandomUniformInvalidRange2, MessageNone, MessageNone) {
    FLAMEGPU->random.uniform<int64_t>(5, 4);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentRandomUniformInvalidRange2) {
    // Add required agent function
    ms->addFunc(AgentRandomUniformInvalidRange2);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentRandomUniformInvalidRange3, MessageNone, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(DeviceEnvironmentGetUnknownProperty, MessageNone, MessageNone) {
    FLAMEGPU->environment.getProperty<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, DeviceEnvironmentGetUnknownProperty) {
    // Add required agent function
    ms->addFunc(DeviceEnvironmentGetUnknownProperty);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(DeviceEnvironmentGetBadType, MessageNone, MessageNone) {
    FLAMEGPU->environment.getProperty<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, DeviceEnvironmentGetBadType) {
    // Add required agent function
    ms->addFunc(DeviceEnvironmentGetBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(DeviceEnvironmentGetOutOfRange, MessageNone, MessageNone) {
    FLAMEGPU->environment.getProperty<int, 2>("array", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, DeviceEnvironmentGetOutOfRange) {
    // Add required agent function
    ms->addFunc(DeviceEnvironmentGetOutOfRange);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(DeviceEnvironmentWrongLength, MessageNone, MessageNone) {
    FLAMEGPU->environment.getProperty<int, 3>("array", 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, DeviceEnvironmentWrongLength) {
    // Add required agent function
    ms->addFunc(DeviceEnvironmentWrongLength);
    // Test Something
    ms->run(1);
}

// AgentOut::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableUnknown, MessageNone, MessageNone) {
    FLAMEGPU->agent_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableUnknown) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableBadType, MessageNone, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayUnknown, MessageNone, MessageNone) {
    FLAMEGPU->agent_out.setVariable<int, 2>("nope", 0, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayUnknown) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayBadType, MessageNone, MessageNone) {
    FLAMEGPU->agent_out.setVariable<double, 2>("array", 0, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayBadType) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayBadType);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayOutOfRange1, MessageNone, MessageNone) {
    FLAMEGPU->agent_out.setVariable<int, 2>("array", 2, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayOutOfRange1) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayOutOfRange1);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(AgentOutVariableArrayOutOfRange2, MessageNone, MessageNone) {
    FLAMEGPU->agent_out.setVariable<int, 3>("array", 2, 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentOutVariableArrayOutOfRange2) {
    // Add required agent function
    ms->addAgentOutFunc(AgentOutVariableArrayOutOfRange2);
    // Test Something
    ms->run(1);
}

// MessageBruteForce::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageBruteForceOutVariableUnknown, MessageNone, MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBruteForceOutVariableUnknown) {
    // Add required agent function
    ms->addMessageOutFunc<MessageBruteForce>(MessageBruteForceOutVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageBruteForceOutVariableBadType, MessageNone, MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<double>("int", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBruteForceOutVariableBadType) {
    // Add required agent function
    ms->addMessageOutFunc<MessageBruteForce>(MessageBruteForceOutVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageBruteForce::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageBruteForceDefaultOut, MessageNone, MessageBruteForce) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageBruteForceInVariableUnknown, MessageBruteForce, MessageNone) {
    for (auto m : FLAMEGPU->message_in) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBruteForceInVariableUnknown) {
    // Add required agent function
    ms->addMessageInFunc<MessageBruteForce>(MessageBruteForceDefaultOut, MessageBruteForceInVariableUnknown);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MessageBruteForceInVariableBadType, MessageBruteForce, MessageNone) {
    for (auto m : FLAMEGPU->message_in) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBruteForceInVariableBadType) {
    // Add required agent function
    ms->addMessageInFunc<MessageBruteForce>(MessageBruteForceDefaultOut, MessageBruteForceInVariableBadType);
    // Test Something
    ms->run(1);
}


// MessageSpatial2D::Out::setVariable<T, N>() (These should be identical to MessageBruteForce due to the object being inherited)
FLAMEGPU_AGENT_FUNCTION(MessageSpatial2DOutVariableUnknown, MessageNone, MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial2DOutVariableUnknown) {
    // Add required agent function
    ms->addMessageS2DOutFunc(MessageSpatial2DOutVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageSpatial2DOutVariableBadType, MessageNone, MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<double>("int", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial2DOutVariableBadType) {
    // Add required agent function
    ms->addMessageS2DOutFunc(MessageSpatial2DOutVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageSpatial2D::In::Filter::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageSpatial2DDefaultOut, MessageNone, MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setLocation(0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageSpatial2DInVariableUnknown, MessageSpatial2D, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0)) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial2DInVariableUnknown) {
    // Add required agent function
    ms->addMessageS2DInFunc(MessageSpatial2DDefaultOut, MessageSpatial2DInVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageSpatial2DInVariableBadType, MessageSpatial2D, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0)) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial2DInVariableBadType) {
    // Add required agent function
    ms->addMessageS2DInFunc(MessageSpatial2DDefaultOut, MessageSpatial2DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageSpatial3D::Out::setVariable<T, N>() (These should be identical to MessageBruteForce due to the object being inherited)
FLAMEGPU_AGENT_FUNCTION(MessageSpatial3DOutVariableUnknown, MessageNone, MessageSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial3DOutVariableUnknown) {
    // Add required agent function
    ms->addMessageS3DOutFunc(MessageSpatial3DOutVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageSpatial3DOutVariableBadType, MessageNone, MessageSpatial3D) {
    FLAMEGPU->message_out.setVariable<double>("int", 2);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial3DOutVariableBadType) {
    // Add required agent function
    ms->addMessageS3DOutFunc(MessageSpatial3DOutVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageSpatial3D::In::Filter::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageSpatial3DDefaultOut, MessageNone, MessageSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setLocation(0, 0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageSpatial3DInVariableUnknown, MessageSpatial3D, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0, 0)) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial3DInVariableUnknown) {
    // Add required agent function
    ms->addMessageS3DInFunc(MessageSpatial3DDefaultOut, MessageSpatial3DInVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageSpatial3DInVariableBadType, MessageSpatial3D, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0 , 0, 0)) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageSpatial3DInVariableBadType) {
    // Add required agent function
    ms->addMessageS3DInFunc(MessageSpatial3DDefaultOut, MessageSpatial3DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageArray::Out::setIndex()
FLAMEGPU_AGENT_FUNCTION(MessageArrayOutIndexOutOfRange, MessageNone, MessageArray) {
    FLAMEGPU->message_out.setIndex(10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArrayOutIndexOutOfRange) {
    // Add required agent function
    ms->addMessageA1DOutFunc(MessageArrayOutIndexOutOfRange);
    // Test Something
    ms->run(1);
}

// MessageArray::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageArrayOutUnknownVariable, MessageNone, MessageArray) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArrayOutUnknownVariable) {
    // Add required agent function
    ms->addMessageA1DOutFunc(MessageArrayOutUnknownVariable);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageArrayVariableBadType, MessageNone, MessageArray) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArrayVariableBadType) {
    // Add required agent function
    ms->addMessageA1DOutFunc(MessageArrayVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageArray::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageArrayDefaultOut, MessageNone, MessageArray) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setIndex(0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageArrayInVariableUnknown, MessageArray, MessageNone) {
    FLAMEGPU->message_in.at(0).getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArrayInVariableUnknown) {
    // Add required agent function
    ms->addMessageA1DInFunc(MessageArrayDefaultOut, MessageArrayInVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageArrayInVariableBadType, MessageArray, MessageNone) {
    FLAMEGPU->message_in.at(0).getVariable<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArrayInVariableBadType) {
    // Add required agent function
    ms->addMessageA1DInFunc(MessageArrayDefaultOut, MessageArrayInVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageArray::In::operator()()
FLAMEGPU_AGENT_FUNCTION(MessageArrayInVariableBadRadius, MessageArray, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0, 0)) {
        m.getVariable<int>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArrayInVariableBadRadius) {
    // Add required agent function
    ms->addMessageA1DInFunc(MessageArrayDefaultOut, MessageArrayInVariableBadRadius);
    // Test Something
    ms->run(1);
}

// MessageArray::In::at()
FLAMEGPU_AGENT_FUNCTION(MessageArrayInVariableIndexOutOfBounds, MessageArray, MessageNone) {
    FLAMEGPU->message_in.at(10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArrayInVariableIndexOutOfBounds) {
    // Add required agent function
    ms->addMessageA1DInFunc(MessageArrayDefaultOut, MessageArrayInVariableIndexOutOfBounds);
    // Test Something
    ms->run(1);
}

// MessageArray2D::Out::setIndex()
FLAMEGPU_AGENT_FUNCTION(MessageArray2DOutIndexOutOfRange, MessageNone, MessageArray2D) {
    FLAMEGPU->message_out.setIndex(10, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DOutIndexOutOfRange) {
    // Add required agent function
    ms->addMessageA2DOutFunc(MessageArray2DOutIndexOutOfRange);
    // Test Something
    ms->run(1);
}

// MessageArray2D::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageArray2DOutUnknownVariable, MessageNone, MessageArray2D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DOutUnknownVariable) {
    // Add required agent function
    ms->addMessageA2DOutFunc(MessageArray2DOutUnknownVariable);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageArray2DVariableBadType, MessageNone, MessageArray2D) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DVariableBadType) {
    // Add required agent function
    ms->addMessageA2DOutFunc(MessageArray2DVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageArray2D::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageArray2DDefaultOut, MessageNone, MessageArray2D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setIndex(0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageArray2DInVariableUnknown, MessageArray2D, MessageNone) {
    FLAMEGPU->message_in.at(0, 0).getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DInVariableUnknown) {
    // Add required agent function
    ms->addMessageA2DInFunc(MessageArray2DDefaultOut, MessageArray2DInVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageArray2DInVariableBadType, MessageArray2D, MessageNone) {
    FLAMEGPU->message_in.at(0, 0).getVariable<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DInVariableBadType) {
    // Add required agent function
    ms->addMessageA2DInFunc(MessageArray2DDefaultOut, MessageArray2DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageArray2D::In::operator()()
FLAMEGPU_AGENT_FUNCTION(MessageArray2DInVariableBadRadius, MessageArray2D, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0, 0, 0)) {
        m.getVariable<int>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DInVariableBadRadius) {
    // Add required agent function
    ms->addMessageA2DInFunc(MessageArray2DDefaultOut, MessageArray2DInVariableBadRadius);
    // Test Something
    ms->run(1);
}

// MessageArray2D::In::at()
FLAMEGPU_AGENT_FUNCTION(MessageArray2DInVariableIndexOutOfBoundsX, MessageArray2D, MessageNone) {
    FLAMEGPU->message_in.at(10, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DInVariableIndexOutOfBoundsX) {
    // Add required agent function
    ms->addMessageA2DInFunc(MessageArray2DDefaultOut, MessageArray2DInVariableIndexOutOfBoundsX);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MessageArray2DInVariableIndexOutOfBoundsY, MessageArray2D, MessageNone) {
    FLAMEGPU->message_in.at(0, 10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray2DInVariableIndexOutOfBoundsY) {
    // Add required agent function
    ms->addMessageA2DInFunc(MessageArray2DDefaultOut, MessageArray2DInVariableIndexOutOfBoundsY);
    // Test Something
    ms->run(1);
}

// MessageArray3D::Out::setIndex()
FLAMEGPU_AGENT_FUNCTION(MessageArray3DOutIndexOutOfRange, MessageNone, MessageArray3D) {
    FLAMEGPU->message_out.setIndex(10, 0, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DOutIndexOutOfRange) {
    // Add required agent function
    ms->addMessageA3DOutFunc(MessageArray3DOutIndexOutOfRange);
    // Test Something
    ms->run(1);
}

// MessageArray3D::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageArray3DOutUnknownVariable, MessageNone, MessageArray3D) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DOutUnknownVariable) {
    // Add required agent function
    ms->addMessageA3DOutFunc(MessageArray3DOutUnknownVariable);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageArray3DVariableBadType, MessageNone, MessageArray3D) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DVariableBadType) {
    // Add required agent function
    ms->addMessageA3DOutFunc(MessageArray3DVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageArray3D::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageArray3DDefaultOut, MessageNone, MessageArray3D) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setIndex(0, 0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageArray3DInVariableUnknown, MessageArray3D, MessageNone) {
    FLAMEGPU->message_in.at(0, 0, 0).getVariable<int>("nope");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DInVariableUnknown) {
    // Add required agent function
    ms->addMessageA3DInFunc(MessageArray3DDefaultOut, MessageArray3DInVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageArray3DInVariableBadType, MessageArray3D, MessageNone) {
    FLAMEGPU->message_in.at(0, 0, 0).getVariable<double>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DInVariableBadType) {
    // Add required agent function
    ms->addMessageA3DInFunc(MessageArray3DDefaultOut, MessageArray3DInVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageArray3D::In::operator()()
FLAMEGPU_AGENT_FUNCTION(MessageArray3DInVariableBadRadius, MessageArray3D, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0, 0, 0, 0)) {
        m.getVariable<int>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DInVariableBadRadius) {
    // Add required agent function
    ms->addMessageA3DInFunc(MessageArray3DDefaultOut, MessageArray3DInVariableBadRadius);
    // Test Something
    ms->run(1);
}

// MessageArray3D::In::at()
FLAMEGPU_AGENT_FUNCTION(MessageArray3DInVariableIndexOutOfBoundsX, MessageArray3D, MessageNone) {
    FLAMEGPU->message_in.at(10, 0, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DInVariableIndexOutOfBoundsX) {
    // Add required agent function
    ms->addMessageA3DInFunc(MessageArray3DDefaultOut, MessageArray3DInVariableIndexOutOfBoundsX);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MessageArray3DInVariableIndexOutOfBoundsY, MessageArray3D, MessageNone) {
    FLAMEGPU->message_in.at(0, 10, 0);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DInVariableIndexOutOfBoundsY) {
    // Add required agent function
    ms->addMessageA3DInFunc(MessageArray3DDefaultOut, MessageArray3DInVariableIndexOutOfBoundsY);
    // Test Something
    ms->run(1);
}
FLAMEGPU_AGENT_FUNCTION(MessageArray3DInVariableIndexOutOfBoundsZ, MessageArray3D, MessageNone) {
    FLAMEGPU->message_in.at(0, 0, 10);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageArray3DInVariableIndexOutOfBoundsZ) {
    // Add required agent function
    ms->addMessageA3DInFunc(MessageArray3DDefaultOut, MessageArray3DInVariableIndexOutOfBoundsZ);
    // Test Something
    ms->run(1);
}

// MessageBucket::Out::setVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageBucketOutUnknownVariable, MessageNone, MessageBucket) {
    FLAMEGPU->message_out.setVariable<int>("nope", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBucketOutUnknownVariable) {
    // Add required agent function
    ms->addMessageBucketOutFunc(MessageBucketOutUnknownVariable);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketVariableBadType, MessageNone, MessageBucket) {
    FLAMEGPU->message_out.setVariable<double>("int", 11);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBucketVariableBadType) {
    // Add required agent function
    ms->addMessageBucketOutFunc(MessageBucketVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageBucket::Out::setKey()
FLAMEGPU_AGENT_FUNCTION(MessageBucketOutBadKey1, MessageNone, MessageBucket) {
    FLAMEGPU->message_out.setKey(-1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketOutBadKey2, MessageNone, MessageBucket) {
    FLAMEGPU->message_out.setKey(1024);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBucketOutBadKey1) {
    // Add required agent function
    ms->addMessageBucketOutFunc(MessageBucketOutBadKey1);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MessageBucketOutBadKey2) {
    // Add required agent function
    ms->addMessageBucketOutFunc(MessageBucketOutBadKey2);
    // Test Something
    ms->run(1);
}

// MessageBucket::Message::getVariable<T, N>()
FLAMEGPU_AGENT_FUNCTION(MessageBucketDefaultOut, MessageNone, MessageBucket) {
    FLAMEGPU->message_out.setVariable<int>("int", 12);
    FLAMEGPU->message_out.setKey(0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketInVariableUnknown, MessageBucket, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0)) {
        m.getVariable<int>("nope");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBucketInVariableUnknown) {
    // Add required agent function
    ms->addMessageBucketInFunc(MessageBucketDefaultOut, MessageBucketInVariableUnknown);
    // Test Something
    ms->run(1, true);
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketInVariableBadType, MessageBucket, MessageNone) {
    for (auto m : FLAMEGPU->message_in(0)) {
        m.getVariable<double>("int");
    }
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBucketInVariableBadType) {
    // Add required agent function
    ms->addMessageBucketInFunc(MessageBucketDefaultOut, MessageBucketInVariableBadType);
    // Test Something
    ms->run(1);
}

// MessageBucket::In::operator()(key)
// MessageBucket::In::operator()(beginKey, endKey)
FLAMEGPU_AGENT_FUNCTION(MessageBucketInBadKey1, MessageBucket, MessageNone) {
    FLAMEGPU->message_in(-1);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketInBadKey2, MessageBucket, MessageNone) {
    FLAMEGPU->message_in(1024);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketInBadKey3, MessageBucket, MessageNone) {
    FLAMEGPU->message_in(-1, 1023);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketInBadKey4, MessageBucket, MessageNone) {
    FLAMEGPU->message_in(0, 1025);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(MessageBucketInBadKey5, MessageBucket, MessageNone) {
    FLAMEGPU->message_in(100, 5);
    return ALIVE;
}
TEST_F(DeviceExceptionTest, MessageBucketInBadKey1) {
    // Add required agent function
    ms->addMessageBucketInFunc(MessageBucketDefaultOut, MessageBucketInBadKey1);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MessageBucketInBadKey2) {
    // Add required agent function
    ms->addMessageBucketInFunc(MessageBucketDefaultOut, MessageBucketInBadKey2);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MessageBucketInBadKey3) {
    // Add required agent function
    ms->addMessageBucketInFunc(MessageBucketDefaultOut, MessageBucketInBadKey3);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MessageBucketInBadKey4) {
    // Add required agent function
    ms->addMessageBucketInFunc(MessageBucketDefaultOut, MessageBucketInBadKey4);
    // Test Something
    ms->run(1);
}
TEST_F(DeviceExceptionTest, MessageBucketInBadKey5) {
    // Add required agent function
    ms->addMessageBucketInFunc(MessageBucketDefaultOut, MessageBucketInBadKey5);
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
FLAMEGPU_AGENT_FUNCTION(GetAgentVariable, MessageNone, MessageNone) {
    FLAMEGPU->getVariable<int>("int");
    return ALIVE;
}
TEST_F(DeviceExceptionTest, AgentFunctionConditionError1) {
    // Add required agent function
    ms->addFuncCdn(GetAgentVariable, AgentFunctionConditionError1);
    // Test Something
    ms->run(1, true);
}
TEST_F(DeviceExceptionTest, AgentFunctionConditionError2) {
    // Add required agent function
    ms->addFuncCdn(GetAgentVariable, AgentFunctionConditionError2);
    // Test Something
    ms->run(1, true);
}

// Test error if agent birth/death not enabled
FLAMEGPU_AGENT_FUNCTION(AgentBirthMock1, MessageNone, MessageNone) {
    FLAMEGPU->agent_out.setVariable<int>("int", 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentBirthMock2, MessageNone, MessageNone) {
    FLAMEGPU->agent_out.setVariable<int, 2>("int", 0, 0);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(AgentDeathMock, MessageNone, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(AgentOutGetID, MessageNone, MessageNone) {
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
