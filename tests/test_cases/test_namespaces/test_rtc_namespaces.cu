#include "flamegpu/flame_api.h"

#include "gtest/gtest.h"

/* 
    Test suite to check that RTC behaves corrrectly with the various methods of using namespaces.
 */

namespace flamegpu {
namespace test_rtc_namespaces {
const unsigned int AGENT_COUNT = 32;

// All agent functions require specialising the Message type, regardless of wheter it is None or not, so not point testing without message output being involved.

// Test using explicitly namespaced types. ie. flamegpu::ALIVE
const char* msg_out_func_explicit = R"###(
FLAMEGPU_AGENT_FUNCTION(msg_out_func_explicit, flamegpu::MsgNone, flamegpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return flamegpu::ALIVE;
}
)###";

const char* msg_in_func_explicit = R"###(
FLAMEGPU_AGENT_FUNCTION(msg_in_func_explicit, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return flamegpu::ALIVE;
}
)###";


TEST(RTCNamespaceTest, AgentFunctionsExplicit) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("msg_out_func_explicit", msg_out_func_explicit);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("msg_in_func_explicit", msg_in_func_explicit);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}

// Test via the using declarations, i.e. using flamegpu::ALIVE
const char* msg_out_func_declaration = R"###(
using flamegpu::ALIVE;
using flamegpu::MsgNone;
using flamegpu::MsgBruteForce;
FLAMEGPU_AGENT_FUNCTION(msg_out_func_declaration, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
)###";

const char* msg_in_func_declaration = R"###(
using flamegpu::ALIVE;
using flamegpu::MsgNone;
using flamegpu::MsgBruteForce;
FLAMEGPU_AGENT_FUNCTION(msg_in_func_declaration, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
}
)###";


TEST(RTCNamespaceTest, AgentFunctionsDeclaration) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("msg_out_func_declaration", msg_out_func_declaration);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("msg_in_func_declaration", msg_in_func_declaration);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}

// Test via the using directives (which outside of rtc will be a lint failure), i.e. using namespace flamegpu
const char* msg_out_func_directive = R"###(
using namespace flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_out_func_directive, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
)###";

const char* msg_in_func_directive = R"###(
using namespace flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_in_func_directive, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
}
)###";


TEST(RTCNamespaceTest, AgentFunctionsDirective) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("msg_out_func_directive", msg_out_func_directive);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("msg_in_func_directive", msg_in_func_directive);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}



/* Test by embedding within the flamegpu namespace. (Users shouldn't be doing this, but it is an option.)
This option requires additional steps for NVRTC to correctly compile things.
https://docs.nvidia.com/cuda/nvrtc/index.html#accessing-lowered-names
__global__, __constant__ and __device__ in namespaces need to be expliciltly made available via  nvrtcAddNameExpression
This is a lot of additional complexity to support edge case behaviour, that we do not want to encourage anyway, so disabling this test. */

/* const char* msg_out_func_named = R"###(
namespace flamegpu {
FLAMEGPU_AGENT_FUNCTION(msg_out_func_named, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
}  // namespace flamegpu
)###";

const char* msg_in_func_named = R"###(
namespace flamegpu {
FLAMEGPU_AGENT_FUNCTION(msg_in_func_named, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
}
}  // namespace flamegpu
)###";


TEST(RTCNamespaceTest, AgentFunctionsNamed) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("msg_out_func_named", msg_out_func_named);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("msg_in_func_named", msg_in_func_named);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}
 */

// Test using an aliased namespace
const char* msg_out_func_alias = R"###(
namespace fgpu = flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_out_func_alias, fgpu::MsgNone, fgpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return fgpu::ALIVE;
}
)###";

const char* msg_in_func_alias = R"###(
namespace fgpu = flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_in_func_alias, fgpu::MsgBruteForce, fgpu::MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return fgpu::ALIVE;
}
)###";


TEST(RTCNamespaceTest, AgentFunctionsAlias) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("msg_out_func_alias", msg_out_func_alias);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("msg_in_func_alias", msg_in_func_alias);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}

// Test aliasing the flamegpu namespace, but accessing using a mix (to check that message type comparisons work)
const char* msg_out_func_alias_mixed = R"###(
namespace fgpu = flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_out_func_alias_mixed, fgpu::MsgNone, fgpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return fgpu::ALIVE;
}
)###";

const char* msg_in_func_alias_mixed = R"###(
namespace fgpu = flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_in_func_alias_mixed, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return fgpu::ALIVE;
}
)###";


TEST(RTCNamespaceTest, AgentFunctionsAliasMixed) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("msg_out_func_alias_mixed", msg_out_func_alias_mixed);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("msg_in_func_alias_mixed", msg_in_func_alias_mixed);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}


}  // namespace test_rtc_namespaces
}  // namespace flamegpu
