#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

/* 
    Test suite to check that c++ models can be build using a range of namespace appraoches.
    Uses several anonymous namespaces to scope namespacing appropraitely. Would likely make more sense to split the files.
 */

namespace test_namespaces {
const unsigned int AGENT_COUNT = 32;

// Explicit
namespace {

// Test using explicitly namespaced types. ie. flamegpu::ALIVE
FLAMEGPU_AGENT_FUNCTION(msg_out_func_explicit, flamegpu::MsgNone, flamegpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(msg_in_func_explicit, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return flamegpu::ALIVE;
}
TEST(CXXNamespaceTest, AgentFunctionsExplicit) {
    flamegpu::ModelDescription m("model");
    flamegpu::MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    flamegpu::AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    flamegpu::AgentFunctionDescription& fo = a.newFunction("msg_out_func_explicit", msg_out_func_explicit);
    fo.setMessageOutput(msg);
    flamegpu::AgentFunctionDescription& fi = a.newFunction("msg_in_func_explicit", msg_in_func_explicit);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    flamegpu::AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (flamegpu::AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    flamegpu::LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    flamegpu::LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    flamegpu::CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (flamegpu::AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}

}  // namespace



// Test via the using declarations, i.e. using flamegpu::ALIVE
namespace {

using flamegpu::ALIVE;
using flamegpu::MsgNone;
using flamegpu::MsgBruteForce;
using flamegpu::ModelDescription;
using flamegpu::MsgBruteForce;
using flamegpu::AgentDescription;
using flamegpu::AgentFunctionDescription;
using flamegpu::AgentVector;
using flamegpu::LayerDescription;
using flamegpu::CUDASimulation;


FLAMEGPU_AGENT_FUNCTION(msg_out_func_declaration, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(msg_in_func_declaration, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
}
TEST(CXXNamespaceTest, AgentFunctionsDeclaration) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newFunction("msg_out_func_declaration", msg_out_func_declaration);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newFunction("msg_in_func_declaration", msg_in_func_declaration);
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

}  // namespace

// Test via the using directives (which outside of rtc will be a lint failure), i.e. using namespace flamegpu
// @note - this will make cpplint very angry, might need to place this test in a subfolder with a .cpplint config file that disables the namespace check?

namespace {

using namespace flamegpu;

FLAMEGPU_AGENT_FUNCTION(msg_out_func_directive, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(msg_in_func_directive, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
}

TEST(CXXNamespaceTest, AgentFunctionsDirective) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newFunction("msg_out_func_directive", msg_out_func_directive);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newFunction("msg_in_func_directive", msg_in_func_directive);
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

}  // namespace


// Place within the flamegpu namespace.
/* namespace {

namespace flamegpu {
FLAMEGPU_AGENT_FUNCTION(msg_out_func_named, MsgNone, MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(msg_in_func_named, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
}
TEST(CXXNamespaceTest, AgentFunctionsNamed) {
    ModelDescription m("model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newFunction("msg_out_func_named", msg_out_func_named);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newFunction("msg_in_func_named", msg_in_func_named);
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

}  // namespace flamegpu
}  // namespace 
 */

// Test aliasing the flamegpu namespace.
namespace {

namespace fgpu = flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_out_func_alias, fgpu::MsgNone, fgpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return fgpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(msg_in_func_alias, fgpu::MsgBruteForce, fgpu::MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return fgpu::ALIVE;
}
TEST(CXXNamespaceTest, AgentFunctionsAlias) {
    fgpu::ModelDescription m("model");
    fgpu::MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    fgpu::AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    fgpu::AgentFunctionDescription& fo = a.newFunction("msg_out_func_alias", msg_out_func_alias);
    fo.setMessageOutput(msg);
    fgpu::AgentFunctionDescription& fi = a.newFunction("msg_in_func_alias", msg_in_func_alias);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    fgpu::AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (fgpu::AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    fgpu::LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    fgpu::LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    fgpu::CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (fgpu::AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}

}  // namespace

namespace {

// Test aliasing the flamegpu namespace, but accessing using a mix (to check that message type comparisons work)
namespace fgpu = flamegpu;
FLAMEGPU_AGENT_FUNCTION(msg_out_func_alias_mixed, fgpu::MsgNone, fgpu::MsgBruteForce) {
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return fgpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(msg_in_func_alias_mixed, flamegpu::MsgBruteForce, flamegpu::MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return fgpu::ALIVE;
}
TEST(CXXNamespaceTest, AgentFunctionsAliasMixed) {
    fgpu::ModelDescription m("model");
    flamegpu::MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    fgpu::AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    flamegpu::AgentFunctionDescription& fo = a.newFunction("msg_out_func_alias_mixed", msg_out_func_alias_mixed);
    fo.setMessageOutput(msg);
    fgpu::AgentFunctionDescription& fi = a.newFunction("msg_in_func_alias_mixed", msg_in_func_alias_mixed);
    fi.setMessageInput(msg);
    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    flamegpu::AgentVector pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (fgpu::AgentVector::Agent ai : pop) {
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    flamegpu::LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    fgpu::LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    flamegpu::CUDASimulation c(m);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (fgpu::AgentVector::Agent ai : pop) {
        ASSERT_EQ(ai.getVariable<int>("sum"), sum);
    }
}
}  // namespace


}  // namespace test_namespaces
