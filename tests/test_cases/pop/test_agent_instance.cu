#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


TEST(AgentInstanceTest, constructor) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    AgentInstance ai(agent);
    // New AgentInstance is default init
    ASSERT_EQ(ai.getVariable<int>("int"), 1);
    ASSERT_EQ(ai.getVariable<unsigned int>("uint"), 2u);
}
TEST(AgentInstanceTest, copy_constructor) {
  ModelDescription model("model");
  AgentDescription& agent = model.newAgent("agent");
  agent.newVariable<int>("int", 1);
  agent.newVariable<unsigned int, 3>("uint3", {2u, 3u, 4u});
  const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
  // Copying an agent instance retains the values
  AgentInstance ai(agent);
  ai.setVariable<int>("int", 12);
  ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
  AgentInstance ai2(ai);
  ASSERT_EQ(ai2.getVariable<int>("int"), 12);
  auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
  ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
  // Copying an agent instance from an AgentVector::Agent retains values
  AgentVector av(agent, 1);
  AgentVector::Agent ava = av.front();
  ava.setVariable<int>("int", 12);
  ava.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
  AgentInstance ai3(ava);
  ASSERT_EQ(ai3.getVariable<int>("int"), 12);
  auto ai2_uint3_check2 = ai3.getVariable<unsigned int, 3>("uint3");
  ASSERT_EQ(ai2_uint3_check2, ai_uint3_ref);
}
TEST(AgentInstanceTest, move_constructor) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int, 3>("uint3", { 2u, 3u, 4u });
    const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
    // Moving an agent instance retains the values
    AgentInstance ai(agent);
    ai.setVariable<int>("int", 12);
    ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
    AgentInstance ai2(std::move(ai));
    ASSERT_EQ(ai2.getVariable<int>("int"), 12);
    auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
}
TEST(AgentInstanceTest, copy_assignment_operator) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    AgentDescription& agent2 = model.newAgent("agent2");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int, 3>("uint3", { 2u, 3u, 4u });
    const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
    // Copying an agent instance retains the values
    AgentInstance ai(agent);
    ai.setVariable<int>("int", 12);
    ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
    AgentInstance ai2(agent2);
    ai2 = ai;
    ASSERT_EQ(ai2.getVariable<int>("int"), 12);
    auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
    // Copying an agent instance from an AgentVector::Agent retains values
    AgentVector av(agent, 1);
    AgentVector::Agent ava = av.front();
    ava.setVariable<int>("int", 12);
    ava.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
    AgentInstance ai3(agent2);
    ai3 = ava;
    ASSERT_EQ(ai3.getVariable<int>("int"), 12);
    auto ai2_uint3_check2 = ai3.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check2, ai_uint3_ref);
}
TEST(AgentInstanceTest, move_assignment_operator) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    AgentDescription& agent2 = model.newAgent("agent2");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int, 3>("uint3", { 2u, 3u, 4u });
    const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
    // Moving an agent instance retains the values
    AgentInstance ai(agent);
    ai.setVariable<int>("int", 12);
    ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
    AgentInstance ai2(agent2);
    ai2 = std::move(ai);
    ASSERT_EQ(ai2.getVariable<int>("int"), 12);
    auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
}
TEST(AgentInstanceTest, getsetVariable) {
    const unsigned int i = 15;  // This is a stripped down version of AgentVectorTest::AgentVector_Agent
    // Test correctness of AgentVector getVariableType
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", 12u);
    agent.newVariable<int, 3>("int3", { 2, 3, 4 });
    agent.newVariable<int, 2>("int2", { 5, 6 });
    agent.newVariable<float>("float", 15.0f);

    // Create pop, variables are as expected
    AgentInstance ai(agent);
    const std::array<int, 3> int3_ref = { 2, 3, 4 };
    {
        ASSERT_EQ(ai.getVariable<unsigned int>("uint"), 12u);
        const std::array<int, 3> int3_check = ai.getVariable<int, 3>("int3");
        ASSERT_EQ(int3_check, int3_ref);
        ASSERT_EQ(ai.getVariable<int>("int2", 0), 5);
        ASSERT_EQ(ai.getVariable<int>("int2", 1), 6);
        ASSERT_EQ(ai.getVariable<float>("float"), 15.0f);
    }

    // Update value
    {
        ai.setVariable<unsigned int>("uint", 12u + static_cast<unsigned int>(i));
        const std::array<int, 3> int3_set = { 2 + static_cast<int>(i), 3 + static_cast<int>(i), 4 + static_cast<int>(i) };
        ai.setVariable<int, 3>("int3", int3_set);
        ai.setVariable<int>("int2", 0, 5 + static_cast<int>(i));
        ai.setVariable<int>("int2", 1, 6 + static_cast<int>(i));
        ai.setVariable<float>("float", 15.0f + static_cast<float>(i));
    }

    // Check vars now match as expected
    {
        ASSERT_EQ(ai.getVariable<unsigned int>("uint"), 12u + static_cast<unsigned int>(i));
        const std::array<int, 3> int3_ref2 = { 2 + static_cast<int>(i), 3 + static_cast<int>(i), 4 + static_cast<int>(i) };
        const std::array<int, 3> int3_check = ai.getVariable<int, 3>("int3");
        ASSERT_EQ(int3_check, int3_ref2);
        ASSERT_EQ(ai.getVariable<int>("int2", 0), 5 + static_cast<int>(i));
        ASSERT_EQ(ai.getVariable<int>("int2", 1), 6 + static_cast<int>(i));
        ASSERT_EQ(ai.getVariable<float>("float"), 15.0f + static_cast<float>(i));
    }

    // Check various exceptions
    {  // setVariable(const std::string &variable_name, const T &value)
        // Bad name
        EXPECT_THROW(ai.setVariable<int>("wrong", 1), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW(ai.setVariable<int>("int2", 1), exception::InvalidVarType);
        // Wrong type
        EXPECT_THROW(ai.setVariable<int>("float", 1), exception::InvalidVarType);
    }
    {  // setVariable(const std::string &variable_name, const std::array<T, N> &value)
        auto fn = &AgentInstance::setVariable<int, 3>;
        auto fn2 = &AgentInstance::setVariable<float, 3>;
        const std::array<int, 3> int3_ref2 = { 2, 3, 4 };
        const std::array<float, 3> float3_ref = { 2.0f, 3.0f, 4.0f };
        // Bad name
        EXPECT_THROW((ai.*fn)("wrong", int3_ref2), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW((ai.*fn)("int2", int3_ref2), exception::InvalidVarType);
        // Wrong type
        EXPECT_THROW((ai.*fn2)("int3", float3_ref), exception::InvalidVarType);
    }
    {  // setVariable(const std::string &variable_name, const unsigned int &array_index, const T &value)
        // Bad name
        EXPECT_THROW(ai.setVariable<int>("wrong", 0, 1), exception::InvalidAgentVar);
        // Index out of bounds
        EXPECT_THROW(ai.setVariable<int>("int2", 2, 1), exception::OutOfBoundsException);
        EXPECT_THROW(ai.setVariable<float>("float", 1, 1), exception::OutOfBoundsException);
        // Wrong type
        EXPECT_THROW(ai.setVariable<int>("float", 0, 1), exception::InvalidVarType);
    }
    {  // getVariable(const std::string &variable_name) const
        // Bad name
        EXPECT_THROW(ai.getVariable<int>("wrong"), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW(ai.getVariable<int>("int2"), exception::InvalidVarType);
        // Wrong type
        EXPECT_THROW(ai.getVariable<int>("float"), exception::InvalidVarType);
    }
    {  // getVariable(const std::string &variable_name) const
        auto fn = &AgentInstance::getVariable<int, 3>;
        auto fn2 = &AgentInstance::getVariable<float, 3>;
        // Bad name
        EXPECT_THROW((ai.*fn)("wrong"), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW((ai.*fn)("int2"), exception::InvalidVarType);
        // Wrong type
        EXPECT_THROW((ai.*fn2)("int3"), exception::InvalidVarType);
    }
    {  // getVariable(const std::string &variable_name, const unsigned int &array_index) const
        // Bad name
        EXPECT_THROW(ai.getVariable<int>("wrong", 0), exception::InvalidAgentVar);
        // Index out of bounds
        EXPECT_THROW(ai.getVariable<int>("int2", 2), exception::OutOfBoundsException);
        EXPECT_THROW(ai.getVariable<float>("float", 1), exception::OutOfBoundsException);
        // Wrong type
        EXPECT_THROW(ai.getVariable<int>("float", 0), exception::InvalidVarType);
    }
}
}  // namespace flamegpu
