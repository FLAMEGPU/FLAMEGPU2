#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


TEST(AgentInstanceTest, constructor) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(4.0f, 5.0f, 6.0f));
#endif
    AgentInstance ai(agent);
    // New AgentInstance is default init
    ASSERT_EQ(ai.getVariable<int>("int"), 1);
    ASSERT_EQ(ai.getVariable<unsigned int>("uint"), 2u);
#ifdef USE_GLM
    ASSERT_EQ(ai.getVariable<glm::vec3>("vec3"), glm::vec3(4.0f, 5.0f, 6.0f));
#endif
}
TEST(AgentInstanceTest, copy_constructor) {
  ModelDescription model("model");
  AgentDescription& agent = model.newAgent("agent");
  agent.newVariable<int>("int", 1);
  agent.newVariable<unsigned int, 3>("uint3", {2u, 3u, 4u});
  const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
#ifdef USE_GLM
  agent.newVariable<glm::vec3>("vec3", glm::vec3(4.0f, 5.0f, 6.0f));
#endif
  // Copying an agent instance retains the values
  AgentInstance ai(agent);
  ai.setVariable<int>("int", 12);
  ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
  AgentInstance ai2(ai);
  ASSERT_EQ(ai2.getVariable<int>("int"), 12);
  auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
  ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
#ifdef USE_GLM
  ASSERT_EQ(ai2.getVariable<glm::vec3>("vec3"), glm::vec3(4.0f, 5.0f, 6.0f));
#endif
  // Copying an agent instance from an AgentVector::Agent retains values
  AgentVector av(agent, 1);
  AgentVector::Agent ava = av.front();
  ava.setVariable<int>("int", 12);
  ava.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
#ifdef USE_GLM
  ava.setVariable<glm::vec3>("vec3", glm::vec3(6.0f, 5.0f, 4.0f));
#endif
  AgentInstance ai3(ava);
  ASSERT_EQ(ai3.getVariable<int>("int"), 12);
  auto ai2_uint3_check2 = ai3.getVariable<unsigned int, 3>("uint3");
  ASSERT_EQ(ai2_uint3_check2, ai_uint3_ref);
  ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
#ifdef USE_GLM
  ASSERT_EQ(ai3.getVariable<glm::vec3>("vec3"), glm::vec3(6.0f, 5.0f, 4.0f));
#endif
}
TEST(AgentInstanceTest, move_constructor) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int, 3>("uint3", { 2u, 3u, 4u });
    const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(4.0f, 5.0f, 6.0f));
#endif
    // Moving an agent instance retains the values
    AgentInstance ai(agent);
    ai.setVariable<int>("int", 12);
    ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
#ifdef USE_GLM
    ai.setVariable<glm::vec3>("vec3", glm::vec3(6.0f, 5.0f, 4.0f));
#endif
    AgentInstance ai2(std::move(ai));
    ASSERT_EQ(ai2.getVariable<int>("int"), 12);
    auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
#ifdef USE_GLM
    ASSERT_EQ(ai2.getVariable<glm::vec3>("vec3"), glm::vec3(6.0f, 5.0f, 4.0f));
#endif
}
TEST(AgentInstanceTest, copy_assignment_operator) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    AgentDescription& agent2 = model.newAgent("agent2");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int, 3>("uint3", { 2u, 3u, 4u });
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(4.0f, 5.0f, 6.0f));
#endif
    const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
    // Copying an agent instance retains the values
    AgentInstance ai(agent);
    ai.setVariable<int>("int", 12);
    ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
#ifdef USE_GLM
    ai.setVariable<glm::vec3>("vec3", glm::vec3(16.0f, 15.0f, 14.0f));
#endif
    AgentInstance ai2(agent2);
    ai2 = ai;
    ASSERT_EQ(ai2.getVariable<int>("int"), 12);
    auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
#ifdef USE_GLM
    ASSERT_EQ(ai2.getVariable<glm::vec3>("vec3"), glm::vec3(16.0f, 15.0f, 14.0f));
#endif
    // Copying an agent instance from an AgentVector::Agent retains values
    AgentVector av(agent, 1);
    AgentVector::Agent ava = av.front();
    ava.setVariable<int>("int", 12);
    ava.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
#ifdef USE_GLM
    ava.setVariable<glm::vec3>("vec3", glm::vec3(6.0f, 5.0f, 4.0f));
#endif
    AgentInstance ai3(agent2);
    ai3 = ava;
    ASSERT_EQ(ai3.getVariable<int>("int"), 12);
    auto ai2_uint3_check2 = ai3.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check2, ai_uint3_ref);
#ifdef USE_GLM
    ASSERT_EQ(ai3.getVariable<glm::vec3>("vec3"), glm::vec3(6.0f, 5.0f, 4.0f));
#endif
}
TEST(AgentInstanceTest, move_assignment_operator) {
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    AgentDescription& agent2 = model.newAgent("agent2");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int, 3>("uint3", { 2u, 3u, 4u });
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(4.0f, 5.0f, 6.0f));
#endif
    const std::array<unsigned int, 3> ai_uint3_ref = { 0u, 1u, 2u };
    // Moving an agent instance retains the values
    AgentInstance ai(agent);
    ai.setVariable<int>("int", 12);
    ai.setVariable<unsigned int, 3>("uint3", ai_uint3_ref);
#ifdef USE_GLM
    ai.setVariable<glm::vec3>("vec3", glm::vec3(16.0f, 15.0f, 14.0f));
#endif
    AgentInstance ai2(agent2);
    ai2 = std::move(ai);
    ASSERT_EQ(ai2.getVariable<int>("int"), 12);
    auto ai2_uint3_check = ai2.getVariable<unsigned int, 3>("uint3");
    ASSERT_EQ(ai2_uint3_check, ai_uint3_ref);
#ifdef USE_GLM
    ASSERT_EQ(ai2.getVariable<glm::vec3>("vec3"), glm::vec3(16.0f, 15.0f, 14.0f));
#endif
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
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
    agent.newVariable<glm::ivec3, 3>("ivec3_3", {glm::ivec3(12, 14, 16), glm::ivec3(2, 4, 6), glm::ivec3(22, 24, 26)});
    agent.newVariable<glm::ivec3, 3>("ivec3_3b", {glm::ivec3(12, 14, 16), glm::ivec3(2, 4, 6), glm::ivec3(22, 24, 26)});
#endif

    // Create pop, variables are as expected
    AgentInstance ai(agent);
    const std::array<int, 3> int3_ref = { 2, 3, 4 };
#ifdef USE_GLM
    const std::array<glm::ivec3, 3> vec_array_check = {glm::ivec3(12, 14, 16), glm::ivec3(2, 4, 6), glm::ivec3(22, 24, 26)};
#endif
    {
        ASSERT_EQ(ai.getVariable<unsigned int>("uint"), 12u);
        const std::array<int, 3> int3_check = ai.getVariable<int, 3>("int3");
        ASSERT_EQ(int3_check, int3_ref);
        ASSERT_EQ(ai.getVariable<int>("int2", 0), 5);
        ASSERT_EQ(ai.getVariable<int>("int2", 1), 6);
        ASSERT_EQ(ai.getVariable<float>("float"), 15.0f);
#ifdef USE_GLM
        ASSERT_EQ(ai.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
        const auto vec_array_test = ai.getVariable<glm::ivec3, 3>("ivec3_3");
        ASSERT_EQ(vec_array_test, vec_array_check);
#endif
    }

    // Update value
    {
        ai.setVariable<unsigned int>("uint", 12u + static_cast<unsigned int>(i));
        const std::array<int, 3> int3_set = { 2 + static_cast<int>(i), 3 + static_cast<int>(i), 4 + static_cast<int>(i) };
        ai.setVariable<int, 3>("int3", int3_set);
        ai.setVariable<int>("int2", 0, 5 + static_cast<int>(i));
        ai.setVariable<int>("int2", 1, 6 + static_cast<int>(i));
        ai.setVariable<float>("float", 15.0f + static_cast<float>(i));
#ifdef USE_GLM
        ai.setVariable<glm::vec3>("vec3", glm::vec3(2.0f + static_cast<float>(i), 4.0f + static_cast<float>(i), 6.0f + static_cast<float>(i)));
        ai.setVariable<glm::ivec3, 3>("ivec3_3", {glm::ivec3(12, 14, 16) + glm::ivec3(static_cast<int>(i)), glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i)), glm::ivec3(22, 24, 26) + glm::ivec3(static_cast<int>(i))});
        ai.setVariable<glm::ivec3>("ivec3_3b", 1, glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i) * 3));
        ai.setVariable<glm::ivec3>("ivec3_3b", 2, glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i) * 4));
#endif
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
#ifdef USE_GLM
        ASSERT_EQ(ai.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f + static_cast<float>(i), 4.0f + static_cast<float>(i), 6.0f + static_cast<float>(i)));
        const std::array<glm::ivec3, 3> vec_array_check2 = {glm::ivec3(12, 14, 16) + glm::ivec3(static_cast<int>(i)), glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i)), glm::ivec3(22, 24, 26) + glm::ivec3(static_cast<int>(i))};
        const std::array<glm::ivec3, 3> vec_array_test = ai.getVariable<glm::ivec3, 3>("ivec3_3");
        ASSERT_EQ(vec_array_test, vec_array_check2);
        ASSERT_EQ(ai.getVariable<glm::ivec3>("ivec3_3b", 0), glm::ivec3(12, 14, 16));
        ASSERT_EQ(ai.getVariable<glm::ivec3>("ivec3_3b", 1), glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i) * 3));
        ASSERT_EQ(ai.getVariable<glm::ivec3>("ivec3_3b", 2), glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i) * 4));
#endif
    }

    // Check various exceptions
    {  // setVariable(const std::string &variable_name, T value)
        // Bad name
        EXPECT_THROW(ai.setVariable<int>("wrong", 1), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW(ai.setVariable<int>("int2", 1), exception::InvalidVarType);
#ifdef USE_GLM
        EXPECT_THROW(ai.setVariable<glm::vec3>("float", {}), exception::InvalidVarType);
#endif
        // Wrong type
        EXPECT_THROW(ai.setVariable<int>("float", 1), exception::InvalidVarType);
    }
    {  // setVariable(const std::string &variable_name, const std::array<T, N> &value)
        const std::array<int, 3> int3_ref2 = { 2, 3, 4 };
        const std::array<float, 3> float3_ref = { 2.0f, 3.0f, 4.0f };
        // Bad name
        EXPECT_THROW((ai.setVariable<int, 3>)("wrong", int3_ref2), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW((ai.setVariable<int, 3>)("int2", int3_ref2), exception::InvalidVarType);
        // Wrong type
        EXPECT_THROW((ai.setVariable<float, 3>)("int3", float3_ref), exception::InvalidVarType);
    }
    {  // setVariable(const std::string &variable_name, unsigned int array_index, T value)
        // Bad name
        EXPECT_THROW(ai.setVariable<int>("wrong", 0, 1), exception::InvalidAgentVar);
        // Index out of bounds
        EXPECT_THROW(ai.setVariable<int>("int2", 2, 1), exception::OutOfBoundsException);
        EXPECT_THROW(ai.setVariable<float>("float", 1, 1), exception::OutOfBoundsException);
#ifdef USE_GLM
        EXPECT_THROW(ai.setVariable<glm::vec3>("ivec3_3", 4, {}), exception::OutOfBoundsException);
        EXPECT_THROW(ai.setVariable<glm::vec3>("int3", 1, {}), exception::OutOfBoundsException);
#endif
        // Wrong type
        EXPECT_THROW(ai.setVariable<int>("float", 0, 1), exception::InvalidVarType);
    }
    {  // getVariable(const std::string &variable_name) const
        // Bad name
        EXPECT_THROW(ai.getVariable<int>("wrong"), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW(ai.getVariable<int>("int2"), exception::InvalidVarType);
#ifdef USE_GLM
        EXPECT_THROW(ai.getVariable<glm::vec3>("float"), exception::InvalidVarType);
#endif
        // Wrong type
        EXPECT_THROW(ai.getVariable<int>("float"), exception::InvalidVarType);
    }
    {  // getVariable(const std::string &variable_name) const
        // Bad name
        EXPECT_THROW((ai.getVariable<int, 3>)("wrong"), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW((ai.getVariable<int, 3>)("int2"), exception::InvalidVarType);
        // Wrong type
        EXPECT_THROW((ai.getVariable<float, 3>)("int3"), exception::InvalidVarType);
    }
    {  // getVariable(const std::string &variable_name, unsigned int array_index) const
        // Bad name
        EXPECT_THROW(ai.getVariable<int>("wrong", 0), exception::InvalidAgentVar);
        // Index out of bounds
        EXPECT_THROW(ai.getVariable<int>("int2", 2), exception::OutOfBoundsException);
        EXPECT_THROW(ai.getVariable<float>("float", 1), exception::OutOfBoundsException);
#ifdef USE_GLM
        EXPECT_THROW(ai.getVariable<glm::vec3>("ivec3_3", 4), exception::OutOfBoundsException);
        EXPECT_THROW(ai.getVariable<glm::vec3>("int3", 1), exception::OutOfBoundsException);
#endif
        // Wrong type
        EXPECT_THROW(ai.getVariable<int>("float", 0), exception::InvalidVarType);
    }
}
}  // namespace flamegpu
