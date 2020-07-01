#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

namespace test_model {
    const std::string MODEL_NAME = "something";
    const std::string AGENT_NAME1 = "something2";
    const std::string AGENT_NAME2 = "something3";
}  // namespace test_model


TEST(ModelDescriptionTest, Name) {
    ModelDescription m(test_model::MODEL_NAME);
    // Model has the right name
    EXPECT_EQ(m.getName(), test_model::MODEL_NAME);
}
TEST(ModelDescriptionTest, Agent) {
    ModelDescription m(test_model::MODEL_NAME);
    EXPECT_FALSE(m.hasAgent(test_model::AGENT_NAME1));
    EXPECT_FALSE(m.hasAgent(test_model::AGENT_NAME2));
    EXPECT_EQ(m.getAgentsCount(), 0u);
    AgentDescription &a = m.newAgent(test_model::AGENT_NAME1);
    EXPECT_EQ(m.getAgentsCount(), 1u);
    AgentDescription &b = m.newAgent(test_model::AGENT_NAME2);
    EXPECT_EQ(m.getAgentsCount(), 2u);
    // Cannot create agent with same name
    EXPECT_THROW(m.newAgent(test_model::AGENT_NAME1), InvalidAgentName);
    // Two created agents are different
    EXPECT_NE(a, b);
    // Agents have the right name
    EXPECT_EQ(a.getName(), test_model::AGENT_NAME1);
    EXPECT_EQ(b.getName(), test_model::AGENT_NAME2);
    EXPECT_TRUE(m.hasAgent(test_model::AGENT_NAME1));
    EXPECT_TRUE(m.hasAgent(test_model::AGENT_NAME2));
    // Returned agent is same
    EXPECT_EQ(a, m.Agent(test_model::AGENT_NAME1));
    EXPECT_EQ(b, m.Agent(test_model::AGENT_NAME2));
    EXPECT_EQ(a, m.getAgent(test_model::AGENT_NAME1));
    EXPECT_EQ(b, m.getAgent(test_model::AGENT_NAME2));
}
TEST(ModelDescriptionTest, Message) {
    ModelDescription m(test_model::MODEL_NAME);
    EXPECT_FALSE(m.hasMessage(test_model::AGENT_NAME1));
    EXPECT_FALSE(m.hasMessage(test_model::AGENT_NAME2));
    EXPECT_EQ(m.getMessagesCount(), 0u);
    MsgBruteForce::Description &a = m.newMessage(test_model::AGENT_NAME1);
    EXPECT_EQ(m.getMessagesCount(), 1u);
    MsgBruteForce::Description &b = m.newMessage(test_model::AGENT_NAME2);
    EXPECT_EQ(m.getMessagesCount(), 2u);
    // Cannot create message with same name
    EXPECT_THROW(m.newMessage(test_model::AGENT_NAME1), InvalidMessageName);
    // Two created messages are different
    EXPECT_NE(a, b);
    // Messages have the right name
    EXPECT_EQ(a.getName(), test_model::AGENT_NAME1);
    EXPECT_EQ(b.getName(), test_model::AGENT_NAME2);
    EXPECT_TRUE(m.hasMessage(test_model::AGENT_NAME1));
    EXPECT_TRUE(m.hasMessage(test_model::AGENT_NAME2));
    // Returned message is same
    EXPECT_EQ(a, m.Message(test_model::AGENT_NAME1));
    EXPECT_EQ(b, m.Message(test_model::AGENT_NAME2));
    EXPECT_EQ(a, m.getMessage(test_model::AGENT_NAME1));
    EXPECT_EQ(b, m.getMessage(test_model::AGENT_NAME2));
}
TEST(ModelDescriptionTest, Layer) {
    ModelDescription m(test_model::MODEL_NAME);
    EXPECT_FALSE(m.hasLayer(test_model::AGENT_NAME1));
    EXPECT_FALSE(m.hasLayer(test_model::AGENT_NAME2));
    EXPECT_FALSE(m.hasLayer(0));
    EXPECT_FALSE(m.hasLayer(1));
    EXPECT_EQ(m.getLayersCount(), 0u);
    LayerDescription &a = m.newLayer(test_model::AGENT_NAME1);
    EXPECT_EQ(m.getLayersCount(), 1u);
    EXPECT_TRUE(m.hasLayer(0));
    EXPECT_TRUE(m.hasLayer(test_model::AGENT_NAME1));
    LayerDescription &b = m.newLayer(test_model::AGENT_NAME2);
    EXPECT_EQ(m.getLayersCount(), 2u);
    // Cannot create layer with same name
    EXPECT_THROW(m.newLayer(test_model::AGENT_NAME1), InvalidFuncLayerIndx);
    // Two created layers are different
    EXPECT_NE(a, b);
    // Layers have the right name
    EXPECT_EQ(a.getName(), test_model::AGENT_NAME1);
    EXPECT_EQ(b.getName(), test_model::AGENT_NAME2);
    EXPECT_TRUE(m.hasLayer(test_model::AGENT_NAME1));
    EXPECT_TRUE(m.hasLayer(test_model::AGENT_NAME2));
    EXPECT_TRUE(m.hasLayer(0));
    EXPECT_TRUE(m.hasLayer(1));
    // Returned layer is same
    EXPECT_EQ(a, m.Layer(test_model::AGENT_NAME1));
    EXPECT_EQ(b, m.Layer(test_model::AGENT_NAME2));
    EXPECT_EQ(a, m.Layer(0));
    EXPECT_EQ(b, m.Layer(1));
    EXPECT_EQ(a, m.getLayer(test_model::AGENT_NAME1));
    EXPECT_EQ(b, m.getLayer(test_model::AGENT_NAME2));
    EXPECT_EQ(a, m.getLayer(0));
    EXPECT_EQ(b, m.getLayer(1));
    EXPECT_EQ(0u, a.getIndex());
    EXPECT_EQ(1u, b.getIndex());
}
FLAMEGPU_EXIT_CONDITION(ExitAlways) {
    return EXIT;
}
TEST(ModelDescriptionTest, SubModelRecursionNotAllowed) {
    ModelDescription a("a");
    ModelDescription b("b");
    ModelDescription c("c");
    ModelDescription d("d");
    a.addExitCondition(ExitAlways);
    b.addExitCondition(ExitAlways);
    c.addExitCondition(ExitAlways);
    d.addExitCondition(ExitAlways);
    EXPECT_NO_THROW(a.newSubModel("b", b));
    EXPECT_NO_THROW(a.newSubModel("c", c));
    // Name already exists
    EXPECT_THROW(a.newSubModel("c", d), InvalidSubModelName);
    EXPECT_NO_THROW(c.newSubModel("d", d));
    // A cannot sub itself
    EXPECT_THROW(a.newSubModel("", a), InvalidSubModel);
    // B is already sub of A
    EXPECT_THROW(a.newSubModel("", b), InvalidSubModel);
    // A is already a parent of B
    EXPECT_THROW(b.newSubModel("", a), InvalidSubModel);
    // A is already a parent of C
    EXPECT_THROW(c.newSubModel("", a), InvalidSubModel);
    EXPECT_THROW(d.newSubModel("", a), InvalidSubModel);
    // D is already a child of A
    EXPECT_THROW(c.newSubModel("", d), InvalidSubModel);
    // Submodel can be added twice if to different branches
    EXPECT_NO_THROW(b.newSubModel("d2", d));
    // D cannot add itself
    EXPECT_THROW(d.newSubModel("", d), InvalidSubModel);
    // C is already a parent of D
    EXPECT_THROW(d.newSubModel("", c), InvalidSubModel);
}
