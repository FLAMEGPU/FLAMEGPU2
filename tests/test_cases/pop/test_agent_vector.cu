#ifndef TESTS_TEST_CASES_POP_TEST_AGENT_VECTOR_H_
#define TESTS_TEST_CASES_POP_TEST_AGENT_VECTOR_H_
#include "flamegpu/flamegpu.h"
#include "gtest/gtest.h"

namespace flamegpu {


TEST(AgentVectorTest, constructor) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector constructors, size(), array operator
    ModelDescription model("model");
    AgentDescription &agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    agent.newVariable<float>("float", 3.0f);
    agent.newVariable<double>("double", 4.0);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
#endif

    // Create empty vector
    AgentVector empty_pop(agent);
    ASSERT_EQ(empty_pop.size(), 0u);

    // Create vector with 10 agents, all default init
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i=0; i< POP_SIZE; ++i) {
        AgentVector::Agent instance = pop[i];
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
#ifdef USE_GLM
        ASSERT_EQ(instance.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
#endif
    }
}
TEST(AgentVectorTest, copy_constructor) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector copy constructors, size(), array operator
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    agent.newVariable<float>("float", 3.0f);
    agent.newVariable<double>("double", 4.0);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
#endif

    // Create empty vector
    AgentVector base_empty_pop(agent);
    AgentVector empty_pop(base_empty_pop);
    ASSERT_EQ(empty_pop.size(), 0u);

    // Create vector with 10 agents, all default init
    AgentVector base_pop(agent, POP_SIZE);
    AgentVector pop(base_pop);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent instance = pop[i];
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
#ifdef USE_GLM
        ASSERT_EQ(instance.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
#endif
    }
}
TEST(AgentVectorTest, move_constructor) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector move constructors, size(), array operator
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    agent.newVariable<float>("float", 3.0f);
    agent.newVariable<double>("double", 4.0);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
#endif

    // Create empty vector
    AgentVector base_empty_pop(agent);
    AgentVector empty_pop(std::move(base_empty_pop));
    ASSERT_EQ(empty_pop.size(), 0u);

    // Create vector with 10 agents, all default init
    AgentVector base_pop(agent, POP_SIZE);
    AgentVector pop(std::move(base_pop));
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent instance = pop[i];
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
#ifdef USE_GLM
        ASSERT_EQ(instance.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
#endif
    }
}
TEST(AgentVectorTest, copy_assignment_operator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector copy assignment, size(), array operator
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    agent.newVariable<float>("float", 3.0f);
    agent.newVariable<double>("double", 4.0);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
#endif

    // Create empty vector
    AgentVector base_empty_pop(agent);
    AgentVector pop(agent, 2u);  // Just some junk pop to be overwritten
    pop = base_empty_pop;
    ASSERT_EQ(pop.size(), 0u);

    // Create vector with 10 agents, all default init
    AgentVector base_pop(agent, POP_SIZE);
    pop = base_pop;
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent instance = pop[i];
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
#ifdef USE_GLM
        ASSERT_EQ(instance.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
#endif
    }
}
TEST(AgentVectorTest, move_assignment_operator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector move assignment, size(), array operator
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    agent.newVariable<float>("float", 3.0f);
    agent.newVariable<double>("double", 4.0);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
#endif

    // Create empty vector
    AgentVector base_empty_pop(agent);
    AgentVector pop(agent, 2u);  // Just some junk pop to be overwritten
    pop = std::move(base_empty_pop);
    ASSERT_EQ(pop.size(), 0u);

    // Create vector with 10 agents, all default init
    AgentVector base_pop(agent, POP_SIZE);
    pop = std::move(base_pop);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent instance = pop[i];
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
#ifdef USE_GLM
        ASSERT_EQ(instance.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
#endif
    }
}
TEST(AgentVectorTest, at) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector at(), synonymous with array operator
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    agent.newVariable<float>("float", 3.0f);
    agent.newVariable<double>("double", 4.0);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
#endif

    // Create vector with 10 agents, all default init
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent instance = pop.at(i);
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
#ifdef USE_GLM
        ASSERT_EQ(instance.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
#endif
    }

    // Create vector with 10 agents, all default init
    const AgentVector const_pop(agent, POP_SIZE);
    ASSERT_EQ(const_pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::CAgent instance = const_pop.at(i);
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
#ifdef USE_GLM
        ASSERT_EQ(instance.getVariable<glm::vec3>("vec3"), glm::vec3(2.0f, 4.0f, 6.0f));
#endif
    }

    // Out of bounds exception
    EXPECT_THROW(pop.at(POP_SIZE), exception::OutOfBoundsException);
    EXPECT_THROW(const_pop.at(POP_SIZE), exception::OutOfBoundsException);
    EXPECT_THROW(pop.at(POP_SIZE + 10), exception::OutOfBoundsException);
    EXPECT_THROW(const_pop.at(POP_SIZE + 10), exception::OutOfBoundsException);
}
TEST(AgentVectorTest, array_operator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector array operator (operator[]()), synonymous with at()
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);
    agent.newVariable<unsigned int>("uint", 2u);
    agent.newVariable<float>("float", 3.0f);
    agent.newVariable<double>("double", 4.0);

    // Create vector with 10 agents, all default init
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent instance = pop[i];
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
    }

    // Create vector with 10 agents, all default init
    const AgentVector const_pop(agent, POP_SIZE);
    ASSERT_EQ(const_pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::CAgent instance = const_pop[i];
        ASSERT_EQ(instance.getVariable<int>("int"), 1);
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), 2u);
        ASSERT_EQ(instance.getVariable<float>("float"), 3.0f);
        ASSERT_EQ(instance.getVariable<double>("double"), 4.0);
    }

    // Out of bounds exception
    EXPECT_THROW(pop[POP_SIZE], exception::OutOfBoundsException);
    EXPECT_THROW(const_pop[POP_SIZE], exception::OutOfBoundsException);
    EXPECT_THROW(pop[POP_SIZE + 10], exception::OutOfBoundsException);
    EXPECT_THROW(const_pop[POP_SIZE + 10], exception::OutOfBoundsException);
}
TEST(AgentVectorTest, front) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector front(), synonymous with at(0)
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);

    // Create vector with 10 agents, all default init
    AgentVector pop(agent, POP_SIZE);
    pop[0].setVariable<int>("int", 12);

    ASSERT_EQ(pop.front().getVariable<int>("int"), 12);
    ASSERT_EQ((static_cast<const AgentVector>(pop)).front().getVariable<int>("int"), 12);
    ASSERT_EQ(pop.front().getVariable<int>("int"), pop[0].getVariable<int>("int"));
    ASSERT_EQ((static_cast<const AgentVector>(pop)).front().getVariable<int>("int"), pop[0].getVariable<int>("int"));
    ASSERT_EQ(pop[1].getVariable<int>("int"), 1);  // Non-0th element is different

    // Out of bounds exception
    AgentVector empty_pop(agent);
    EXPECT_THROW(empty_pop.front(), exception::OutOfBoundsException);
    EXPECT_THROW(static_cast<const AgentVector>(empty_pop).front(), exception::OutOfBoundsException);
}
TEST(AgentVectorTest, back) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector back(), synonymous with at(size()-1)
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);

    // Create vector with 10 agents, all default init
    AgentVector pop(agent, POP_SIZE);
    pop[POP_SIZE-1].setVariable<int>("int", 12);

    ASSERT_EQ(pop.back().getVariable<int>("int"), 12);
    ASSERT_EQ((static_cast<const AgentVector>(pop)).back().getVariable<int>("int"), 12);
    ASSERT_EQ(pop.back().getVariable<int>("int"), pop[pop.size() - 1].getVariable<int>("int"));
    ASSERT_EQ((static_cast<const AgentVector>(pop)).back().getVariable<int>("int"), pop[pop.size() - 1].getVariable<int>("int"));
    ASSERT_EQ(pop[pop .size()-2].getVariable<int>("int"), 1);  // Non-0th element is different

    // Out of bounds exception
    AgentVector empty_pop(agent);
    EXPECT_THROW(empty_pop.back(), exception::OutOfBoundsException);
    EXPECT_THROW(static_cast<const AgentVector>(empty_pop).back(), exception::OutOfBoundsException);
}
TEST(AgentVectorTest, data) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector data()
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<int>("int", 1);

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    AgentVector &cpop = pop;
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<int>("int", static_cast<int>(i));
    }

    // Data vector has the right data
    const int* data_ptr = pop.data<int>("int");
    const int* const_data_ptr = cpop.data<int>("int");
    const int* void_data_ptr = static_cast<const int*>(pop.data("int"));
    const int* const_void_data_ptr = static_cast<const int*>(cpop.data("int"));
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        EXPECT_EQ(data_ptr[i], static_cast<int>(i));
        EXPECT_EQ(const_data_ptr[i], static_cast<int>(i));
        EXPECT_EQ(void_data_ptr[i], static_cast<int>(i));
        EXPECT_EQ(const_void_data_ptr[i], static_cast<int>(i));
    }

    // Empty returns nullptr
    AgentVector empty_pop(agent);
    EXPECT_EQ(empty_pop.data<int>("int"), nullptr);
    EXPECT_EQ(empty_pop.data("int"), nullptr);
    EXPECT_EQ(static_cast<const AgentVector>(empty_pop).data<int>("int"), nullptr);
    EXPECT_EQ(static_cast<const AgentVector>(empty_pop).data("int"), nullptr);

    // Invalid exception::InvalidAgentVar
    EXPECT_THROW(pop.data<int>("float"), exception::InvalidAgentVar);
    EXPECT_THROW(pop.data<unsigned int>("uint"), exception::InvalidAgentVar);
    EXPECT_THROW(pop.data<int>("int12"), exception::InvalidAgentVar);
    // Invalid exception::InvalidVarType
    EXPECT_THROW(pop.data<float>("int"), exception::InvalidVarType);
    EXPECT_THROW(pop.data<unsigned int>("int"), exception::InvalidVarType);
    EXPECT_THROW(pop.data<int64_t>("int"), exception::InvalidVarType);
}
TEST(AgentVectorTest, iterator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector array iterator, and the member functions for creating them.
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }

    // Iterate vector
    unsigned int i = 0;
    for (AgentVector::Agent instance : pop) {
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), i++);
    }
    ASSERT_EQ(i, pop.size());

    // Test empty is empty
    AgentVector empty_pop(agent);
    i = 0;
    for (AgentVector::Agent instance : empty_pop) {
        ++i;
    }
    ASSERT_EQ(i, 0u);
}
#ifdef USE_GLM
TEST(AgentVectorTest, iterator_GLM) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector array iterator, and the member functions for creating them.
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<glm::uvec3>("uvec3");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<glm::uvec3>("uvec3", glm::uvec3(i +3, i + 6, i));
    }

    // Iterate vector
    unsigned int i = 0;
    for (AgentVector::Agent instance : pop) {
        auto a = instance.getVariable<glm::uvec3>("uvec3");
        ASSERT_EQ(instance.getVariable<glm::uvec3>("uvec3"), glm::uvec3(i + 3, i + 6, i));
        ++i;
    }
    ASSERT_EQ(i, pop.size());

    // Test empty is empty
    AgentVector empty_pop(agent);
    i = 0;
    for (AgentVector::Agent instance : empty_pop) {
        ++i;
    }
    ASSERT_EQ(i, 0u);
}
#else
TEST(AgentVectorTest, DISABLED_iterator_glm) { }
#endif
TEST(AgentVectorTest, const_iterator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector const_iterator
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    AgentVector &cpop = pop;
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }

    // Iterate vector
    unsigned int i = 0;
    for (AgentVector::CAgent instance : cpop) {
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), i++);
    }
    ASSERT_EQ(i, cpop.size());

    // Iterator vector with alt const_iterator version
    i = 0;
    for (AgentVector::const_iterator it = pop.cbegin(); it != pop.cend(); ++it) {
        AgentVector::CAgent instance = *it;
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), i++);
    }
    ASSERT_EQ(i, pop.size());

    // Test empty is empty
    AgentVector empty_pop(agent);
    AgentVector &cempty_pop = empty_pop;
    i = 0;
    for (AgentVector::CAgent instance : cempty_pop) {
        ++i;
    }
    ASSERT_EQ(i, 0u);

    // Test empty is empty, alt notation
    i = 0;
    for (AgentVector::const_iterator it = empty_pop.cbegin(); it != empty_pop.cend(); ++it) {
        ++i;
    }
    ASSERT_EQ(i, 0u);
}
TEST(AgentVectorTest, reverse_iterator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector reverse_iterator, and the member functions for creating them.
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }

    // Iterate vector
    unsigned int i = pop.size();
    for (AgentVector::reverse_iterator it = pop.rbegin();  it != pop.rend(); ++it) {
        AgentVector::Agent instance = *it;
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), --i);
    }
    ASSERT_EQ(i, 0u);

    // Test empty is empty
    AgentVector empty_pop(agent);
    i = 0;
    for (AgentVector::reverse_iterator it = empty_pop.rbegin(); it != empty_pop.rend(); ++it) {
        ++i;
    }
    ASSERT_EQ(i, 0u);
}
TEST(AgentVectorTest, const_reverse_iterator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector const_reverse_iterator, and the member functions for creating them.
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    const AgentVector& cpop = pop;
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }

    // Iterate vector
    unsigned int i = cpop.size();
    for (AgentVector::const_reverse_iterator it = cpop.rbegin(); it != cpop.rend(); ++it) {
        AgentVector::CAgent instance = *it;
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), --i);
    }
    ASSERT_EQ(i, 0u);

    // Iterator vector with alt const_reverse_iterator notation
    i = pop.size();
    for (AgentVector::const_reverse_iterator it = pop.crbegin(); it != pop.crend(); ++it) {
        AgentVector::CAgent instance = *it;
        ASSERT_EQ(instance.getVariable<unsigned int>("uint"), --i);
    }
    ASSERT_EQ(i, 0u);

    // Test empty is empty
    AgentVector empty_pop(agent);
    const AgentVector &cempty_pop = empty_pop;
    i = 0;
    for (AgentVector::const_reverse_iterator it = cempty_pop.rbegin(); it != cempty_pop.rend(); ++it) {
        ++i;
    }
    ASSERT_EQ(i, 0u);

    // Test empty is empty, alt notation
    i = 0;
    for (AgentVector::const_reverse_iterator it = empty_pop.crbegin(); it != empty_pop.crend(); ++it) {
        ++i;
    }
    ASSERT_EQ(i, 0u);
}
TEST(AgentVectorTest, empty) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector empty
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.empty(), false);
    AgentVector pop2(agent);
    ASSERT_EQ(pop2.empty(), true);
}
TEST(AgentVectorTest, size) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector size
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    AgentVector pop2(agent);
    ASSERT_EQ(pop2.size(), 0u);
}
// TEST(AgentVectorTest, size): Nothing to test
TEST(AgentVectorTest, reserve) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector reserve
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }

    // Init size
    ASSERT_EQ(pop.size(), POP_SIZE);
    ASSERT_GE(pop.capacity(), POP_SIZE);

    // Reserving up works as expected
    const unsigned int RESERVE_SIZE = pop.capacity() * 10;
    pop.reserve(RESERVE_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    ASSERT_GE(pop.capacity(), RESERVE_SIZE);

    // Reserving down does nothing
    pop.reserve(POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    ASSERT_GE(pop.capacity(), RESERVE_SIZE);

    // Data remains initialised
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
    }
}
// TEST(AgentVectorTest, capacity): reserve contains best testing of this
TEST(AgentVectorTest, shrink_to_fit) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector shrink_to_fit
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    // Create vector with 10 agents, init to their index
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }

    // Init size
    ASSERT_EQ(pop.size(), POP_SIZE);
    ASSERT_GE(pop.capacity(), POP_SIZE);

    // Grow the vector's capacity
    const unsigned int RESERVE_SIZE = pop.capacity() * 10;
    pop.reserve(RESERVE_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    ASSERT_GE(pop.capacity(), RESERVE_SIZE);

    //  Shrink to fit
    pop.shrink_to_fit();
    ASSERT_EQ(pop.size(), POP_SIZE);
    ASSERT_GE(pop.capacity(), POP_SIZE);

    // Data remains initialised
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
    }
}
TEST(AgentVectorTest, clear) {
    const unsigned int POP_SIZE = 10;
    const unsigned int DEFAULT_VALUE = 12;
    // Test correctness of AgentVector array operator (operator[]()), synonymous with at()
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", DEFAULT_VALUE);

    // Create vector with 10 agents, init to non-default value
    AgentVector pop(agent, POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }

    ASSERT_EQ(pop.size(), POP_SIZE);
    const unsigned int capacity = pop.capacity();
    // Clear resets size, but does not affect capacity
    pop.clear();
    ASSERT_EQ(pop.size(), 0u);
    ASSERT_EQ(pop.capacity(), capacity);

    // If items are added back, they are default init
    pop.push_back();
    pop.push_back();
    pop.push_back();
    pop.resize(POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), DEFAULT_VALUE);
    }
}
TEST(AgentVectorTest, insert) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector insert
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");
    // AgentVector to insert
    AgentVector insert_av(agent, POP_SIZE);
    ASSERT_EQ(insert_av.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        insert_av[i].setVariable<unsigned int>("uint", 23u + i);
    }
    AgentVector::Agent insert_ava = insert_av.front();
    // AgentInstance to insert
    AgentInstance insert_ai(agent);
    insert_ai.setVariable<unsigned int>("uint", 24u);

    // Insert single item
    {  // insert(const_iterator pos, const AgentInstance& value)
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Increment iterator to point to 4th item
        auto it = pop.begin();
        for (int i = 0; i < 4; ++i) {
            ++it;
        }
        // Use iterator to insert item
        pop.insert(it, insert_ai);
        ASSERT_EQ(pop.size(), POP_SIZE + 1);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i == 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 24u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 1);
            }
        }
    }
    {  // insert(size_type pos, const AgentInstance& value)
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Use iterator to insert item
        pop.insert(4, insert_ai);
        ASSERT_EQ(pop.size(), POP_SIZE + 1);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i == 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 24u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 1);
            }
        }
    }
    {  // insert(const_iterator pos, const Agent& value)
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Increment iterator to point to 4th item
        auto it = pop.begin();
        for (int i = 0; i < 4; ++i) {
            ++it;
        }
        // Use iterator to insert item
        pop.insert(it, insert_ava);
        ASSERT_EQ(pop.size(), POP_SIZE + 1);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i == 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 23u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 1);
            }
        }
    }
    {  // insert(size_type pos, const Agent& value)
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Use iterator to insert item
        pop.insert(4, insert_ava);
        ASSERT_EQ(pop.size(), POP_SIZE + 1);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i == 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 23u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 1);
            }
        }
    }
    // Insert multiple copies
    {  // insert(const_iterator pos, size_type count, const AgentInstance& value);
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Increment iterator to point to 4th item
        auto it = pop.begin();
        for (int i = 0; i < 4; ++i) {
            ++it;
        }
        // Use iterator to insert 3 items
        pop.insert(it, 3, insert_ai);
        ASSERT_EQ(pop.size(), POP_SIZE + 3);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i < 7) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 24u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 3);
            }
        }
    }
    {  // insert(size_type pos, size_type count, const AgentInstance& value);
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Use iterator to insert 3 items
        pop.insert(4, 3, insert_ai);
        ASSERT_EQ(pop.size(), POP_SIZE + 3);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i < 7) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 24u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 3);
            }
        }
    }
    {  // insert(const_iterator pos, size_type count, const Agent& value);
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Increment iterator to point to 4th item
        auto it = pop.begin();
        for (int i = 0; i < 4; ++i) {
            ++it;
        }
        // Use iterator to insert 3 items
        pop.insert(it, 3, insert_ava);
        ASSERT_EQ(pop.size(), POP_SIZE + 3);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i < 7) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 23u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 3);
            }
        }
    }
    {  // insert(size_type pos, size_type count, const Agent& value);
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Use iterator to insert 3 items
        pop.insert(4, 3, insert_ava);
        ASSERT_EQ(pop.size(), POP_SIZE + 3);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i < 7) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 23u);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - 3);
            }
        }
    }
    // Insert range
    {  // insert(const_iterator pos, InputIt first, InputIt last)
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Increment iterator to point to 4th item
        auto it = pop.begin();
        for (int i = 0; i < 4; ++i) {
            ++it;
        }
        // Use iterator to insert all items
        pop.insert(it, insert_av.begin(), insert_av.end());
        ASSERT_EQ(pop.size(), POP_SIZE + POP_SIZE);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i < 4 + POP_SIZE) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 23u + i - 4);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - POP_SIZE);
            }
        }
    }
    {  // insert(size_type pos, InputIt first, InputIt last)
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Use iterator to insert all items
        pop.insert(4, insert_av.begin(), insert_av.end());
        ASSERT_EQ(pop.size(), POP_SIZE + POP_SIZE);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else if (i < 4 + POP_SIZE) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 23u + i - 4);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i - POP_SIZE);
            }
        }
    }

    // Wrong agents for exceptions
    AgentDescription& agent2 = model.newAgent("agent2");
    agent2.newVariable<float>("float");
    AgentVector pop(agent, POP_SIZE);
    auto it = pop.begin();
    for (int i = 0; i < 4; ++i)
      ++it;
    // AgentVector to insert
    AgentVector wrong_insert_av(agent2, POP_SIZE);
    AgentVector::Agent  wrong_insert_ava = wrong_insert_av.front();
    // AgentInstance to insert
    AgentInstance  wrong_insert_ai(agent2);

    EXPECT_THROW(pop.insert(it, wrong_insert_ai), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(4, wrong_insert_ai), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(it, wrong_insert_ava), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(4, wrong_insert_ava), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(it, 3, wrong_insert_ai), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(4, 3, wrong_insert_ai), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(it, 3, wrong_insert_ava), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(4, 3, wrong_insert_ava), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(it, wrong_insert_av.begin(), wrong_insert_av.end()), exception::InvalidAgent);
    EXPECT_THROW(pop.insert(4, wrong_insert_av.begin(), wrong_insert_av.end()), exception::InvalidAgent);
}
TEST(AgentVectorTest, erase_single) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector erase (on single items)
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", POP_SIZE + 2);

    {
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Increment iterator to point to 4th item
        auto it = pop.begin();
        for (int i = 0; i < 4; ++i) {
            ++it;
        }
        // Use iterator to remove item
        pop.erase(it);
        // Check 4 has gone missing from vector
        ASSERT_EQ(pop.size(), POP_SIZE-1);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i + 1);
            }
        }
        // If we add back an item, it is default init
        pop.push_back();
        ASSERT_EQ(pop.back().getVariable<unsigned int>("uint"), POP_SIZE + 2);
    }

    {
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Use index to remove item
        pop.erase(4);
        // Check 4 has gone missing from vector
        ASSERT_EQ(pop.size(), POP_SIZE - 1);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i + 1);
            }
        }
        // If we add back an item, it is default init
        pop.push_back();
        ASSERT_EQ(pop.back().getVariable<unsigned int>("uint"), POP_SIZE + 2);
    }

    // Test exceptions
    AgentVector pop(agent, POP_SIZE);
    EXPECT_THROW(pop.erase(POP_SIZE), exception::OutOfBoundsException);
    EXPECT_THROW(pop.erase(POP_SIZE + 2), exception::OutOfBoundsException);
    EXPECT_NO_THROW(pop.erase(POP_SIZE-1));
}
TEST(AgentVectorTest, erase_range) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector erase (on single items)
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");

    {
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Increment iterator to point to 4th item
        auto it = pop.begin();
        for (int i = 0; i < 4; ++i) {
            ++it;
        }
        // Increment 2nd iterator to point behind the 7th item
        auto it2 = it;
        for (int i = 4; i <= 7; ++i) {
            ++it2;
        }
        // Use iterator to remove item
        pop.erase(it, it2);
        // Check 4,5,6,7 has gone missing from vector
        ASSERT_EQ(pop.size(), POP_SIZE - 4);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i + 4);
            }
        }
    }

    {
        // Create vector with 10 agents, init to their index
        AgentVector pop(agent, POP_SIZE);
        ASSERT_EQ(pop.size(), POP_SIZE);
        for (unsigned int i = 0; i < POP_SIZE; ++i) {
            pop[i].setVariable<unsigned int>("uint", i);
        }
        // Use index to remove item
        pop.erase(4, 8);
        // Check 4,5,6,7 has gone missing from vector
        ASSERT_EQ(pop.size(), POP_SIZE - 4);
        for (unsigned int i = 0; i < pop.size(); ++i) {
            if (i < 4) {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
            } else {
                ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i + 4);
            }
        }
    }

    // Test exceptions
    AgentVector pop(agent, POP_SIZE);
    EXPECT_THROW(pop.erase(POP_SIZE, POP_SIZE + 2), exception::OutOfBoundsException);
    EXPECT_THROW(pop.erase(POP_SIZE/2, POP_SIZE + 2), exception::OutOfBoundsException);
    EXPECT_THROW(pop.erase(POP_SIZE + 2, POP_SIZE + 4), exception::OutOfBoundsException);
    EXPECT_NO_THROW(pop.erase(0, POP_SIZE));
}
TEST(AgentVectorTest, push_back) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector push_back, and whether created item is default init
    // The impact on erase/clear/pop_back/etc functions on default init, is tested by their respective tests
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", 2u);
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    pop.back().setVariable<unsigned int>("uint", 12);
    pop.push_back();
    ASSERT_EQ(pop.size(), POP_SIZE + 1);
    ASSERT_EQ(pop.back().getVariable<unsigned int>("uint"), 2u);
    // Test alt-push_back
    AgentInstance ai(agent);
    ai.setVariable<unsigned int>("uint", 22u);
    pop.push_back(ai);
    ASSERT_EQ(pop.size(), POP_SIZE + 2);
    ASSERT_EQ(pop.back().getVariable<unsigned int>("uint"), 22u);
    // Diff agent fail
    AgentDescription& agent2 = model.newAgent("agent2");
    agent2.newVariable<float>("float", 2.0f);
    AgentInstance ai2(agent2);
    EXPECT_THROW(pop.push_back(ai2), exception::InvalidAgent);
}
TEST(AgentVectorTest, pop_back) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector pop_back
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", 2u);
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }
    ASSERT_EQ(pop.size(), POP_SIZE);
    // Pop back removes the last item
    pop.pop_back();
    ASSERT_EQ(pop.size(), POP_SIZE-1);
    ASSERT_EQ(pop.back().getVariable<unsigned int>("uint"), POP_SIZE - 2);
    // Adding back a new item is default init
    pop.push_back();
    ASSERT_EQ(pop.size(), POP_SIZE);
    ASSERT_EQ(pop.back().getVariable<unsigned int>("uint"), 2u);

    // Test that pop_back on empty has no effect
    AgentVector pop2(agent);
    ASSERT_EQ(pop2.size(), 0u);
    EXPECT_NO_THROW(pop2.pop_back());
    ASSERT_EQ(pop2.size(), 0u);
}
TEST(AgentVectorTest, resize) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector resize
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", 2u);
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }
    ASSERT_EQ(pop.size(), POP_SIZE);
    // Resizing to a lower size, removes the trailing elements
    const unsigned int SMALL_POP_SIZE = POP_SIZE - POP_SIZE / 3;
    pop.resize(SMALL_POP_SIZE);
    for (unsigned int i = 0; i  < SMALL_POP_SIZE; ++i) {
        ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
    }
    ASSERT_EQ(pop.size(), SMALL_POP_SIZE);
    // Resizing to a bigger size, adds back default items
    const unsigned int BIG_POP_SIZE = POP_SIZE + 3;
    pop.resize(BIG_POP_SIZE);
    for (unsigned int i = 0; i < BIG_POP_SIZE; ++i) {
        if (i < SMALL_POP_SIZE) {
            ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
        } else {
            ASSERT_EQ(pop[i].getVariable<unsigned int>("uint"), 2u);
        }
    }
    ASSERT_EQ(pop.size(), BIG_POP_SIZE);
}
TEST(AgentVectorTest, swap) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector swap
    // This can be applied to agents of different types, but not testing that
    // Should work effectively the same
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", 2u);
    AgentVector pop_default(agent,  2 * POP_SIZE);
    AgentVector pop(agent, POP_SIZE);
    ASSERT_EQ(pop.size(), POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        pop[i].setVariable<unsigned int>("uint", i);
    }
    // Check pops are expected
    EXPECT_EQ(pop.size(), POP_SIZE);
    EXPECT_EQ(pop_default.size(), 2 * POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        EXPECT_EQ(pop[i].getVariable<unsigned int>("uint"), i);
        EXPECT_EQ(pop_default[i].getVariable<unsigned int>("uint"), 2u);
        EXPECT_EQ(pop_default[POP_SIZE + i].getVariable<unsigned int>("uint"), 2u);
    }
    // Swap and check again
    pop.swap(pop_default);
    EXPECT_EQ(pop_default.size(), POP_SIZE);
    EXPECT_EQ(pop.size(), 2 * POP_SIZE);
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        EXPECT_EQ(pop_default[i].getVariable<unsigned int>("uint"), i);
        EXPECT_EQ(pop[i].getVariable<unsigned int>("uint"), 2u);
        EXPECT_EQ(pop[POP_SIZE + i].getVariable<unsigned int>("uint"), 2u);
    }
}
TEST(AgentVectorTest, equality_operator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector operator==
    ModelDescription model("model");
    AgentDescription& agent1 = model.newAgent("agent");
    agent1.newVariable<unsigned int>("uint", 2u);
    ModelDescription model2("model2");
    AgentDescription& agent2 = model2.newAgent("agent");
    agent2.newVariable<unsigned int>("uint", 2u);
    ModelDescription model3("model3");
    AgentDescription& agent3 = model3.newAgent("agent");
    agent3.newVariable<unsigned int>("uint", 2u);
    agent3.newVariable<float>("float", 3u);
    ModelDescription model4("model4");
    AgentDescription& agent4 = model4.newAgent("agent");
    agent4.newVariable<int>("int", 2);
    ModelDescription model5("model5");
    AgentDescription& agent5 = model5.newAgent("agent");
    agent5.newVariable<int>("uint", 2);
    AgentDescription& agent6 = model.newAgent("agent2");
    agent6.newVariable<unsigned int>("uint", 2u);
    AgentVector pop(agent1, POP_SIZE);
    // Copy of the list is equal
    AgentVector pop2 = pop;
    EXPECT_TRUE(pop == pop2);
    EXPECT_TRUE(pop2 == pop);
    // Different, but identical agentdesc is equal
    AgentVector pop3(agent2, POP_SIZE);
    EXPECT_TRUE(pop == pop3);
    EXPECT_TRUE(pop3 == pop);
    // But not if the lengths differ
    AgentVector pop4(agent2, POP_SIZE + 1);
    EXPECT_FALSE(pop == pop4);
    EXPECT_FALSE(pop4 == pop);
    // Not if we have additional vars
    AgentVector pop5(agent3, POP_SIZE);
    EXPECT_FALSE(pop == pop5);
    EXPECT_FALSE(pop5 == pop);
    // Or var has diff type
    AgentVector pop6(agent4, POP_SIZE);
    EXPECT_FALSE(pop == pop6);
    EXPECT_FALSE(pop6 == pop);
    // Or diff type, same name
    AgentVector pop7(agent5, POP_SIZE);
    EXPECT_FALSE(pop == pop7);
    EXPECT_FALSE(pop7 == pop);
    // Different agent name
    AgentVector pop8(agent6, POP_SIZE);
    EXPECT_FALSE(pop == pop8);
    EXPECT_FALSE(pop8 == pop);
    // Or if the value of the variable differs
    pop2.front().setVariable<unsigned int>("uint", 12u);
    EXPECT_FALSE(pop == pop2);
    EXPECT_FALSE(pop2 == pop);
}
TEST(AgentVectorTest, inequality_operator) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector operator!=
    ModelDescription model("model");
    AgentDescription& agent1 = model.newAgent("agent");
    agent1.newVariable<unsigned int>("uint", 2u);
    ModelDescription model2("model2");
    AgentDescription& agent2 = model2.newAgent("agent");
    agent2.newVariable<unsigned int>("uint", 2u);
    ModelDescription model3("model3");
    AgentDescription& agent3 = model3.newAgent("agent");
    agent3.newVariable<unsigned int>("uint", 2u);
    agent3.newVariable<float>("float", 3u);
    ModelDescription model4("model4");
    AgentDescription& agent4 = model4.newAgent("agent");
    agent4.newVariable<int>("int", 2);
    ModelDescription model5("model5");
    AgentDescription& agent5 = model5.newAgent("agent");
    agent5.newVariable<int>("uint", 2);
    AgentDescription& agent6 = model.newAgent("agent2");
    agent6.newVariable<unsigned int>("uint", 2u);
    AgentVector pop(agent1, POP_SIZE);
    // Copy of the list is equal
    AgentVector pop2 = pop;
    EXPECT_FALSE(pop != pop2);
    EXPECT_FALSE(pop2 != pop);
    // Different, but identical agentdesc is equal
    AgentVector pop3(agent2, POP_SIZE);
    EXPECT_FALSE(pop != pop3);
    EXPECT_FALSE(pop3 != pop);
    // But not if the lengths differ
    AgentVector pop4(agent2, POP_SIZE + 1);
    EXPECT_TRUE(pop != pop4);
    EXPECT_TRUE(pop4 != pop);
    // Not if we have additional vars
    AgentVector pop5(agent3, POP_SIZE);
    EXPECT_TRUE(pop != pop5);
    EXPECT_TRUE(pop5 != pop);
    // Or var has diff type
    AgentVector pop6(agent4, POP_SIZE);
    EXPECT_TRUE(pop != pop6);
    EXPECT_TRUE(pop6 != pop);
    // Or diff type, same name
    AgentVector pop7(agent5, POP_SIZE);
    EXPECT_TRUE(pop != pop7);
    EXPECT_TRUE(pop7 != pop);
    // Different agent name
    AgentVector pop8(agent6, POP_SIZE);
    EXPECT_TRUE(pop != pop8);
    EXPECT_TRUE(pop8 != pop);
    // Or if the value of the variable differs
    pop2.front().setVariable<unsigned int>("uint", 12u);
    EXPECT_TRUE(pop != pop2);
    EXPECT_TRUE(pop2 != pop);
}
TEST(AgentVectorTest, getAgentName) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector getAgentName
    ModelDescription model("model");
    AgentDescription& agent1 = model.newAgent("agent");
    agent1.newVariable<unsigned int>("uint", 2u);
    AgentDescription& agent2 = model.newAgent("testtest");
    agent2.newVariable<unsigned int>("uint", 2u);
    AgentVector pop(agent1, POP_SIZE);
    EXPECT_EQ(pop.getAgentName(), "agent");
    AgentVector pop2(agent2, POP_SIZE);
    EXPECT_EQ(pop2.getAgentName(), "testtest");
}
TEST(AgentVectorTest, matchesAgentType) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector::matchesAgentType(AgentDescription)
    ModelDescription model("model");
    AgentDescription& agent1 = model.newAgent("agent");
    agent1.newVariable<unsigned int>("uint", 2u);
    AgentDescription& agent2 = model.newAgent("testtest");
    agent2.newVariable<int>("int", 2u);
    AgentVector pop(agent1, POP_SIZE);
    AgentVector pop1(agent1);
    EXPECT_TRUE(pop.matchesAgentType(agent1));
    EXPECT_FALSE(pop.matchesAgentType(agent2));
    EXPECT_TRUE(pop1.matchesAgentType(agent1));
    EXPECT_FALSE(pop1.matchesAgentType(agent2));
    AgentVector pop2(agent2, POP_SIZE);
    EXPECT_FALSE(pop2.matchesAgentType(agent1));
    EXPECT_TRUE(pop2.matchesAgentType(agent2));
}
TEST(AgentVectorTest, getVariableType) {
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector getVariableType
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");
    agent.newVariable<int>("int");
    AgentVector pop(agent, POP_SIZE);
    EXPECT_EQ(pop.getVariableType("uint"), std::type_index(typeid(unsigned int)));
    EXPECT_EQ(pop.getVariableType("int"), std::type_index(typeid(int)));
    EXPECT_THROW(pop.getVariableType("float"), exception::InvalidAgentVar);
}
// TEST(AgentVectorTest, getVariableMetaData): can't test this
TEST(AgentVectorTest, getInitialState) {
    // Test correctness of AgentVector getInitialState
    // Though this is moreso testing how iniital state works
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint");
    AgentDescription& agent2 = model.newAgent("agent2");
    agent2.newState("test");
    AgentDescription& agent3 = model.newAgent("agent3");
    agent3.newState("test");
    agent3.newState("test2");
    AgentDescription& agent4 = model.newAgent("agent4");
    agent4.newState("test");
    agent4.newState("test2");
    agent4.setInitialState("test2");
    AgentVector pop(agent);
    AgentVector pop2(agent2);
    AgentVector pop3(agent3);
    AgentVector pop4(agent4);
    EXPECT_EQ(pop.getInitialState(), ModelData::DEFAULT_STATE);
    EXPECT_EQ(pop2.getInitialState(), "test");
    EXPECT_EQ(pop3.getInitialState(), "test");
    EXPECT_EQ(pop4.getInitialState(), "test2");
}
TEST(AgentVectorTest, AgentVector_Agent) {
  // Test that AgentVector::Agent provides set/get access to an agent of an AgentVector
    const unsigned int POP_SIZE = 10;
    // Test correctness of AgentVector getVariableType
    ModelDescription model("model");
    AgentDescription& agent = model.newAgent("agent");
    agent.newVariable<unsigned int>("uint", 12u);
    agent.newVariable<int, 3>("int3", {2, 3, 4});
    agent.newVariable<int, 2>("int2", { 5, 6 });
    agent.newVariable<float>("float", 15.0f);
#ifdef USE_GLM
    agent.newVariable<glm::vec3>("vec3", glm::vec3(2.0f, 4.0f, 6.0f));
    agent.newVariable<glm::ivec3, 3>("ivec3_3", {glm::ivec3(12, 14, 16), glm::ivec3(2, 4, 6), glm::ivec3(22, 24, 26)});
    agent.newVariable<glm::ivec3, 3>("ivec3_3b", {glm::ivec3(12, 14, 16), glm::ivec3(2, 4, 6), glm::ivec3(22, 24, 26)});
#endif

    // Create pop, variables are as expected
    AgentVector pop(agent, POP_SIZE);
    const std::array<int, 3> int3_ref = { 2, 3, 4 };
#ifdef USE_GLM
    const std::array<glm::ivec3, 3> vec_array_check = {glm::ivec3(12, 14, 16), glm::ivec3(2, 4, 6), glm::ivec3(22, 24, 26)};
#endif
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent ai = pop[i];
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
        ASSERT_EQ(ai.getVariable<glm::ivec3>("ivec3_3b", 0), glm::ivec3(12, 14, 16));
        ASSERT_EQ(ai.getVariable<glm::ivec3>("ivec3_3b", 1), glm::ivec3(2, 4, 6));
        ASSERT_EQ(ai.getVariable<glm::ivec3>("ivec3_3b", 2), glm::ivec3(22, 24, 26));
#endif
    }

    // Update values
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("uint", 12u + static_cast<unsigned int>(i));
        const std::array<int, 3> int3_set = { 2 + static_cast<int>(i), 3 + static_cast<int>(i), 4 + static_cast<int>(i) };
        ai.setVariable<int, 3>("int3", int3_set);
        ai.setVariable<int>("int2", 0, 5 + static_cast<int>(i));
        ai.setVariable<int>("int2", 1, 6 + static_cast<int>(i));
        ai.setVariable<float>("float", 15.0f + static_cast<float>(i));
#ifdef USE_GLM
        ai.setVariable<glm::vec3>("vec3", glm::vec3(2.0f + static_cast<float>(i), 4.0f + static_cast<float>(i), 6.0f + static_cast<float>(i)));
        ai.setVariable<glm::ivec3, 3>("ivec3_3", {glm::ivec3(12, 14, 16) + glm::ivec3(static_cast<int>(i)), glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i)), glm::ivec3(22, 24, 26) + glm::ivec3(static_cast<int>(i))});
        // Don't update ivec3_3b index 0
        ai.setVariable<glm::ivec3>("ivec3_3b", 1, glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i) * 3));
        ai.setVariable<glm::ivec3>("ivec3_3b", 2, glm::ivec3(2, 4, 6) + glm::ivec3(static_cast<int>(i) * 4));
#endif
    }

    // Check vars now match as expected
    for (unsigned int i = 0; i < POP_SIZE; ++i) {
        AgentVector::Agent ai = pop[i];
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
    AgentVector::Agent ai = pop.front();
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
        EXPECT_THROW(ai.setVariable<glm::ivec3>("ivec3_3", 4, {}), exception::OutOfBoundsException);
        EXPECT_THROW(ai.setVariable<glm::ivec3>("int3", 1, {}), exception::OutOfBoundsException);
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
    {  // getVariable(const std::string &variable_name)
        // Bad name
        EXPECT_THROW((ai.getVariable<int, 3>)("wrong"), exception::InvalidAgentVar);
        // Array passed to non-array method
        EXPECT_THROW((ai.getVariable<int, 3>)("int2"), exception::InvalidVarType);
        // Wrong type
        EXPECT_THROW((ai.getVariable<float, 3>)("int3"), exception::InvalidVarType);
    }
    {  // getVariable(const std::string &variable_name, unsigned int array_index)
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
#endif  // TESTS_TEST_CASES_POP_TEST_AGENT_VECTOR_H_
