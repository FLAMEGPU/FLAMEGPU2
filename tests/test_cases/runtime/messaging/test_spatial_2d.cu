/**
* Tests of feature Spatial 3D messaging
*
* Tests cover:
* > mandatory messaging, send/recieve
*/
#include <unordered_map>

#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_message_spatial2d {

FLAMEGPU_AGENT_FUNCTION(out_mandatory2D, MessageNone, MessageSpatial2D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optional2D, MessageNone, MessageSpatial2D) {
    if (FLAMEGPU->getVariable<int>("do_output")) {
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
        FLAMEGPU->message_out.setLocation(
            FLAMEGPU->getVariable<float>("x"),
            FLAMEGPU->getVariable<float>("y"));
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optional2DNone, MessageNone, MessageSpatial2D) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(in2D, MessageSpatial2D, MessageNone) {
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    unsigned int count = 0;
    unsigned int badCount = 0;
     unsigned int myBin[2] = {
         static_cast<unsigned int>(x1),
         static_cast<unsigned int>(y1)
     };
    // Count how many messages we recieved (including our own)
    // This is all those which fall within the 3x3x3 Moore neighbourhood
    // Not our search radius
    for (const auto &message : FLAMEGPU->message_in(x1, y1)) {
         unsigned int messageBin[2] = {
             static_cast<unsigned int>(message.getVariable<float>("x")),
             static_cast<unsigned int>(message.getVariable<float>("y"))
         };
         bool isBad = false;
         for (unsigned int i = 0; i < 2; ++i) {  // Iterate axis
             int binDiff = myBin[i] - messageBin[i];
             if (binDiff > 1 || binDiff < -1) {
                 isBad = true;
             }
         }
        count++;
        badCount = isBad ? badCount + 1 : badCount;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    FLAMEGPU->setVariable<unsigned int>("badCount", badCount);
    return ALIVE;
}
TEST(Spatial2DMessageTest, Mandatory) {
    std::unordered_map<int, unsigned int> bin_counts;
    // Construct model
    ModelDescription model("Spatial2DMessageTestModel");
    {   // Location message
        MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
        message.setMin(0, 0);
        message.setMax(11, 11);
        message.setRadius(1);
        // 11x11 bins, total 121
        message.newVariable<flamegpu::id_t>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        agent.newFunction("out", out_mandatory2D).setMessageOutput("location");
        agent.newFunction("in", in2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(out_mandatory2D);
    }
    {   // Layer #2
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(in2D);
    }
    CUDASimulation cudaSimulation(model);

    const int AGENT_COUNT = 2049;
    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, 11.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            float pos[3] = { dist(rng), dist(rng), dist(rng) };
            instance.setVariable<float>("x", pos[0]);
            instance.setVariable<float>("y", pos[1]);
            // Solve the bin index
            const unsigned int bin_pos[2] = {
                (unsigned int)(pos[0] / 1),
                (unsigned int)(pos[1] / 1)
            };
            const unsigned int bin_index =
                bin_pos[1] * 11 +
                bin_pos[0];
            instance.setVariable<unsigned int>("myBin", bin_index);
            // Create it if it doesn't already exist
            if (bin_counts.find(bin_index) == bin_counts.end()) {
                bin_counts.emplace(bin_index, 0);
            }
            bin_counts[bin_index] += 1;
        }
        cudaSimulation.setPopulationData(population);
    }

    // Generate results expectation
    std::unordered_map<int, unsigned int> bin_results;
    // Iterate host bin
    for (unsigned int x1 = 0; x1 < 11; x1++) {
        for (unsigned int y1 = 0; y1 < 11; y1++) {
            // Solve the bin index
            const unsigned int bin_pos1[3] = {
                x1,
                y1
            };
            const unsigned int bin_index1 =
                bin_pos1[1] * 11 +
                bin_pos1[0];
            // Count our neighbours
            unsigned int count_sum = 0;
            for (int x2 = -1; x2 <= 1; x2++) {
                int bin_pos2[2] = {
                    static_cast<int>(bin_pos1[0]) + x2,
                    0
                };
                for (int y2 = -1; y2 <= 1; y2++) {
                    bin_pos2[1] = static_cast<int>(bin_pos1[1]) + y2;
                    // Ensure bin is in bounds
                    if (
                        bin_pos2[0] >= 0 &&
                        bin_pos2[1] >= 0 &&
                        bin_pos2[0] < 11 &&
                        bin_pos2[1] < 11
                        ) {
                        const unsigned int bin_index2 =
                            bin_pos2[1] * 11 +
                            bin_pos2[0];
                        count_sum += bin_counts[bin_index2];
                    }
                }
            }
            bin_results.emplace(bin_index1, count_sum);
         }
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected

    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    unsigned int badCountWrong = 0;
    for (AgentVector::Agent ai : population) {
        unsigned int myBin = ai.getVariable<unsigned int>("myBin");
        unsigned int myResult = ai.getVariable<unsigned int>("count");
        EXPECT_EQ(myResult, bin_results.at(myBin));
        if (ai.getVariable<unsigned int>("badCount"))
            badCountWrong++;
    }
    EXPECT_EQ(badCountWrong, 0u);
}

TEST(Spatial2DMessageTest, Optional) {
    /**
     * This test is same as Mandatory, however extra flag has been added to block certain agents from outputting messages
     * Look for NEW!
     */
    std::unordered_map<int, unsigned int> bin_counts;
    std::unordered_map<int, unsigned int> bin_counts_optional;
    // Construct model
    ModelDescription model("Spatial2DMessageTestModel");
    {   // Location message
        MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
        message.setMin(0, 0);
        message.setMax(11, 11);
        message.setRadius(1);
        // 11x11 bins, total 121
        message.newVariable<flamegpu::id_t>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<int>("do_output");  // NEW!
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        auto af = agent.newFunction("out", out_optional2D);  // NEW!
        af.setMessageOutput("location");
        af.setMessageOutputOptional(true);  // NEW!
        agent.newFunction("in", in2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(out_optional2D);  // NEW!
    }
    {   // Layer #2
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(in2D);
    }
    CUDASimulation cudaSimulation(model);

    const int AGENT_COUNT = 2049;
    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, 11.0f);
        std::uniform_real_distribution<float> dist5(0.0f, 5.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            float pos[3] = { dist(rng), dist(rng), dist(rng) };
            int do_output = dist5(rng) < 4 ? 1 : 0;  // 80% chance of output  // NEW!
            instance.setVariable<float>("x", pos[0]);
            instance.setVariable<float>("y", pos[1]);
            instance.setVariable<int>("do_output", do_output);  // NEW!
            // Solve the bin index
            const unsigned int bin_pos[2] = {
                (unsigned int)(pos[0] / 1),
                (unsigned int)(pos[1] / 1)
            };
            const unsigned int bin_index =
                bin_pos[1] * 11 +
                bin_pos[0];
            instance.setVariable<unsigned int>("myBin", bin_index);
            // Create it if it doesn't already exist
            bin_counts[bin_index] += 1;
            if (do_output) {  // NEW!
                bin_counts_optional[bin_index] += 1;  // NEW!
            }
        }
        cudaSimulation.setPopulationData(population);
    }

    // Generate results expectation
    std::unordered_map<int, unsigned int> bin_results;
    std::unordered_map<int, unsigned int> bin_results_optional;
    // Iterate host bin
    for (unsigned int x1 = 0; x1 < 11; x1++) {
        for (unsigned int y1 = 0; y1 < 11; y1++) {
            // Solve the bin index
            const unsigned int bin_pos1[3] = {
                x1,
                y1
            };
            const unsigned int bin_index1 =
                bin_pos1[1] * 11 +
                bin_pos1[0];
            // Count our neighbours
            unsigned int count_sum = 0;
            unsigned int count_sum_optional = 0;  // NEW!
            for (int x2 = -1; x2 <= 1; x2++) {
                int bin_pos2[2] = {
                    static_cast<int>(bin_pos1[0]) + x2,
                    0
                };
                for (int y2 = -1; y2 <= 1; y2++) {
                    bin_pos2[1] = static_cast<int>(bin_pos1[1]) + y2;
                    // Ensure bin is in bounds
                    if (
                        bin_pos2[0] >= 0 &&
                        bin_pos2[1] >= 0 &&
                        bin_pos2[0] < 11 &&
                        bin_pos2[1] < 11
                        ) {
                        const unsigned int bin_index2 =
                            bin_pos2[1] * 11 +
                            bin_pos2[0];
                        count_sum += bin_counts[bin_index2];
                        count_sum_optional += bin_counts_optional[bin_index2];  // NEW!
                    }
                }
            }
            bin_results.emplace(bin_index1, count_sum);
            bin_results_optional.emplace(bin_index1, count_sum_optional);  // NEW!
        }
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected

    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    unsigned int badCountWrong = 0;
    for (AgentVector::Agent ai : population) {
        unsigned int myBin = ai.getVariable<unsigned int>("myBin");
        unsigned int myResult = ai.getVariable<unsigned int>("count");
        if (ai.getVariable<unsigned int>("badCount"))
            badCountWrong++;
        EXPECT_EQ(myResult, bin_results_optional.at(myBin));  // NEW!
    }
    EXPECT_EQ(badCountWrong, 0u);
}
TEST(Spatial2DMessageTest, OptionalNone) {
    /**
     * This test is same as Mandatory, however extra flag has been added to block certain agents from outputting messages
     * Look for NEW!
     */
    std::unordered_map<int, unsigned int> bin_counts;
    std::unordered_map<int, unsigned int> bin_counts_optional;
    // Construct model
    ModelDescription model("Spatial2DMessageTestModel");
    {   // Location message
        MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
        message.setMin(0, 0);
        message.setMax(11, 11);
        message.setRadius(1);
        // 11x11 bins, total 121
        message.newVariable<flamegpu::id_t>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<int>("do_output");  // NEW!
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        auto af = agent.newFunction("out", out_optional2DNone);  // NEW!
        af.setMessageOutput("location");
        af.setMessageOutputOptional(true);  // NEW!
        agent.newFunction("in", in2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(out_optional2DNone);  // NEW!
    }
    {   // Layer #2
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(in2D);
    }
    CUDASimulation cudaSimulation(model);

    const int AGENT_COUNT = 2049;
    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, 11.0f);
        std::uniform_real_distribution<float> dist5(0.0f, 5.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            float pos[3] = { dist(rng), dist(rng), dist(rng) };
            int do_output = dist5(rng) < 4 ? 1 : 0;  // 80% chance of output  // NEW!
            instance.setVariable<float>("x", pos[0]);
            instance.setVariable<float>("y", pos[1]);
            instance.setVariable<int>("do_output", do_output);  // NEW!
            // Solve the bin index
            const unsigned int bin_pos[2] = {
                (unsigned int)(pos[0] / 1),
                (unsigned int)(pos[1] / 1)
            };
            const unsigned int bin_index =
                bin_pos[1] * 11 +
                bin_pos[0];
            instance.setVariable<unsigned int>("myBin", bin_index);
            // Create it if it doesn't already exist
            bin_counts[bin_index] += 1;
            if (do_output) {  // NEW!
                bin_counts_optional[bin_index] += 1;  // NEW!
            }
        }
        cudaSimulation.setPopulationData(population);
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected

    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    unsigned int badCountWrong = 0;
    for (AgentVector::Agent ai : population) {
        unsigned int myResult = ai.getVariable<unsigned int>("count");
        if (ai.getVariable<unsigned int>("badCount"))
            badCountWrong++;
        EXPECT_EQ(myResult, 0u);  // NEW!
    }
    EXPECT_EQ(badCountWrong, 0u);
}

TEST(Spatial2DMessageTest, BadRadius) {
    ModelDescription model("Spatial2DMessageTestModel");
    MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
    EXPECT_THROW(message.setRadius(0), exception::InvalidArgument);
    EXPECT_THROW(message.setRadius(-10), exception::InvalidArgument);
}
TEST(Spatial2DMessageTest, BadMin) {
    ModelDescription model("Spatial2DMessageTestModel");
    MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
    message.setMax(5, 5);
    EXPECT_THROW(message.setMin(5, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(0, 5), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(6, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(0, 6), exception::InvalidArgument);
}
TEST(Spatial2DMessageTest, BadMax) {
    ModelDescription model("Spatial2DMessageTestModel");
    MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
    message.setMin(5, 5);
    EXPECT_THROW(message.setMax(5, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(0, 5), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(4, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(0, 4), exception::InvalidArgument);
}
TEST(Spatial2DMessageTest, UnsetMax) {
    ModelDescription model("Spatial2DMessageTestModel");
    MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
    message.setMin(5, 5);
    EXPECT_THROW(CUDASimulation m(model), exception::InvalidMessage);
}
TEST(Spatial2DMessageTest, UnsetMin) {
    ModelDescription model("Spatial2DMessageTestModel");
    MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
    message.setMin(5, 5);
    EXPECT_THROW(CUDASimulation m(model), exception::InvalidMessage);
}
TEST(Spatial2DMessageTest, reserved_name) {
    ModelDescription model("Spatial2DMessageTestModel");
    MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
    EXPECT_THROW(message.newVariable<int>("_"), exception::ReservedName);
}

FLAMEGPU_AGENT_FUNCTION(count2D, MessageSpatial2D, MessageNone) {
    unsigned int count = 0;
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3 Moore neighbourhood
    for (const auto &message : FLAMEGPU->message_in(0, 0)) {
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return ALIVE;
}
TEST(Spatial2DMessageTest, ReadEmpty) {
// What happens if we read a message list before it has been output?
    ModelDescription model("Model");
    {   // Location message
        MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
        message.setMin(-3, -3);
        message.setMax(3, 3);
        message.setRadius(2);
        message.newVariable<flamegpu::id_t>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<unsigned int>("count", 0);  // Count the number of messages read
        agent.newFunction("in", count2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(count2D);
    }
    // Create 1 agent
    AgentVector pop_in(model.Agent("agent"), 1);
    CUDASimulation cudaSimulation(model);
    cudaSimulation.setPopulationData(pop_in);
    // Execute model
    EXPECT_NO_THROW(cudaSimulation.step());
    // Check result
    AgentVector pop_out(model.Agent("agent"), 1);
    pop_out[0].setVariable<unsigned int>("count", 1);
    cudaSimulation.getPopulationData(pop_out);
    EXPECT_EQ(pop_out.size(), 1u);
    EXPECT_EQ(pop_out[0].getVariable<unsigned int>("count"), 0u);
}

FLAMEGPU_AGENT_FUNCTION(ArrayOut, MessageNone, MessageSpatial2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, y * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, y * 11);
    FLAMEGPU->message_out.setLocation(static_cast<float>(x), static_cast<float>(y));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn, MessageSpatial2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    for (auto &message : FLAMEGPU->message_in(static_cast<float>(x), static_cast<float>(y))) {
        if (static_cast<unsigned int>(message.getVariable<float>("x")) == x &&
            static_cast<unsigned int>(message.getVariable<float>("y")) == y) {
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
            break;
        }
    }
    return ALIVE;
}
TEST(Spatial2DMessageTest, ArrayVariable) {
    const char* MODEL_NAME = "Model";
    const char* AGENT_NAME = "Agent";
    const char* MESSAGE_NAME = "Message";
    const char* IN_FUNCTION_NAME = "InFunction";
    const char* OUT_FUNCTION_NAME = "OutFunction";
    const char* IN_LAYER_NAME = "InLayer";
    const char* OUT_LAYER_NAME = "OutLayer";
    const unsigned int SQRT_AGENT_COUNT = 64;
    ModelDescription m(MODEL_NAME);
    MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>(MESSAGE_NAME);
    message.setMin(0, 0);
    message.setMax(static_cast<float>(SQRT_AGENT_COUNT), static_cast<float>(SQRT_AGENT_COUNT));
    message.setRadius(1);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}
const char* rtc_ArrayOut_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, y * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, y * 11);
    FLAMEGPU->message_out.setLocation(static_cast<float>(x), static_cast<float>(y));
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    for (auto &message : FLAMEGPU->message_in(static_cast<float>(x), static_cast<float>(y))) {
        if (static_cast<unsigned int>(message.getVariable<float>("x")) == x &&
            static_cast<unsigned int>(message.getVariable<float>("y")) == y) {
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
            break;
        }
    }
    return flamegpu::ALIVE;
}
)###";
TEST(RTCSpatial2DMessageTest, ArrayVariable) {
    const char* MODEL_NAME = "Model";
    const char* AGENT_NAME = "Agent";
    const char* MESSAGE_NAME = "Message";
    const char* IN_FUNCTION_NAME = "InFunction";
    const char* OUT_FUNCTION_NAME = "OutFunction";
    const char* IN_LAYER_NAME = "InLayer";
    const char* OUT_LAYER_NAME = "OutLayer";
    const unsigned int SQRT_AGENT_COUNT = 64;
    ModelDescription m(MODEL_NAME);
    MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>(MESSAGE_NAME);
    message.setMin(0, 0);
    message.setMax(static_cast<float>(SQRT_AGENT_COUNT), static_cast<float>(SQRT_AGENT_COUNT));
    message.setRadius(1);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}

#if defined(FLAMEGPU_USE_GLM)
FLAMEGPU_AGENT_FUNCTION(ArrayOut_glm, MessageNone, MessageSpatial2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    glm::uvec3 t = glm::uvec3(x * 3, y * 7, y * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setLocation(static_cast<float>(x), static_cast<float>(y));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_glm, MessageSpatial2D, MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    for (auto &message : FLAMEGPU->message_in(static_cast<float>(x), static_cast<float>(y))) {
        if (static_cast<unsigned int>(message.getVariable<float>("x")) == x &&
            static_cast<unsigned int>(message.getVariable<float>("y")) == y) {
            FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
            break;
        }
    }
    return ALIVE;
}
TEST(Spatial2DMessageTest, ArrayVariable_glm) {
    const char* MODEL_NAME = "Model";
    const char* AGENT_NAME = "Agent";
    const char* MESSAGE_NAME = "Message";
    const char* IN_FUNCTION_NAME = "InFunction";
    const char* OUT_FUNCTION_NAME = "OutFunction";
    const char* IN_LAYER_NAME = "InLayer";
    const char* OUT_LAYER_NAME = "OutLayer";
    const unsigned int SQRT_AGENT_COUNT = 64;
    ModelDescription m(MODEL_NAME);
    MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>(MESSAGE_NAME);
    message.setMin(0, 0);
    message.setMax(static_cast<float>(SQRT_AGENT_COUNT), static_cast<float>(SQRT_AGENT_COUNT));
    message.setRadius(1);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_glm);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}
const char* rtc_ArrayOut_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    glm::uvec3 t = glm::uvec3(x * 3, y * 7, y * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setLocation(static_cast<float>(x), static_cast<float>(y));
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 2>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 2>("index", 1);
    for (auto &message : FLAMEGPU->message_in(static_cast<float>(x), static_cast<float>(y))) {
        if (static_cast<unsigned int>(message.getVariable<float>("x")) == x &&
            static_cast<unsigned int>(message.getVariable<float>("y")) == y) {
            FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
            break;
        }
    }
    return flamegpu::ALIVE;
}
)###";
TEST(RTCSpatial2DMessageTest, ArrayVariable_glm) {
    const char* MODEL_NAME = "Model";
    const char* AGENT_NAME = "Agent";
    const char* MESSAGE_NAME = "Message";
    const char* IN_FUNCTION_NAME = "InFunction";
    const char* OUT_FUNCTION_NAME = "OutFunction";
    const char* IN_LAYER_NAME = "InLayer";
    const char* OUT_LAYER_NAME = "OutLayer";
    const unsigned int SQRT_AGENT_COUNT = 64;
    ModelDescription m(MODEL_NAME);
    MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>(MESSAGE_NAME);
    message.setMin(0, 0);
    message.setMax(static_cast<float>(SQRT_AGENT_COUNT), static_cast<float>(SQRT_AGENT_COUNT));
    message.setRadius(1);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 2>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func_glm);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    AgentVector pop(a, SQRT_AGENT_COUNT * SQRT_AGENT_COUNT);
    int k = 0;
    for (unsigned int i = 0; i < SQRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < SQRT_AGENT_COUNT; ++j) {
            AgentVector::Agent ai = pop[k++];
            ai.setVariable<unsigned int, 2>("index", { i, j });
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 2> index = ai.getVariable<unsigned int, 2>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[1] * 11);
    }
}
#else
TEST(Spatial2DMessageTest, DISABLED_ArrayVariable_glm) { }
TEST(RTCSpatial2DMessageTest, DISABLED_ArrayVariable_glm) { }
#endif

FLAMEGPU_AGENT_FUNCTION(inWrapped2D, MessageSpatial2D, MessageNone) {
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const flamegpu::id_t ID = FLAMEGPU->getID();
    unsigned int count = 0;
    unsigned int badCount = 0;
    float xSum = 0;
    float ySum = 0;
    // Count how many messages we recieved (including our own)
    // This is all those which fall within the 3x3x3 Moore neighbourhood
    // Not our search radius
    for (const auto& message : FLAMEGPU->message_in.wrap(x1, y1)) {
        const float x2 = message.getVirtualX(x1);
        const float y2 = message.getVirtualY(y1);
        float x21 = x2 - x1;
        float y21 = y2 - y1;
        const float distance = sqrt(x21 * x21 + y21 * y21);
        if (distance > FLAMEGPU->message_in.radius() ||
            (abs((x21)) != 2.0f && x2 != x1) ||
            (abs((y21)) != 2.0f && y2 != y1)
        ) {
            badCount++;
        } else {
            count++;
            if (message.getVariable<flamegpu::id_t>("id") != ID) {
                xSum += (x21);
                ySum += (y21);
            }
        }
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    FLAMEGPU->setVariable<unsigned int>("badCount", badCount);
    FLAMEGPU->setVariable<float>("result_x", xSum);
    FLAMEGPU->setVariable<float>("result_y", ySum);
    return ALIVE;
}
void wrapped_2d_test(const float x_offset, const float y_offset, const float out_of_bounds = 0) {
    std::unordered_map<int, unsigned int> bin_counts;
    // Construct model
    ModelDescription model("Spatial2DMessageTestModel");
    {   // Location message
        MessageSpatial2D::Description message = model.newMessage<MessageSpatial2D>("location");
        message.setMin(0 + x_offset, 0 + y_offset);
        message.setMax(30 + x_offset, 30 + y_offset);
        message.setRadius(3);  // With a grid of agents spaced 2 units apart, this configuration should give each agent 8 neighbours (assuming my basic maths guessing works out)
        message.newVariable<flamegpu::id_t>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("result_x");  // Sum all virtual X values, and this should equal 0 (or very close)
        agent.newVariable<float>("result_y");  // Sum all virtual X values, and this should equal 0 (or very close)
        agent.newVariable<unsigned int>("count");  // Count how many messages we receive
        agent.newVariable<unsigned int>("badCount");  // Count how many messages we receive that have bad data
        agent.newFunction("out", out_mandatory2D).setMessageOutput("location");
        agent.newFunction("in", inWrapped2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(out_mandatory2D);
    }
    {   // Layer #2
        LayerDescription layer = model.newLayer();
        layer.addAgentFunction(inWrapped2D);
    }
    CUDASimulation cudaSimulation(model);

    AgentVector population(model.Agent("agent"), 15u * 15u);  // This must fit the env dims/radius set out above
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        for (unsigned int i = 0; i < 15u; i++) {
            for (unsigned int j = 0; j < 15u; j++) {
                unsigned int k = i * 15u + j;
                AgentVector::Agent instance = population[k];
                instance.setVariable<float>("x", i * 2.0f + x_offset + out_of_bounds);
                instance.setVariable<float>("y", j * 2.0f + y_offset);
            }
        }
        cudaSimulation.setPopulationData(population);
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected
    cudaSimulation.getPopulationData(population);
    // Validate each agent has same result
    for (AgentVector::Agent ai : population) {
        EXPECT_EQ(0.0f, ai.getVariable<float>("result_x"));
        EXPECT_EQ(0.0f, ai.getVariable<float>("result_y"));
        EXPECT_LE(ai.getVariable<unsigned int>("badCount"), 18u);  // Vague maths relative to count, the value is not constant due to boundary alignment
        EXPECT_EQ(9u, ai.getVariable<unsigned int>("count"));
    }
}
TEST(Spatial2DMessageTest, Wrapped) {
    wrapped_2d_test(0.0f, 0.0f);
}
// Test that it doesn't fall over if the environment min is not 0, with a few configurations
TEST(Spatial2DMessageTest, Wrapped2) {
    wrapped_2d_test(141.0f, 0.0f);
}
TEST(Spatial2DMessageTest, Wrapped3) {
    wrapped_2d_test(0.0f, 3440.0f);
}
TEST(Spatial2DMessageTest, Wrapped4) {
    wrapped_2d_test(-2342.0f, 0.0f);
}
TEST(Spatial2DMessageTest, Wrapped5) {
    wrapped_2d_test(0.0f, -7540.0f);
}
TEST(Spatial2DMessageTest, Wrapped6) {
    wrapped_2d_test(-141.0f, 0.0f);
}
TEST(Spatial2DMessageTest, Wrapped7) {
    wrapped_2d_test(141.4f, -540.7f);
}
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
// Test that FLAMEGPU_SEATBELTS catches out of bounds messages
TEST(Spatial2DMessageTest, Wrapped_OutOfBounds) {
    EXPECT_THROW(wrapped_2d_test(141.0f, -540.0f, 200.0f), exception::DeviceError);
}
FLAMEGPU_AGENT_FUNCTION(in_wrapped_EnvDimsNotFactor, MessageSpatial2D, MessageNone) {
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    for (auto& t : FLAMEGPU->message_in.wrap(x1, y1)) {
        // Do nothing, it should throw a device exception
    }
    return ALIVE;
}
TEST(Spatial2DMessageTest, Wrapped_EnvDimsNotFactor) {
    // This tests that bug #1157 is fixed
    // When the interaction radius is not a factor of the width
    // that agent's near the max env bound all have the full interaction radius
    ModelDescription m("model");
    MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>("location");
    message.setMin(0, 0);
    message.setMax(50.1f, 50.1f);
    message.setRadius(10);
    message.newVariable<flamegpu::id_t>("id");  // unused by current test
    AgentDescription agent = m.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    AgentFunctionDescription fo = agent.newFunction("out", out_mandatory2D);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = agent.newFunction("in", in_wrapped_EnvDimsNotFactor);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer();
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer();
    li.addAgentFunction(fi);
    // Set pop in model
    CUDASimulation c(m);
    // Create an agent in the middle of each edge
    AgentVector population(agent, 1);
    // Initialise agents
    // Vertical pair that can interact
    // Top side
    AgentVector::Agent i1 = population[0];
    i1.setVariable<float>("x", 25.0f);
    i1.setVariable<float>("y", 25.0f);
    c.setPopulationData(population);
    c.SimulationConfig().steps = 1;
    EXPECT_THROW(c.simulate(), exception::DeviceError);
}
#else
TEST(Spatial2DMessageTest, DISABLED_Wrapped_OutOfBounds) { }
TEST(Spatial2DMessageTest, DISABLED_Wrapped_EnvDimsNotFactor) { }
#endif
FLAMEGPU_AGENT_FUNCTION(out_mandatory2D_OddStep, MessageNone, MessageSpatial2D) {
    if (FLAMEGPU->getStepCounter() % 2 == 0) {
        FLAMEGPU->message_out.setLocation(
            FLAMEGPU->getVariable<float>("x"),
            FLAMEGPU->getVariable<float>("y"));
    }
    return ALIVE;
}
FLAMEGPU_HOST_FUNCTION(create_agents_step_zero) {
    if (FLAMEGPU->getStepCounter() == 1) {
        auto agent = FLAMEGPU->agent("agent");
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, 5.0f);
        for (unsigned int i = 0; i < 2049; ++i) {
            auto instance = agent.newAgent();
            float pos[2] = { dist(rng), dist(rng) };
            instance.setVariable<float>("x", pos[0]);
            instance.setVariable<float>("y", pos[1]);
        }
    }
}
TEST(Spatial2DMessageTest, buffer_not_init) {
    // This tests that a bug is fixed
    // The bug occurred when a message list, yet to have messages output to it was used as a message input
    // This requires no agents at the first message output function during the second iteration
    // It does 4 iterations to ensure PBM is reset too.
    ModelDescription m("model");
    MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>("location");
    message.setMin(0, 0);
    message.setMax(5, 5);
    message.setRadius(1);
    AgentDescription agent = m.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
    agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
    AgentFunctionDescription fo = agent.newFunction("out", out_mandatory2D_OddStep);
    fo.setMessageOutput(message);
    fo.setMessageOutputOptional(true);
    AgentFunctionDescription fi = agent.newFunction("in", in2D);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer();
    lo.addAgentFunction(fo);
    LayerDescription la = m.newLayer();
    la.addHostFunction(create_agents_step_zero);
    LayerDescription li = m.newLayer();
    li.addAgentFunction(fi);
    // Set pop in model
    CUDASimulation c(m);
    c.SimulationConfig().steps = 4;
    EXPECT_NO_THROW(c.simulate());
}

FLAMEGPU_AGENT_FUNCTION(in_bounds_not_factor, MessageSpatial2D, MessageNone) {
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    unsigned int count = 0;
    // Count how many messages we received (including our own)
    for (const auto& message : FLAMEGPU->message_in(x1, y1)) {
        ++count;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return ALIVE;
}
TEST(Spatial2DMessageTest, bounds_not_factor_radius) {
    // This tests that bug #1157 is fixed
    // When the interaction radius is not a factor of the width
    // that agent's near the max env bound all have the full interaction radius
    ModelDescription m("model");
    MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>("location");
    message.setMin(0, 0);
    message.setMax(50.1f, 50.1f);
    message.setRadius(10);
    // Grid will be 6x6
    // 6th column/row should only be  0.1 wide of the environment
    // Bug would incorrectly divide the whole environment by 6
    // So bin widths would instead become 8.35 (down from 10)
    message.newVariable<flamegpu::id_t>("id");  // unused by current test
    AgentDescription agent = m.newAgent("agent");
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
    agent.newVariable<unsigned int>("count", 0);
    AgentFunctionDescription fo = agent.newFunction("out", out_mandatory2D);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = agent.newFunction("in", in_bounds_not_factor);
    fi.setMessageInput(message);
    LayerDescription lo = m.newLayer();
    lo.addAgentFunction(fo);
    LayerDescription li = m.newLayer();
    li.addAgentFunction(fi);
    // Set pop in model
    CUDASimulation c(m);
    // Create an agent in the middle of each edge
    AgentVector population(agent, 4);
    // Initialise agents
    // Vertical pair that can interact
    // Top side
    AgentVector::Agent i1 = population[0];
    i1.setVariable<float>("x", 10.0f);
    i1.setVariable<float>("y", 0.0f);
    // Top side inner
    AgentVector::Agent i2 = population[1];
    i2.setVariable<float>("x", 10.0f);
    i2.setVariable<float>("y", 18.0f);
    // Right side
    AgentVector::Agent i3 = population[2];
    i3.setVariable<float>("x", 50.1f);
    i3.setVariable<float>("y", 40.0f);
    // Horizontal pair that can interact
    // Right side inner
    AgentVector::Agent i4 = population[3];
    i4.setVariable<float>("x", 50.1f - 10.11f);
    i4.setVariable<float>("y", 40.0f);
    c.setPopulationData(population);
    c.SimulationConfig().steps = 1;
    EXPECT_NO_THROW(c.simulate());
    // Recover the results and check they match what was expected
    c.getPopulationData(population);
    // Validate each agent has same result
    for (AgentVector::Agent ai : population) {
        if (ai.getID() < 3) {
            EXPECT_EQ(2u, ai.getVariable<unsigned int>("count"));
        } else {
            EXPECT_EQ(1u, ai.getVariable<unsigned int>("count"));
        }
    }
}

}  // namespace test_message_spatial2d
}  // namespace flamegpu
