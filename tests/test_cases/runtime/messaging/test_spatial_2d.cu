/**
* Tests of feature Spatial 3D messaging
*
* Tests cover:
* > mandatory messaging, send/recieve
*/

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


namespace test_message_spatial2d {

FLAMEGPU_AGENT_FUNCTION(out_mandatory2D, MsgNone, MsgSpatial2D) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optional2D, MsgNone, MsgSpatial2D) {
    if (FLAMEGPU->getVariable<int>("do_output")) {
        FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
        FLAMEGPU->message_out.setLocation(
            FLAMEGPU->getVariable<float>("x"),
            FLAMEGPU->getVariable<float>("y"));
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(in2D, MsgSpatial2D, MsgNone) {
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
         unsigned int msgBin[2] = {
             static_cast<unsigned int>(message.getVariable<float>("x")),
             static_cast<unsigned int>(message.getVariable<float>("y"))
         };
         bool isBad = false;
         for (unsigned int i = 0; i < 2; ++i) {  // Iterate axis
             int binDiff = myBin[i] - msgBin[i];
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
TEST(Spatial2DMsgTest, Mandatory) {
    std::unordered_map<int, unsigned int> bin_counts;
    // Construct model
    ModelDescription model("Spatial2DMsgTestModel");
    {   // Location message
        MsgSpatial2D::Description &message = model.newMessage<MsgSpatial2D>("location");
        message.setMin(0, 0);
        message.setMax(11, 11);
        message.setRadius(1);
        // 11x11 bins, total 121
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        agent.newFunction("out", out_mandatory2D).setMessageOutput("location");
        agent.newFunction("in", in2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_mandatory2D);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in2D);
    }
    CUDAAgentModel cuda_model(model);

    const int AGENT_COUNT = 2049;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 11.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentInstance instance = population.getNextInstance();
            instance.setVariable<int>("id", i);
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
        cuda_model.setPopulationData(population);
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
    cuda_model.step();

    // Recover the results and check they match what was expected

    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    unsigned int badCountWrong = 0;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        unsigned int myBin = ai.getVariable<unsigned int>("myBin");
        unsigned int myResult = ai.getVariable<unsigned int>("count");
        EXPECT_EQ(myResult, bin_results.at(myBin));
        if (ai.getVariable<unsigned int>("badCount"))
            badCountWrong++;
    }
    EXPECT_EQ(badCountWrong, 0u);
}

TEST(Spatial2DMsgTest, Optional) {
    /**
     * This test is same as Mandatory, however extra flag has been added to block certain agents from outputting messages
     * Look for NEW!
     */
    std::unordered_map<int, unsigned int> bin_counts;
    std::unordered_map<int, unsigned int> bin_counts_optional;
    // Construct model
    ModelDescription model("Spatial2DMsgTestModel");
    {   // Location message
        MsgSpatial2D::Description &message = model.newMessage<MsgSpatial2D>("location");
        message.setMin(0, 0);
        message.setMax(11, 11);
        message.setRadius(1);
        // 11x11 bins, total 121
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<int>("do_output");  // NEW!
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        auto &af = agent.newFunction("out", out_optional2D);  // NEW!
        af.setMessageOutput("location");
        af.setMessageOutputOptional(true);  // NEW!
        agent.newFunction("in", in2D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_optional2D);  // NEW!
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in2D);
    }
    CUDAAgentModel cuda_model(model);

    const int AGENT_COUNT = 2049;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 11.0f);
        std::uniform_real_distribution<float> dist5(0.0f, 5.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentInstance instance = population.getNextInstance();
            instance.setVariable<int>("id", i);
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
        cuda_model.setPopulationData(population);
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
    cuda_model.step();

    // Recover the results and check they match what was expected

    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    unsigned int badCountWrong = 0;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        unsigned int myBin = ai.getVariable<unsigned int>("myBin");
        unsigned int myResult = ai.getVariable<unsigned int>("count");
        if (ai.getVariable<unsigned int>("badCount"))
            badCountWrong++;
        EXPECT_EQ(myResult, bin_results_optional.at(myBin));  // NEW!
    }
    EXPECT_EQ(badCountWrong, 0u);
}
TEST(Spatial2DMsgTest, BadRadius) {
    ModelDescription model("Spatial2DMsgTestModel");
    MsgSpatial2D::Description &message = model.newMessage<MsgSpatial2D>("location");
    EXPECT_THROW(message.setRadius(0), InvalidArgument);
    EXPECT_THROW(message.setRadius(-10), InvalidArgument);
}
TEST(Spatial2DMsgTest, BadMin) {
    ModelDescription model("Spatial2DMsgTestModel");
    MsgSpatial2D::Description &message = model.newMessage<MsgSpatial2D>("location");
    message.setMax(5, 5);
    EXPECT_THROW(message.setMin(5, 0), InvalidArgument);
    EXPECT_THROW(message.setMin(0, 5), InvalidArgument);
    EXPECT_THROW(message.setMin(6, 0), InvalidArgument);
    EXPECT_THROW(message.setMin(0, 6), InvalidArgument);
}
TEST(Spatial2DMsgTest, BadMax) {
    ModelDescription model("Spatial2DMsgTestModel");
    MsgSpatial2D::Description &message = model.newMessage<MsgSpatial2D>("location");
    message.setMin(5, 5);
    EXPECT_THROW(message.setMax(5, 0), InvalidArgument);
    EXPECT_THROW(message.setMax(0, 5), InvalidArgument);
    EXPECT_THROW(message.setMax(4, 0), InvalidArgument);
    EXPECT_THROW(message.setMax(0, 4), InvalidArgument);
}
TEST(Spatial2DMsgTest, UnsetMax) {
    ModelDescription model("Spatial2DMsgTestModel");
    MsgSpatial2D::Description &message = model.newMessage<MsgSpatial2D>("location");
    message.setMin(5, 5);
    EXPECT_THROW(CUDAAgentModel m(model), InvalidMessage);
}
TEST(Spatial2DMsgTest, UnsetMin) {
    ModelDescription model("Spatial2DMsgTestModel");
    MsgSpatial2D::Description &message = model.newMessage<MsgSpatial2D>("location");
    message.setMin(5, 5);
    EXPECT_THROW(CUDAAgentModel m(model), InvalidMessage);
}

}  // namespace test_message_spatial2d
