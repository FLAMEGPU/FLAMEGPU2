/**
* Tests of feature Spatial 3D messaging
*
* Tests cover:
* > mandatory messaging, send/recieve
*/

#include "gtest/gtest.h"

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


FLAMEGPU_AGENT_FUNCTION(out_mandatory, MsgNone, MsgSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(in, MsgSpatial3D, MsgNone) {
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    unsigned int count = 0;
    unsigned int myBin[3] = {
        static_cast<unsigned int>(x1),
        static_cast<unsigned int>(y1),
        static_cast<unsigned int>(z1)
    };
    const unsigned int bin_index =
        myBin[2] * 5 * 5 +
        myBin[1] * 5 +
        myBin[0];
    // Count how many messages we recieved (including our own)
    // This is all those which fall within the 3x3x3 Moore neighbourhood
    // Not our search radius
    for (const auto &message : FLAMEGPU->message_in(x1, y1, z1)) {
        unsigned int msgBin[3] = {
            static_cast<unsigned int>(message.getVariable<float>("x")),
            static_cast<unsigned int>(message.getVariable<float>("y")),
            static_cast<unsigned int>(message.getVariable<float>("z"))
        };
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return ALIVE;
}
TEST(Spatial3DMsgTest, Mandatory) {
    std::unordered_map<int, unsigned int> bin_counts;
    // Construct model
    ModelDescription model("Spatial3DMsgTestModel");
    {   // Location message
        Spatial3DMessageDescription &message = model.newSpatial3DMessage("location");
        message.setMin(0, 0, 0);
        message.setMax(5, 5, 5);
        message.setRadius(1);
        //5x5x5 bins, total 125
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newVariable<unsigned int>("myBin"); // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newFunction("out", out_mandatory).setMessageOutput("location");
        agent.newFunction("in", in).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_mandatory);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in);
    }
    CUDAAgentModel cuda_model(model);

    const int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 5.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentInstance instance = population.getNextInstance();
            instance.setVariable<int>("id", i);
            float pos[3] = { dist(rng), dist(rng), dist(rng) };
            instance.setVariable<float>("x", pos[0]);
            instance.setVariable<float>("y", pos[1]);
            instance.setVariable<float>("z", pos[2]);
            // Solve the bin index
            const unsigned int bin_pos[3] = {
                (unsigned int)(pos[0] / 1),
                (unsigned int)(pos[1] / 1),
                (unsigned int)(pos[2] / 1)
            };
            const unsigned int bin_index =
                bin_pos[2] * 5 * 5 +
                bin_pos[1] * 5 +
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
    for (unsigned int x1 = 0; x1 < 5; x1++) {
        for (unsigned int y1 = 0; y1 < 5; y1++) {
            for (unsigned int z1 = 0; z1 < 5; z1++) {
                // Solve the bin index
                const unsigned int bin_pos1[3] = {
                    x1,
                    y1,
                    z1
                };
                const unsigned int bin_index1 =
                    bin_pos1[2] * 5 * 5 +
                    bin_pos1[1] * 5 +
                    bin_pos1[0];
                // Count our neighbours
                unsigned int count_sum = 0;
                for (int x2 = -1; x2 <= 1; x2++) {
                    int bin_pos2[3] = {
                        (int)bin_pos1[0] + x2,
                        0,
                        0
                    };
                    for (int y2 = -1; y2 <= 1; y2++) {
                        bin_pos2[1] = (int)bin_pos1[1] + y2;
                        for (int z2 = -1; z2 <= 1; z2++) {
                            bin_pos2[2] = (int)bin_pos1[2] + z2;
                            // Ensure bin is in bounds
                            if (
                                bin_pos2[0] >= 0 &&
                                bin_pos2[1] >= 0 &&
                                bin_pos2[2] >= 0 &&
                                bin_pos2[0] < 5 &&
                                bin_pos2[1] < 5 &&
                                bin_pos2[2] < 5
                                ) {
                                const unsigned int bin_index2 =
                                    bin_pos2[2] * 5 * 5 +
                                    bin_pos2[1] * 5 +
                                    bin_pos2[0];
                                count_sum += bin_counts[bin_index2];
                            }
                        }
                    }
                }
                bin_results.emplace(bin_index1, count_sum);
            }
        }
    }

    // Execute a single step of the model
    cuda_model.step();

    // Recover the results and check they match what was expected

    cuda_model.getPopulationData(population);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        unsigned int myBin = ai.getVariable<unsigned int>("myBin");
        unsigned int myResult = ai.getVariable<unsigned int>("count");
        EXPECT_EQ(myResult, bin_results.at(myBin));
    }
}