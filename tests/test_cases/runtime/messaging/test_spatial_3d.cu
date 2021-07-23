/**
* Tests of feature Spatial 3D messaging
*
* Tests cover:
* > mandatory messaging, send/recieve
*/
#include "flamegpu/flamegpu.h"

#include "gtest/gtest.h"

namespace flamegpu {


namespace test_message_spatial3d {

FLAMEGPU_AGENT_FUNCTION(out_mandatory3D, MsgNone, MsgSpatial3D) {
    FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optional3D, MsgNone, MsgSpatial3D) {
    if (FLAMEGPU->getVariable<int>("do_output")) {
        FLAMEGPU->message_out.setVariable<int>("id", FLAMEGPU->getVariable<int>("id"));
        FLAMEGPU->message_out.setLocation(
            FLAMEGPU->getVariable<float>("x"),
            FLAMEGPU->getVariable<float>("y"),
            FLAMEGPU->getVariable<float>("z"));
    }
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(out_optional3DNone, MsgNone, MsgSpatial3D) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(in3D, MsgSpatial3D, MsgNone) {
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    unsigned int count = 0;
    unsigned int badCount = 0;
     int myBin[3] = {
         static_cast<int>(x1),
         static_cast<int>(y1),
         static_cast<int>(z1)
     };
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3x3 Moore neighbourhood
    // Not our search radius
    for (const auto &message : FLAMEGPU->message_in(x1, y1, z1)) {
         int msgBin[3] = {
             static_cast<int>(message.getVariable<float>("x")),
             static_cast<int>(message.getVariable<float>("y")),
             static_cast<int>(message.getVariable<float>("z"))
        };
        bool isBad = false;
        for (unsigned int i = 0; i < 3; ++i) {  // Iterate axis
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
TEST(Spatial3DMsgTest, Mandatory) {
    std::unordered_map<int, unsigned int> bin_counts;
    // Construct model
    ModelDescription model("Spatial3DMsgTestModel");
    {   // Location message
        MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
        message.setMin(0, 0, 0);
        message.setMax(5, 5, 5);
        message.setRadius(1);
        // 5x5x5 bins, total 125
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        agent.newFunction("out", out_mandatory3D).setMessageOutput("location");
        agent.newFunction("in", in3D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_mandatory3D);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in3D);
    }
    CUDASimulation cudaSimulation(model);

    const int AGENT_COUNT = 2049;
    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 5.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
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
        cudaSimulation.setPopulationData(population);
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
                        static_cast<int>(bin_pos1[0]) + x2,
                        0,
                        0
                    };
                    for (int y2 = -1; y2 <= 1; y2++) {
                        bin_pos2[1] = static_cast<int>(bin_pos1[1]) + y2;
                        for (int z2 = -1; z2 <= 1; z2++) {
                            bin_pos2[2] = static_cast<int>(bin_pos1[2]) + z2;
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

TEST(Spatial3DMsgTest, Optional) {
    /**
     * This test is same as Mandatory, however extra flag has been added to block certain agents from outputting messages
     * Look for NEW!
     */
    std::unordered_map<int, unsigned int> bin_counts;
    std::unordered_map<int, unsigned int> bin_counts_optional;
    // Construct model
    ModelDescription model("Spatial3DMsgTestModel");
    {   // Location message
        MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
        message.setMin(0, 0, 0);
        message.setMax(5, 5, 5);
        message.setRadius(1);
        // 5x5x5 bins, total 125
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newVariable<int>("do_output");  // NEW!
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        auto &af = agent.newFunction("out", out_optional3D);  // NEW!
        af.setMessageOutput("location");
        af.setMessageOutputOptional(true);  // NEW!
        agent.newFunction("in", in3D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_optional3D);  // NEW!
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in3D);
    }
    CUDASimulation cudaSimulation(model);

    const int AGENT_COUNT = 2049;
    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 5.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            instance.setVariable<int>("id", i);
            float pos[3] = { dist(rng), dist(rng), dist(rng) };
            int do_output = dist(rng) < 4 ? 1 : 0;  // 80% chance of output  // NEW!
            instance.setVariable<float>("x", pos[0]);
            instance.setVariable<float>("y", pos[1]);
            instance.setVariable<float>("z", pos[2]);
            instance.setVariable<int>("do_output", do_output);  // NEW!
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
                unsigned int count_sum_optional = 0;  // NEW!
                for (int x2 = -1; x2 <= 1; x2++) {
                    int bin_pos2[3] = {
                        static_cast<int>(bin_pos1[0]) + x2,
                        0,
                        0
                    };
                    for (int y2 = -1; y2 <= 1; y2++) {
                        bin_pos2[1] = static_cast<int>(bin_pos1[1]) + y2;
                        for (int z2 = -1; z2 <= 1; z2++) {
                            bin_pos2[2] = static_cast<int>(bin_pos1[2]) + z2;
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
                                count_sum_optional += bin_counts_optional[bin_index2];  // NEW!
                            }
                        }
                    }
                }
                bin_results.emplace(bin_index1, count_sum);
                bin_results_optional.emplace(bin_index1, count_sum_optional);  // NEW!
            }
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
// Test optional message output, with no messaegs
TEST(Spatial3DMsgTest, OptionalNone) {
    /**
     * This test is same as Mandatory, however extra flag has been added to block certain agents from outputting messages
     * Look for NEW!
     */
    std::unordered_map<int, unsigned int> bin_counts;
    std::unordered_map<int, unsigned int> bin_counts_optional;
    // Construct model
    ModelDescription model("Spatial3DMsgTestModel");
    {   // Location message
        MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
        message.setMin(0, 0, 0);
        message.setMax(5, 5, 5);
        message.setRadius(1);
        // 5x5x5 bins, total 125
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newVariable<int>("do_output");  // NEW!
        agent.newVariable<unsigned int>("myBin");  // This will be presumed bin index of the agent, might not use this
        agent.newVariable<unsigned int>("count");  // Store the distance moved here, for validation
        agent.newVariable<unsigned int>("badCount");  // Store how many messages are out of range
        auto &af = agent.newFunction("out", out_optional3DNone);  // NEW!
        af.setMessageOutput("location");
        af.setMessageOutputOptional(true);  // NEW!
        agent.newFunction("in", in3D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_optional3DNone);  // NEW!
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in3D);
    }
    CUDASimulation cudaSimulation(model);

    const int AGENT_COUNT = 2049;
    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 5.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            instance.setVariable<int>("id", i);
            float pos[3] = { dist(rng), dist(rng), dist(rng) };
            int do_output = dist(rng) < 4 ? 1 : 0;  // 80% chance of output  // NEW!
            instance.setVariable<float>("x", pos[0]);
            instance.setVariable<float>("y", pos[1]);
            instance.setVariable<float>("z", pos[2]);
            instance.setVariable<int>("do_output", do_output);  // NEW!
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



TEST(Spatial3DMsgTest, BadRadius) {
    ModelDescription model("Spatial3DMsgTestModel");
    MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
    EXPECT_THROW(message.setRadius(0), exception::InvalidArgument);
    EXPECT_THROW(message.setRadius(-10), exception::InvalidArgument);
}
TEST(Spatial3DMsgTest, BadMin) {
    ModelDescription model("Spatial3DMsgTestModel");
    MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
    message.setMax(5, 5, 5);
    EXPECT_THROW(message.setMin(5, 0, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(0, 5, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(0, 0, 5), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(6, 0, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(0, 6, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMin(0, 0, 6), exception::InvalidArgument);
}
TEST(Spatial3DMsgTest, BadMax) {
    ModelDescription model("Spatial3DMsgTestModel");
    MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
    message.setMin(5, 5, 5);
    EXPECT_THROW(message.setMax(5, 0, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(0, 5, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(0, 0, 5), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(4, 0, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(0, 4, 0), exception::InvalidArgument);
    EXPECT_THROW(message.setMax(0, 0, 4), exception::InvalidArgument);
}
TEST(Spatial3DMsgTest, UnsetMax) {
    ModelDescription model("Spatial23MsgTestModel");
    MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
    message.setMin(5, 5, 5);
    EXPECT_THROW(CUDASimulation m(model), exception::InvalidMessage);
}
TEST(Spatial3DMsgTest, UnsetMin) {
    ModelDescription model("Spatial3DMsgTestModel");
    MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
    message.setMin(5, 5, 5);
    EXPECT_THROW(CUDASimulation m(model), exception::InvalidMessage);
}
TEST(Spatial3DMsgTest, reserved_name) {
    ModelDescription model("Spatial3DMsgTestModel");
    MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
    EXPECT_THROW(message.newVariable<int>("_"), exception::ReservedName);
}

FLAMEGPU_AGENT_FUNCTION(count3D, MsgSpatial3D, MsgNone) {
    unsigned int count = 0;
    // Count how many messages we received (including our own)
    // This is all those which fall within the 3x3x3 Moore neighbourhood
    for (const auto &message : FLAMEGPU->message_in(0, 0, 0)) {
        count++;
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    return ALIVE;
}
TEST(Spatial3DMsgTest, ReadEmpty) {
// What happens if we read a message list before it has been output?
    ModelDescription model("Model");
    {   // Location message
        MsgSpatial3D::Description &message = model.newMessage<MsgSpatial3D>("location");
        message.setMin(-3, -3, -3);
        message.setMax(3, 3, 3);
        message.setRadius(2);
        message.newVariable<int>("id");  // unused by current test
    }
    {   // Circle agent
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<unsigned int>("count", 0);  // Count the number of messages read
        agent.newFunction("in", count3D).setMessageInput("location");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(count3D);
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



FLAMEGPU_AGENT_FUNCTION(ArrayOut, MsgNone, MsgSpatial3D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 3>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 3>("index", 1);
    const unsigned int z = FLAMEGPU->getVariable<unsigned int, 3>("index", 2);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, y * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, z * 11);
    FLAMEGPU->message_out.setLocation(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn, MsgSpatial3D, MsgNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 3>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 3>("index", 1);
    const unsigned int z = FLAMEGPU->getVariable<unsigned int, 3>("index", 2);
    for (auto &message : FLAMEGPU->message_in(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))) {
        if (static_cast<unsigned int>(message.getVariable<float>("x")) == x &&
            static_cast<unsigned int>(message.getVariable<float>("y")) == y &&
            static_cast<unsigned int>(message.getVariable<float>("z")) == z) {
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
            break;
        }
    }
    return ALIVE;
}
TEST(Spatial3DMsgTest, ArrayVariable) {
    const char* MODEL_NAME = "Model";
    const char* AGENT_NAME = "Agent";
    const char* MESSAGE_NAME = "Message";
    const char* IN_FUNCTION_NAME = "InFunction";
    const char* OUT_FUNCTION_NAME = "OutFunction";
    const char* IN_LAYER_NAME = "InLayer";
    const char* OUT_LAYER_NAME = "OutLayer";
    const unsigned int CBRT_AGENT_COUNT = 11;
    ModelDescription m(MODEL_NAME);
    MsgSpatial3D::Description &msg = m.newMessage<MsgSpatial3D>(MESSAGE_NAME);
    msg.setMin(0, 0, 0);
    msg.setMax(static_cast<float>(CBRT_AGENT_COUNT), static_cast<float>(CBRT_AGENT_COUNT), static_cast<float>(CBRT_AGENT_COUNT));
    msg.setRadius(1);
    msg.newVariable<unsigned int, 3>("v");
    AgentDescription &a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 3>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription &fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(msg);
    AgentFunctionDescription &fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn);
    fi.setMessageInput(msg);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    AgentVector pop(a, CBRT_AGENT_COUNT * CBRT_AGENT_COUNT * CBRT_AGENT_COUNT);
    int t = 0;
    for (unsigned int i = 0; i < CBRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < CBRT_AGENT_COUNT; ++j) {
            for (unsigned int k = 0; k < CBRT_AGENT_COUNT; ++k) {
                AgentVector::Agent ai = pop[t++];
                ai.setVariable<unsigned int, 3>("index", { i, j, k });
            }
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 3> index = ai.getVariable<unsigned int, 3>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[2] * 11);
    }
}
const char* rtc_ArrayOut_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MsgNone, flamegpu::MsgSpatial3D) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 3>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 3>("index", 1);
    const unsigned int z = FLAMEGPU->getVariable<unsigned int, 3>("index", 2);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, x * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, y * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, z * 11);
    FLAMEGPU->message_out.setLocation(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MsgSpatial3D, flamegpu::MsgNone) {
    const unsigned int x = FLAMEGPU->getVariable<unsigned int, 3>("index", 0);
    const unsigned int y = FLAMEGPU->getVariable<unsigned int, 3>("index", 1);
    const unsigned int z = FLAMEGPU->getVariable<unsigned int, 3>("index", 2);
    for (auto &message : FLAMEGPU->message_in(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z))) {
        if (static_cast<unsigned int>(message.getVariable<float>("x")) == x &&
            static_cast<unsigned int>(message.getVariable<float>("y")) == y &&
            static_cast<unsigned int>(message.getVariable<float>("z")) == z) {
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
            FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
            break;
        }
    }
    return flamegpu::ALIVE;
}
)###";
TEST(RTCSpatial3DMsgTest, ArrayVariable) {
    const char* MODEL_NAME = "Model";
    const char* AGENT_NAME = "Agent";
    const char* MESSAGE_NAME = "Message";
    const char* IN_FUNCTION_NAME = "InFunction";
    const char* OUT_FUNCTION_NAME = "OutFunction";
    const char* IN_LAYER_NAME = "InLayer";
    const char* OUT_LAYER_NAME = "OutLayer";
    const unsigned int CBRT_AGENT_COUNT = 11;
    ModelDescription m(MODEL_NAME);
    MsgSpatial3D::Description& msg = m.newMessage<MsgSpatial3D>(MESSAGE_NAME);
    msg.setMin(0, 0, 0);
    msg.setMax(static_cast<float>(CBRT_AGENT_COUNT), static_cast<float>(CBRT_AGENT_COUNT), static_cast<float>(CBRT_AGENT_COUNT));
    msg.setRadius(1);
    msg.newVariable<unsigned int, 3>("v");
    AgentDescription& a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int, 3>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription& fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func);
    fi.setMessageInput(msg);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    AgentVector pop(a, CBRT_AGENT_COUNT * CBRT_AGENT_COUNT * CBRT_AGENT_COUNT);
    int t = 0;
    for (unsigned int i = 0; i < CBRT_AGENT_COUNT; ++i) {
        for (unsigned int j = 0; j < CBRT_AGENT_COUNT; ++j) {
            for (unsigned int k = 0; k < CBRT_AGENT_COUNT; ++k) {
                AgentVector::Agent ai = pop[t++];
                ai.setVariable<unsigned int, 3>("index", { i, j, k });
            }
        }
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const std::array<unsigned int, 3> index = ai.getVariable<unsigned int, 3>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index[0] * 3);
        ASSERT_EQ(v[1], index[1] * 7);
        ASSERT_EQ(v[2], index[2] * 11);
    }
}

}  // namespace test_message_spatial3d
}  // namespace flamegpu
