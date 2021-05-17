/**
* Tests of feature Spatial 3D messaging
*
* Tests cover:
* > validation on MsgBucket::Description
*/


#include "gtest/gtest.h"
#include "flamegpu/flame_api.h"
namespace flamegpu {


namespace test_message_bucket {
TEST(BucketMsgTest, DescriptionValidation) {
    ModelDescription model("BucketMsgTest");
    // Test description accessors
    MsgBucket::Description &message = model.newMessage<MsgBucket>("buckets");
    EXPECT_THROW(message.setUpperBound(0), InvalidArgument);  // Min should default to 0, this would mean no buckets
    EXPECT_NO_THROW(message.setLowerBound(10));
    EXPECT_NO_THROW(message.setUpperBound(11));
    EXPECT_THROW(message.setUpperBound(0), InvalidArgument);  // Max < Min
    EXPECT_NO_THROW(message.setUpperBound(12));
    EXPECT_THROW(message.setLowerBound(13), InvalidArgument);  // Min > Max
    EXPECT_THROW(message.setBounds(12, 12), InvalidArgument);  // Min == Max
    EXPECT_THROW(message.setBounds(13, 12), InvalidArgument);  // Min > Max
    EXPECT_NO_THROW(message.setBounds(12, 13));
    EXPECT_NO_THROW(message.newVariable<int>("somevar"));
}
TEST(BucketMsgTest, DataValidation) {
    ModelDescription model("BucketMsgTest");
    // Test Data copy constructor knows when bounds have not been init
    MsgBucket::Description &message = model.newMessage<MsgBucket>("buckets");
    EXPECT_THROW(CUDASimulation c(model), InvalidMessage);  // Max not set
    message.setLowerBound(1);  // It should default to 0
    EXPECT_THROW(CUDASimulation c(model), InvalidMessage);  // Max not set
    message.setUpperBound(10);
    EXPECT_NO_THROW(CUDASimulation c(model));
}
TEST(BucketMsgTest, reserved_name) {
    ModelDescription model("BucketMsgTest");
    MsgBucket::Description &message = model.newMessage<MsgBucket>("buckets");
    EXPECT_THROW(message.newVariable<int>("_"), ReservedName);
}

FLAMEGPU_AGENT_FUNCTION(out_mandatory, MsgNone, MsgBucket) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setKey(12 + (id/2));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optional, MsgNone, MsgBucket) {
    if (FLAMEGPU->getVariable<int>("do_output")) {
        int id = FLAMEGPU->getVariable<int>("id");
        FLAMEGPU->message_out.setVariable<int>("id", id);
        FLAMEGPU->message_out.setKey(12 + (id/2));
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optionalNone, MsgNone, MsgBucket) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(in, MsgBucket, MsgNone) {
    const int id = FLAMEGPU->getVariable<int>("id");
    const int id_m1 = id == 0 ? 0 : id-1;
    unsigned int count = 0;
    unsigned int sum = 0;
    for (auto &m : FLAMEGPU->message_in(12 + (id_m1/2))) {
        count++;
        sum += m.getVariable<int>("id");
    }
    FLAMEGPU->setVariable<unsigned int>("count1", count);
    FLAMEGPU->setVariable<unsigned int>("count2", FLAMEGPU->message_in(12 + (id_m1/2)).size());
    FLAMEGPU->setVariable<unsigned int>("sum", sum);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(in_range, MsgBucket, MsgNone) {
    const int id = FLAMEGPU->getVariable<int>("id");
    const int id_m4 = 12 + ((id / 8) * 4);
    unsigned int count = 0;
    unsigned int sum = 0;
    for (auto &m : FLAMEGPU->message_in(id_m4, id_m4 + 4)) {
        count++;
        sum += m.getVariable<int>("id");
    }
    FLAMEGPU->setVariable<unsigned int>("count1", count);
    FLAMEGPU->setVariable<unsigned int>("count2", FLAMEGPU->message_in(12 + id/2).size());
    FLAMEGPU->setVariable<unsigned int>("sum", sum);
    return ALIVE;
}
TEST(BucketMsgTest, Mandatory) {
    const int AGENT_COUNT = 1024;
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMsgTest");
    {   // MsgBucket::Description
        MsgBucket::Description &message = model.newMessage<MsgBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count1", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("count2", 0);  // Size of bucket as returned by size()
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        agent.newFunction("out", out_mandatory).setMessageOutput("bucket");
        agent.newFunction("in", in).setMessageInput("bucket");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_mandatory);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in);
    }
    CUDASimulation cuda_model(model);

    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            instance.setVariable<int>("id", i);
            // Create it if it doesn't already exist
            if (bucket_count.find(i/2) == bucket_count.end()) {
                bucket_count.emplace(i/2, 0);
                bucket_sum.emplace(i/2, 0);
            }
            bucket_count[i/2] += 1;
            bucket_sum[i/2] += i;
        }
        cuda_model.setPopulationData(population);
    }

    // Execute a single step of the model
    cuda_model.step();

    // Recover the results and check they match what was expected
    cuda_model.getPopulationData(population);
    // Validate each agent has correct result
    for (AgentVector::Agent ai : population) {
        const int id = ai.getVariable<int>("id");
        const int id_m1 = id == 0 ? 0 : id-1;
        unsigned int count1 = ai.getVariable<unsigned int>("count1");
        unsigned int count2 = ai.getVariable<unsigned int>("count2");
        unsigned int sum = ai.getVariable<unsigned int>("sum");
        EXPECT_EQ(count1, bucket_count.at(id_m1/2));
        EXPECT_EQ(count2, bucket_count.at(id_m1/2));
        EXPECT_EQ(sum, bucket_sum.at(id_m1/2));
    }
}
TEST(BucketMsgTest, Optional) {
    const int AGENT_COUNT = 1024;
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMsgTest");
    {   // MsgBucket::Description
        MsgBucket::Description &message = model.newMessage<MsgBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<int>("do_output");
        agent.newVariable<unsigned int>("count1", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("count2", 0);  // Size of bucket as returned by size()
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        auto &af = agent.newFunction("out", out_optional);
        af.setMessageOutput("bucket");
        af.setMessageOutputOptional(true);
        agent.newFunction("in", in).setMessageInput("bucket");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_optional);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in);
    }
    CUDASimulation cuda_model(model);

    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            int do_out =  dist(rng) > 0.3 ? 1 : 0;
            AgentVector::Agent instance = population[i];
            instance.setVariable<int>("id", i);
            instance.setVariable<int>("do_output", do_out);
            // Create it if it doesn't already exist
            if (bucket_count.find(i/2) == bucket_count.end()) {
                bucket_count.emplace(i/2, 0);
                bucket_sum.emplace(i/2, 0);
            }
            if (do_out) {
                bucket_count[i/2] += 1;
                bucket_sum[i/2] += i;
            }
        }
        cuda_model.setPopulationData(population);
    }

    // Execute a single step of the model
    cuda_model.step();

    // Recover the results and check they match what was expected
    cuda_model.getPopulationData(population);
    // Validate each agent has correct result
    for (AgentVector::Agent ai : population) {
        const int id = ai.getVariable<int>("id");
        const int id_m1 = id == 0 ? 0 : id-1;
        unsigned int count1 = ai.getVariable<unsigned int>("count1");
        unsigned int count2 = ai.getVariable<unsigned int>("count2");
        unsigned int sum = ai.getVariable<unsigned int>("sum");
        EXPECT_EQ(count1, bucket_count.at(id_m1/2));
        EXPECT_EQ(count2, bucket_count.at(id_m1/2));
        EXPECT_EQ(sum, bucket_sum.at(id_m1/2));
    }
}
// Test optional message output, wehre no messages are output.
TEST(BucketMsgTest, OptionalNone) {
    const int AGENT_COUNT = 1024;
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMsgTest");
    {   // MsgBucket::Description
        MsgBucket::Description &message = model.newMessage<MsgBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count1", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("count2", 0);  // Size of bucket as returned by size()
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        auto &af = agent.newFunction("out", out_optionalNone);
        af.setMessageOutput("bucket");
        af.setMessageOutputOptional(true);
        agent.newFunction("in", in).setMessageInput("bucket");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_optionalNone);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in);
    }
    CUDASimulation cuda_model(model);

    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            instance.setVariable<int>("id", i);
            // Create it if it doesn't already exist
            if (bucket_count.find(i/2) == bucket_count.end()) {
                bucket_count.emplace(i/2, 0);
                bucket_sum.emplace(i/2, 0);
            }
        }
        cuda_model.setPopulationData(population);
    }

    // Execute a single step of the model
    cuda_model.step();

    // Recover the results and check they match what was expected
    cuda_model.getPopulationData(population);
    // Validate each agent has correct result
    for (AgentVector::Agent ai : population) {
        unsigned int count1 = ai.getVariable<unsigned int>("count1");
        unsigned int count2 = ai.getVariable<unsigned int>("count2");
        unsigned int sum = ai.getVariable<unsigned int>("sum");
        EXPECT_EQ(0u, count1);
        EXPECT_EQ(0u, count2);
        EXPECT_EQ(0u, sum);
    }
}
TEST(BucketMsgTest, Mandatory_Range) {
    // Agent count must be multiple of 4
    const int AGENT_COUNT = 1024;
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMsgTest");
    {   // MsgBucket::Description
        MsgBucket::Description &message = model.newMessage<MsgBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription &agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count1", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("count2", 0);  // Size of id bucket as returned by size()
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        agent.newFunction("out", out_mandatory).setMessageOutput("bucket");
        agent.newFunction("in", in_range).setMessageInput("bucket");
    }
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(out_mandatory);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(in_range);
    }
    CUDASimulation cuda_model(model);

    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentVector::Agent instance = population[i];
            instance.setVariable<int>("id", i);
            // Create it if it doesn't already exist
            if (bucket_count.find(i/2) == bucket_count.end()) {
                bucket_count.emplace(i/2, 0);
                bucket_sum.emplace(i/2, 0);
            }
            bucket_count[i/2] += 1;
            bucket_sum[i/2] += i;
        }
        cuda_model.setPopulationData(population);
    }

    // Execute a single step of the model
    cuda_model.step();

    // Recover the results and check they match what was expected
    cuda_model.getPopulationData(population);
    // Validate each agent has correct result
    for (AgentVector::Agent ai : population) {
        const int id = ai.getVariable<int>("id");
        const int id_m4 = ((id / 8) * 4);
        unsigned int count1 = ai.getVariable<unsigned int>("count1");
        unsigned int count2 = ai.getVariable<unsigned int>("count2");
        unsigned int sum = ai.getVariable<unsigned int>("sum");
        unsigned int _count1 = 0;
        unsigned int _sum = 0;
        for (int j = 0; j < 4; ++j) {
            _count1 += bucket_count.at(id_m4 + j);
            _sum += bucket_sum.at(id_m4 + j);
        }
        EXPECT_EQ(count1, _count1);
        EXPECT_EQ(count2, bucket_count.at(id/2));
        EXPECT_EQ(sum, _sum);
    }
}

}  // namespace test_message_bucket
}  // namespace flamegpu
