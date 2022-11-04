/**
* Tests of feature Spatial 3D messaging
*
* Tests cover:
* > validation on MessageBucket::Description
*/


#include "gtest/gtest.h"
#include "flamegpu/flamegpu.h"
namespace flamegpu {


namespace test_message_bucket {
    const char* MODEL_NAME = "Model";
    const char* AGENT_NAME = "Agent";
    const char* MESSAGE_NAME = "Message";
    const char* IN_FUNCTION_NAME = "InFunction";
    const char* OUT_FUNCTION_NAME = "OutFunction";
    const char* IN_LAYER_NAME = "InLayer";
    const char* OUT_LAYER_NAME = "OutLayer";
    const unsigned int AGENT_COUNT = 1024;  // must be a multiple of 4
TEST(BucketMessageTest, DescriptionValidation) {
    ModelDescription model("BucketMessageTest");
    // Test description accessors
    MessageBucket::Description &message = model.newMessage<MessageBucket>("buckets");
    EXPECT_THROW(message.setUpperBound(0), exception::InvalidArgument);  // Min should default to 0, this would mean no buckets
    EXPECT_NO_THROW(message.setLowerBound(10));
    EXPECT_NO_THROW(message.setUpperBound(11));
    EXPECT_THROW(message.setUpperBound(0), exception::InvalidArgument);  // Max < Min
    EXPECT_NO_THROW(message.setUpperBound(12));
    EXPECT_THROW(message.setLowerBound(13), exception::InvalidArgument);  // Min > Max
    EXPECT_THROW(message.setBounds(12, 12), exception::InvalidArgument);  // Min == Max
    EXPECT_THROW(message.setBounds(13, 12), exception::InvalidArgument);  // Min > Max
    EXPECT_NO_THROW(message.setBounds(12, 13));
    EXPECT_NO_THROW(message.newVariable<int>("somevar"));
}
TEST(BucketMessageTest, DataValidation) {
    ModelDescription model("BucketMessageTest");
    // Test Data copy constructor knows when bounds have not been init
    MessageBucket::Description &message = model.newMessage<MessageBucket>("buckets");
    EXPECT_THROW(CUDASimulation c(model), exception::InvalidMessage);  // Max not set
    message.setLowerBound(1);  // It should default to 0
    EXPECT_THROW(CUDASimulation c(model), exception::InvalidMessage);  // Max not set
    message.setUpperBound(10);
    EXPECT_NO_THROW(CUDASimulation c(model));
}
TEST(BucketMessageTest, reserved_name) {
    ModelDescription model("BucketMessageTest");
    MessageBucket::Description &message = model.newMessage<MessageBucket>("buckets");
    EXPECT_THROW(message.newVariable<int>("_"), exception::ReservedName);
}

FLAMEGPU_AGENT_FUNCTION(out_mandatory, MessageNone, MessageBucket) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setKey(12 + (id/2));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optional, MessageNone, MessageBucket) {
    if (FLAMEGPU->getVariable<int>("do_output")) {
        int id = FLAMEGPU->getVariable<int>("id");
        FLAMEGPU->message_out.setVariable<int>("id", id);
        FLAMEGPU->message_out.setKey(12 + (id/2));
    }
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(out_optionalNone, MessageNone, MessageBucket) {
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(in, MessageBucket, MessageNone) {
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
FLAMEGPU_AGENT_FUNCTION(in_range, MessageBucket, MessageNone) {
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
TEST(BucketMessageTest, Mandatory) {
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMessageTest");
    {   // MessageBucket::Description
        MessageBucket::Description &message = model.newMessage<MessageBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription agent = model.newAgent("agent");
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
    CUDASimulation cudaSimulation(model);

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
        cudaSimulation.setPopulationData(population);
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected
    cudaSimulation.getPopulationData(population);
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
TEST(BucketMessageTest, Optional) {
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMessageTest");
    {   // MessageBucket::Description
        MessageBucket::Description &message = model.newMessage<MessageBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<int>("do_output");
        agent.newVariable<unsigned int>("count1", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("count2", 0);  // Size of bucket as returned by size()
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        auto af = agent.newFunction("out", out_optional);
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
    CUDASimulation cudaSimulation(model);

    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::mt19937_64 rng;
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
        cudaSimulation.setPopulationData(population);
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected
    cudaSimulation.getPopulationData(population);
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
TEST(BucketMessageTest, OptionalNone) {
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMessageTest");
    {   // MessageBucket::Description
        MessageBucket::Description &message = model.newMessage<MessageBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription agent = model.newAgent("agent");
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count1", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("count2", 0);  // Size of bucket as returned by size()
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        auto af = agent.newFunction("out", out_optionalNone);
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
    CUDASimulation cudaSimulation(model);

    AgentVector population(model.Agent("agent"), AGENT_COUNT);
    // Initialise agents (TODO)
    {
        // Currently population has not been init, so generate an agent population on the fly
        std::mt19937_64 rng;
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
        cudaSimulation.setPopulationData(population);
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected
    cudaSimulation.getPopulationData(population);
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
// Initialise a population for the 0-bin simple persistnce testing
FLAMEGPU_INIT_FUNCTION(InitPopulationEvenOutputOnly) {
    // Initialise a population in an init function
    auto agent = FLAMEGPU->agent(AGENT_NAME);
    for (uint32_t i = 0; i < AGENT_COUNT; ++i) {
        auto instance = agent.newAgent();
        instance.setVariable<int>("id", i);
        instance.setVariable<unsigned int>("count", 0u);
        instance.setVariable<unsigned int>("sum", 0u);
    }
}
// Host function to forward a parent models step counter into a submodel via env var
FLAMEGPU_HOST_FUNCTION(ForwardParentStepCounter) {
    FLAMEGPU->environment.setProperty<unsigned int>("parentStepCounter", FLAMEGPU->getStepCounter());
}
// Fn condition to only run on even iteraitons
FLAMEGPU_AGENT_FUNCTION_CONDITION(ParentEvenOnlyCondition) {
    return FLAMEGPU->environment.getProperty<unsigned int>("parentStepCounter") % 2 == 0;
}
// Simple versionm of the output function, using just a single bin for simplicity
FLAMEGPU_AGENT_FUNCTION(out_simple, MessageNone, MessageBucket) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setKey(0);
    return ALIVE;
}
// Agent function which iterates read in mesasges and sums the ID, from a single bin.
FLAMEGPU_AGENT_FUNCTION(in_simple, MessageBucket, MessageNone) {
    const int id = FLAMEGPU->getVariable<int>("id");
    unsigned int count = 0;
    unsigned int sum = 0;
    for (auto &m : FLAMEGPU->message_in(0)) {
        count++;
        sum += m.getVariable<int>("id");
    }
    FLAMEGPU->setVariable<unsigned int>("count", count);
    FLAMEGPU->setVariable<unsigned int>("sum", sum);
    return ALIVE;
}
// Exit conditon that exits after the 2 steps have executed
FLAMEGPU_EXIT_CONDITION(ExitAfter2) {
    return FLAMEGPU->getStepCounter() >= 2 ? flamegpu::EXIT: flamegpu::CONTINUE;
}
// Step function to assert that the correct number of messsages have been read, ensuring that the PBM has been reset between calls to the same submodel.
FLAMEGPU_STEP_FUNCTION(AssertParentEvenOutputOnly) {
    HostAgentAPI agent = FLAMEGPU->agent(AGENT_NAME);
    // Get the population data
    DeviceAgentVector av = agent.getPopulationData();
    // Iterate the population, ensuring that each agent read the correct number of messages and got the correct sum of messages.
    // These values expect only a single bin is used, in the interest of simplicitly.
    const unsigned int exepctedCountEven = agent.count();
    const unsigned int expectedCountOdd = 0u;
    for (const auto& a : av) {
        if (FLAMEGPU->environment.getProperty<unsigned int>("parentStepCounter") % 2 == 0) {
            // Even iterations expect the count to match the number of agents, and sum to be non zero.
            ASSERT_EQ(a.getVariable<unsigned int>("count"), exepctedCountEven);
            ASSERT_NE(a.getVariable<unsigned int>("sum"), 0u);
        } else {
            // Odd iters expect 0 count and 0 sum
            ASSERT_EQ(a.getVariable<unsigned int>("count"), expectedCountOdd);
            ASSERT_EQ(a.getVariable<unsigned int>("sum"), 0u);
        }
    }
}
// Test that message list PBM is correcttly reset for subsequent steps of the outer model.
// This was a bug encountered during schelling model implemetnation.
TEST(BucketMessageTest, SubmodelPBMPersistence) {
    // Construct submodel
    ModelDescription submodel("submodel");
    {   // MessageBucket::Description
        MessageBucket::Description &message = submodel.newMessage<MessageBucket>("bucket");
        message.setBounds(0, AGENT_COUNT);
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription &agent = submodel.newAgent(AGENT_NAME);
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        auto &af = agent.newFunction("out", out_simple);
        af.setMessageOutput("bucket");
        af.setMessageOutputOptional(true);
        af.setFunctionCondition(ParentEvenOnlyCondition);
        agent.newFunction("in", in_simple).setMessageInput("bucket");
    }
    {   // Layer #1
        LayerDescription &layer = submodel.newLayer();
        layer.addAgentFunction(out_simple);
    }
    {   // Layer #2
        LayerDescription &layer = submodel.newLayer();
        layer.addAgentFunction(in_simple);
    }
    // Add an enviornment variable access the parent model iteration number in the submodel
    submodel.Environment().newProperty<unsigned int>("parentStepCounter", 0u);
    // Add a step function which validates the correct number of messages was read
    submodel.addStepFunction(AssertParentEvenOutputOnly);
    // Add the required exit condition, which exits after 2 iters of the submodel]
    submodel.addExitCondition(ExitAfter2);
    // Construct the parent model
    ModelDescription model("model");
    auto &smd = model.newSubModel("sub", submodel);
    {
        AgentDescription &agent = model.newAgent(AGENT_NAME);
        agent.newVariable<int>("id");
        agent.newVariable<unsigned int>("count", 0);  // Number of messages iterated
        agent.newVariable<unsigned int>("sum", 0);  // Sums of IDs in bucket
        smd.bindAgent(AGENT_NAME, AGENT_NAME, true, true);  // auto map vars and states
    }
    // Add an enviornment variable access the parent model iteration number in the submodel
    model.Environment().newProperty<unsigned int>("parentStepCounter", 0u);
    // Bind the env var to the sub env var
    smd.SubEnvironment().mapProperty("parentStepCounter", "parentStepCounter");
    // Add an init function to generate a population
    model.addInitFunction(InitPopulationEvenOutputOnly);
    // Adda  layer containing a host function which updatest the counter
    model.newLayer().addHostFunction(ForwardParentStepCounter);
    // Add the submodel to the outer model control flow
    model.newLayer().addSubModel("sub");
    // Construct the cuda simulation
    CUDASimulation cudaSimulation(model);
    // Run for 2 steps, to trigger an odd and an even step.
    cudaSimulation.SimulationConfig().steps = 2;
    EXPECT_NO_THROW(cudaSimulation.simulate());
}

TEST(BucketMessageTest, Mandatory_Range) {
    // Agent count must be multiple of 4
    std::unordered_map<int, unsigned int> bucket_count;
    std::unordered_map<int, unsigned int> bucket_sum;
    // Construct model
    ModelDescription model("BucketMessageTest");
    {   // MessageBucket::Description
        MessageBucket::Description &message = model.newMessage<MessageBucket>("bucket");
        message.setBounds(12, 12 +(AGENT_COUNT/2));  // None zero lowerBound, to check that's working
        message.newVariable<int>("id");
    }
    {   // AgentDescription
        AgentDescription agent = model.newAgent("agent");
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
    CUDASimulation cudaSimulation(model);

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
        cudaSimulation.setPopulationData(population);
    }

    // Execute a single step of the model
    cudaSimulation.step();

    // Recover the results and check they match what was expected
    cudaSimulation.getPopulationData(population);
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

FLAMEGPU_AGENT_FUNCTION(ArrayOut, MessageNone, MessageBucket) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, index * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, index * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, index * 11);
    FLAMEGPU->message_out.setKey(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn, MessageBucket, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in(my_index)) {
        FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
        FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
        FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
    }
    return ALIVE;
}
TEST(TestMessage_Bucket, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageBucket::Description &message = m.newMessage<MessageBucket>(MESSAGE_NAME);
    message.setBounds(0, AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}
const char* rtc_ArrayOut_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageBucket) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 0, index * 3);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 1, index * 7);
    FLAMEGPU->message_out.setVariable<unsigned int, 3>("v", 2, index * 11);
    FLAMEGPU->message_out.setKey(index);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageBucket, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in(my_index)) {
        FLAMEGPU->setVariable<unsigned int, 3>("message_read", 0, message.getVariable<unsigned int, 3>("v", 0));
        FLAMEGPU->setVariable<unsigned int, 3>("message_read", 1, message.getVariable<unsigned int, 3>("v", 1));
        FLAMEGPU->setVariable<unsigned int, 3>("message_read", 2, message.getVariable<unsigned int, 3>("v", 2));
    }
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_Bucket, ArrayVariable) {
    ModelDescription m(MODEL_NAME);
    MessageBucket::Description& message = m.newMessage<MessageBucket>(MESSAGE_NAME);
    message.setBounds(0, AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}

#if defined(USE_GLM)
FLAMEGPU_AGENT_FUNCTION(ArrayOut_glm, MessageNone, MessageBucket) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    glm::uvec3 t = glm::uvec3(index * 3, index * 7, index * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setKey(index);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(ArrayIn_glm, MessageBucket, MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in(my_index)) {
        FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
    }
    return ALIVE;
}
TEST(TestMessage_Bucket, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageBucket::Description &message = m.newMessage<MessageBucket>(MESSAGE_NAME);
    message.setBounds(0, AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", {UINT_MAX, UINT_MAX, UINT_MAX});
    AgentFunctionDescription fo = a.newFunction(OUT_FUNCTION_NAME, ArrayOut_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newFunction(IN_FUNCTION_NAME, ArrayIn_glm);
    fi.setMessageInput(message);
    LayerDescription &lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription &li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}
const char* rtc_ArrayOut_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayOut, flamegpu::MessageNone, flamegpu::MessageBucket) {
    const unsigned int index = FLAMEGPU->getVariable<unsigned int>("index");
    glm::uvec3 t = glm::uvec3(index * 3, index * 7, index * 11);
    FLAMEGPU->message_out.setVariable<glm::uvec3>("v", t);
    FLAMEGPU->message_out.setKey(index);
    return flamegpu::ALIVE;
}
)###";
const char* rtc_ArrayIn_func_glm = R"###(
FLAMEGPU_AGENT_FUNCTION(ArrayIn, flamegpu::MessageBucket, flamegpu::MessageNone) {
    const unsigned int my_index = FLAMEGPU->getVariable<unsigned int>("index");
    for (auto &message : FLAMEGPU->message_in(my_index)) {
        FLAMEGPU->setVariable<glm::uvec3>("message_read", message.getVariable<glm::uvec3>("v"));
    }
    return flamegpu::ALIVE;
}
)###";
TEST(TestRTCMessage_Bucket, ArrayVariable_glm) {
    ModelDescription m(MODEL_NAME);
    MessageBucket::Description& message = m.newMessage<MessageBucket>(MESSAGE_NAME);
    message.setBounds(0, AGENT_COUNT);
    message.newVariable<unsigned int, 3>("v");
    AgentDescription a = m.newAgent(AGENT_NAME);
    a.newVariable<unsigned int>("index");
    a.newVariable<unsigned int, 3>("message_read", { UINT_MAX, UINT_MAX, UINT_MAX });
    AgentFunctionDescription fo = a.newRTCFunction(OUT_FUNCTION_NAME, rtc_ArrayOut_func_glm);
    fo.setMessageOutput(message);
    AgentFunctionDescription fi = a.newRTCFunction(IN_FUNCTION_NAME, rtc_ArrayIn_func_glm);
    fi.setMessageInput(message);
    LayerDescription& lo = m.newLayer(OUT_LAYER_NAME);
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer(IN_LAYER_NAME);
    li.addAgentFunction(fi);
    // Assign the numbers in shuffled order to agents
    AgentVector pop(a, AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentVector::Agent ai = pop[i];
        ai.setVariable<unsigned int>("index", i);
    }
    // Set pop in model
    CUDASimulation c(m);
    c.setPopulationData(pop);
    c.step();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (AgentVector::Agent ai : pop) {
        const unsigned int index = ai.getVariable<unsigned int>("index");
        std::array<unsigned int, 3> v = ai.getVariable<unsigned int, 3>("message_read");
        ASSERT_EQ(v[0], index * 3);
        ASSERT_EQ(v[1], index * 7);
        ASSERT_EQ(v[2], index * 11);
    }
}
#else
TEST(TestMessage_Bucket, DISABLED_ArrayVariable_glm) { }
TEST(TestRTCMessage_Bucket, DISABLED_ArrayVariable_glm) { }
#endif

}  // namespace test_message_bucket
}  // namespace flamegpu
