import pytest
from unittest import TestCase
from pyflamegpu import *
import random as rand

AGENT_COUNT = 24

out_mandatory = """
FLAMEGPU_AGENT_FUNCTION(out_mandatory, flamegpu::MessageNone, flamegpu::MessageBucket) {
    int id = FLAMEGPU->getVariable<int>("id");
    FLAMEGPU->message_out.setVariable<int>("id", id);
    FLAMEGPU->message_out.setKey(12 + (id/2));
    return flamegpu::ALIVE;
}
"""
out_optional = """
FLAMEGPU_AGENT_FUNCTION(out_optional, flamegpu::MessageNone, flamegpu::MessageBucket) {
    if (FLAMEGPU->getVariable<int>("do_output")) {
        int id = FLAMEGPU->getVariable<int>("id");
        FLAMEGPU->message_out.setVariable<int>("id", id);
        FLAMEGPU->message_out.setKey(12 + (id/2));
    }
    return flamegpu::ALIVE;
}
"""
out_optionalNone = """
FLAMEGPU_AGENT_FUNCTION(out_optionalNone, flamegpu::MessageNone, flamegpu::MessageBucket) {
    return flamegpu::ALIVE;
}
"""
in_fn = """
FLAMEGPU_AGENT_FUNCTION(in, flamegpu::MessageBucket, flamegpu::MessageNone) {
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
    return flamegpu::ALIVE;
}
"""
in_range = """
FLAMEGPU_AGENT_FUNCTION(in_range, flamegpu::MessageBucket, flamegpu::MessageNone) {
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
    return flamegpu::ALIVE;
}
"""

class TestMessage_Bucket(TestCase):

    def test_DescriptionValidation(self): 
        m = pyflamegpu.ModelDescription("BucketMessageTest")
        # Test description accessors
        message = m.newMessageBucket("buckets")
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setUpperBound(0); # Min should default to 0, this would mean no buckets
        assert e.value.type() == "InvalidArgument"
        message.setLowerBound(10);
        message.setUpperBound(11);
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setUpperBound(0); # Max < Min
        assert e.value.type() == "InvalidArgument"
        message.setUpperBound(12);
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setLowerBound(13); # Min > Max
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setBounds(12, 12); # Min == Max
        assert e.value.type() == "InvalidArgument"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.setBounds(13, 12); # Min > Max
        assert e.value.type() == "InvalidArgument"
        message.setBounds(12, 13);
        message.newVariableInt("somevar")
        
    def test_DataValidation(self): 
        m = pyflamegpu.ModelDescription("BucketMessageTest")
        # Test Data copy constructor knows when bounds have not been init
        message = m.newMessageBucket("buckets")
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            cudaSimulation = pyflamegpu.CUDASimulation(m)  # Max not set
        assert e.value.type() == "InvalidMessage"
        message.setLowerBound(1);  # It should default to 0
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            cudaSimulation = pyflamegpu.CUDASimulation(m)  # Min not set
        assert e.value.type() == "InvalidMessage"
        message.setUpperBound(10);
        cudaSimulation = pyflamegpu.CUDASimulation(m)
        
    def test_reserved_name(self):
        m = pyflamegpu.ModelDescription("BucketMessageTest")
        message = m.newMessageBucket("buckets")
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            message.newVariableInt("_")
        assert e.value.type() == "ReservedName"
        
    def test_Mandatory(self):
        bucket_count = {};
        bucket_sum = {};
        # Construct model
        model = pyflamegpu.ModelDescription("BucketMessageTest");
        # MessageBucket::Description
        message = model.newMessageBucket("bucket")
        message.setBounds(12, int(12 +(AGENT_COUNT/2))); # None zero lowerBound, to check that's working
        message.newVariableInt("id")
        # AgentDescription
        agent = model.newAgent("agent");
        agent.newVariableInt("id");
        agent.newVariableUInt("count1", 0);  # Number of messages iterated
        agent.newVariableUInt("count2", 0);  # Size of bucket as returned by size()
        agent.newVariableUInt("sum", 0);  # Sums of IDs in bucket
        fo = agent.newRTCFunction("out", out_mandatory);
        fo.setMessageOutput(message);
        fi = agent.newRTCFunction("in", in_fn);
        fi.setMessageInput(message);
        # Layer #1
        lo = model.newLayer();
        lo.addAgentFunction(fo);     
        # Layer #2
        li = model.newLayer();
        li.addAgentFunction(fi);
       
        cudaSimulation = pyflamegpu.CUDASimulation(model)

        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        # Initialise agents
        # Currently population has not been init, so generate an agent population on the fly
        for i in range(AGENT_COUNT):
            instance = population[i];
            instance.setVariableInt("id", i);
            # Create it if it doesn't already exist
            if not int(i/2) in bucket_count:
                bucket_count[int(i/2)] = 0;
                bucket_sum[int(i/2)] = 0;
            bucket_count[int(i/2)] += 1;
            bucket_sum[int(i/2)] += i;
            
        cudaSimulation.setPopulationData(population);

        # Execute a single step of the model
        cudaSimulation.step();

        # Recover the results and check they match what was expected
        cudaSimulation.getPopulationData(population);
        # Validate each agent has correct result
        for ai in population:
            id = ai.getVariableInt("id");
            id_m1 = 0 if id == 0 else id-1;
            count1 = ai.getVariableUInt("count1");
            count2 = ai.getVariableUInt("count2");
            sum = ai.getVariableUInt("sum");
            assert count1 == bucket_count[int(id_m1/2)];
            assert count2 == bucket_count[int(id_m1/2)];
            assert sum == bucket_sum[int(id_m1/2)];
        
    def test_Optional(self):
        bucket_count = {};
        bucket_sum = {};
        # Construct model
        model = pyflamegpu.ModelDescription("BucketMessageTest");
        # MessageBucket::Description
        message = model.newMessageBucket("bucket")
        message.setBounds(12, int(12 +(AGENT_COUNT/2))); # None zero lowerBound, to check that's working
        message.newVariableInt("id")
        # AgentDescription
        agent = model.newAgent("agent");
        agent.newVariableInt("id");
        agent.newVariableInt("do_output");
        agent.newVariableUInt("count1", 0);  # Number of messages iterated
        agent.newVariableUInt("count2", 0);  # Size of bucket as returned by size()
        agent.newVariableUInt("sum", 0);  # Sums of IDs in bucket
        fo = agent.newRTCFunction("out", out_optional);
        fo.setMessageOutput(message);
        fo.setMessageOutputOptional(True);
        fi = agent.newRTCFunction("in", in_fn);
        fi.setMessageInput(message);
        # Layer #1
        lo = model.newLayer();
        lo.addAgentFunction(fo);     
        # Layer #2
        li = model.newLayer();
        li.addAgentFunction(fi);
        
        cudaSimulation = pyflamegpu.CUDASimulation(model)

        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        # Initialise agents
        # Currently population has not been init, so generate an agent population on the fly
        for i in range(AGENT_COUNT):
            do_out = 1 if rand.random() > 0.3 else 0;
            instance = population[i];
            instance.setVariableInt("id", i);
            instance.setVariableInt("do_output", do_out);
            # Create it if it doesn't already exist
            if not int(i/2) in bucket_count:
                bucket_count[int(i/2)] = 0;
                bucket_sum[int(i/2)] = 0;
            if do_out:
                bucket_count[int(i/2)] += 1;
                bucket_sum[int(i/2)] += i;
            
        cudaSimulation.setPopulationData(population);

        # Execute a single step of the model
        cudaSimulation.step();

        # Recover the results and check they match what was expected
        cudaSimulation.getPopulationData(population);
        # Validate each agent has correct result
        for ai in population:
            id = ai.getVariableInt("id");
            id_m1 = 0 if id == 0 else id-1;
            count1 = ai.getVariableUInt("count1");
            count2 = ai.getVariableUInt("count2");
            sum = ai.getVariableUInt("sum");
            assert count1 == bucket_count[int(id_m1/2)];
            assert count2 == bucket_count[int(id_m1/2)];
            assert sum == bucket_sum[int(id_m1/2)];
    
    def test_OptionalNone(self): 
        bucket_count = {};
        bucket_sum = {};
        # Construct model
        model = pyflamegpu.ModelDescription("BucketMessageTest");
        # MessageBucket::Description
        message = model.newMessageBucket("bucket")
        message.setBounds(12, int(12 +(AGENT_COUNT/2))); # None zero lowerBound, to check that's working
        message.newVariableInt("id")
        # AgentDescription
        agent = model.newAgent("agent");
        agent.newVariableInt("id");
        agent.newVariableUInt("count1", 0);  # Number of messages iterated
        agent.newVariableUInt("count2", 0);  # Size of bucket as returned by size()
        agent.newVariableUInt("sum", 0);  # Sums of IDs in bucket
        fo = agent.newRTCFunction("out", out_optionalNone);
        fo.setMessageOutput(message);
        fo.setMessageOutputOptional(True);
        fi = agent.newRTCFunction("in", in_fn);
        fi.setMessageInput(message);
        # Layer #1
        lo = model.newLayer();
        lo.addAgentFunction(fo);     
        # Layer #2
        li = model.newLayer();
        li.addAgentFunction(fi);
       
        cudaSimulation = pyflamegpu.CUDASimulation(model)

        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        # Initialise agents
        # Currently population has not been init, so generate an agent population on the fly
        for i in range(AGENT_COUNT):
            instance = population[i];
            instance.setVariableInt("id", i);
            # Create it if it doesn't already exist
            if not int(i/2) in bucket_count:
                bucket_count[int(i/2)] = 0;
                bucket_sum[int(i/2)] = 0;
            
        cudaSimulation.setPopulationData(population);

        # Execute a single step of the model
        cudaSimulation.step();

        # Recover the results and check they match what was expected
        cudaSimulation.getPopulationData(population);
        # Validate each agent has correct result
        for ai in population:
            id = ai.getVariableInt("id");
            id_m1 = 0 if id == 0 else id-1;
            count1 = ai.getVariableUInt("count1");
            count2 = ai.getVariableUInt("count2");
            sum = ai.getVariableUInt("sum");
            assert count1 == bucket_count[int(id_m1/2)];
            assert count2 == bucket_count[int(id_m1/2)];
            assert sum == bucket_sum[int(id_m1/2)];
            
    def test_Mandatory_Range(self): 
        bucket_count = {};
        bucket_sum = {};
        # Construct model
        model = pyflamegpu.ModelDescription("BucketMessageTest");
        # MessageBucket::Description
        message = model.newMessageBucket("bucket")
        message.setBounds(12, int(12 +(AGENT_COUNT/2))); # None zero lowerBound, to check that's working
        message.newVariableInt("id")
        # AgentDescription
        agent = model.newAgent("agent");
        agent.newVariableInt("id");
        agent.newVariableUInt("count1", 0);  # Number of messages iterated
        agent.newVariableUInt("count2", 0);  # Size of bucket as returned by size()
        agent.newVariableUInt("sum", 0);  # Sums of IDs in bucket
        fo = agent.newRTCFunction("out", out_mandatory);
        fo.setMessageOutput(message);
        fi = agent.newRTCFunction("in", in_range);
        fi.setMessageInput(message);
        # Layer #1
        lo = model.newLayer();
        lo.addAgentFunction(fo);     
        # Layer #2
        li = model.newLayer();
        li.addAgentFunction(fi);
       
        cudaSimulation = pyflamegpu.CUDASimulation(model)

        population = pyflamegpu.AgentVector(agent, AGENT_COUNT)
        # Initialise agents
        # Currently population has not been init, so generate an agent population on the fly
        for i in range(AGENT_COUNT):
            instance = population[i];
            instance.setVariableInt("id", i);
            # Create it if it doesn't already exist
            if not int(i/2) in bucket_count:
                bucket_count[int(i/2)] = 0;
                bucket_sum[int(i/2)] = 0;
            bucket_count[int(i/2)] += 1;
            bucket_sum[int(i/2)] += i;
            
        cudaSimulation.setPopulationData(population);

        # Execute a single step of the model
        cudaSimulation.step();

        # Recover the results and check they match what was expected
        cudaSimulation.getPopulationData(population);
        # Validate each agent has correct result
        for ai in population:
            id = ai.getVariableInt("id");
            id_m4 = int(int(id / 8) * 4);
            count1 = ai.getVariableUInt("count1");
            count2 = ai.getVariableUInt("count2");
            sum = ai.getVariableUInt("sum");
            _count1 = 0;
            _sum = 0;
            for j in range(4):
                _count1 += bucket_count[int(id_m4 + j)];
                _sum += bucket_sum[int(id_m4 + j)];
            assert count1 == _count1
            assert count2 == bucket_count[int(id/2)];
            assert sum == _sum