import pytest
from unittest import TestCase
from pyflamegpu import *
import time

AGENT_COUNT = 10;
MODEL_NAME = "model";
SUBMODEL_NAME = "submodel";
AGENT_NAME = "agent";

class setGet(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    # Accessing DeviceAgentVector like this would previously lead to an access violation (Issue #522, PR #751)
    av = FLAMEGPU.agent(AGENT_NAME).getPopulationData();
    for ai in av:
        ai.setVariableInt("int", ai.getVariableInt("int") + 12);


class setGetHalf(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    av = agent.getPopulationData();
    for i in range(int(av.size()/4), av.size() - int(av.size()/4)):
        av[i].setVariableInt("int", av[i].getVariableInt("int") + 12);
    # agent.setPopulationData(av);

class GetIndex(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agents = FLAMEGPU.agent(AGENT_NAME);
    # Get DeviceAgentVector to the population
    agent_vector = agents.getPopulationData();
    # check all index values
    counter = 0;
    for a in agent_vector:
        assert a.getIndex() == counter;
        counter+=1;

class Insert(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    av = agent.getPopulationData();
    ai = pyflamegpu.AgentInstance(av[0]);
    av.insert(av.size() - int(AGENT_COUNT/2), AGENT_COUNT, ai);

class Erase(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    av = agent.getPopulationData();
    av.erase(int(AGENT_COUNT / 4), int(AGENT_COUNT / 2));
    av.push_back();
    av.back().setVariableInt("int", -2);

class AlwaysExit(pyflamegpu.HostCondition):
  def run(self,FLAMEGPU):
    return pyflamegpu.EXIT;
    
class HostAgentBirthAutoSync_step_at(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Test again (need to test each with it's own update)
    for i in range(4):
        assert av.at(AGENT_COUNT + i).getVariableInt("int") == -i;

    for i in range(AGENT_COUNT):
        assert av.at(i).getVariableInt("int") == i;

    
class HostAgentBirthAutoSync_step_front(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == 0;
    # Host agent birth 1 agent
    agent.newAgent().setVariableInt("int", -12);
    # Test again (need to test each with it's own update)
    assert av.front().getVariableInt("int") == -12;

    
class HostAgentBirthAutoSync_step_back(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Test again (need to test each with it's own update)
    assert av.back().getVariableInt("int") == -3;

    
class HostAgentBirthAutoSync_step_begin(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(AGENT_COUNT):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Test again (need to test each with it's own update)
    i = 0;
    for a in av:
        if i < AGENT_COUNT:
            assert a.getVariableInt("int") == i;
        else:
            assert a.getVariableInt("int") == -(i- AGENT_COUNT);

        i += 1;

    # Check we iterated the expected amount
    assert i == AGENT_COUNT * 2;
  
class HostAgentBirthAutoSync_step_empty(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == 0;
    assert av.empty() == True
    # Host agent birth 1 agent
    agent.newAgent().setVariableInt("int", -12);
    # Test again (need to test each with it's own update)
    assert av.empty() == False;

    
class HostAgentBirthAutoSync_step_size(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
      agent.newAgent();  # This creates the agent, we don't actually care about it's values at this point
    # Test again (need to test each with it's own update)
    assert av.size() == AGENT_COUNT + 4;

class HostAgentBirthAutoSync_step_resize_down(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Insert 4 agents at the end of the initial list
    av.resize(AGENT_COUNT - 4);
    # Check the size has changed correctly
    assert av.size() == AGENT_COUNT - 4;
    # Test again (need to test each with it's own update)
    for i in range(AGENT_COUNT - 4):
        assert av[i].getVariableInt("int") == i;


class AgentID_DeviceAgentVectorBirth(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agt_a = FLAMEGPU.agent("agent", "a");
    agt_b = FLAMEGPU.agent("agent", "b");
    vec_a = agt_a.getPopulationData();
    vec_b = agt_b.getPopulationData();
    birth_ct_a = vec_a.size();
    birth_ct_b = vec_b.size();
    vec_a.resize(birth_ct_a * 2);
    vec_b.resize(birth_ct_b * 2);
    for i in range(birth_ct_a, 2 * birth_ct_a):
        t = vec_a[i];
        t.setVariableID("id_copy", t.getID());

    for i in range(birth_ct_b,2 * birth_ct_b):
        t = vec_b[i];
        t.setVariableID("id_copy", t.getID());

    
class HostAgentBirthAutoSync_step_shrink_to_fit(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Add agents until we reach a state where av.capacity() > av.size()
    total_size = AGENT_COUNT;
    ct = 0;
    while (av.capacity() == av.size()):
        # Exit the test early if capacity always equals size.
        if ct >= 10:
            return;
        for i in range(4):
            agent.newAgent();  # This creates the agent, we don't actually care about it's values at this point
            total_size+=1;

        # Force auto resize
        assert av.size() == total_size;  # Don't need to test this here, but lint doesn't like us ignoring the value it returns
        ct += 1;

    assert av.capacity() >= av.size();
    # Add 1 more agent and shrink to fit
    agent.newAgent();  # This creates the agent, we don't actually care about it's values at this point
    total_size += 1;
    av.shrink_to_fit();
    # Check capacity now equals total_count
    assert av.capacity() >= total_size;
    assert av.size() >= total_size;
 
class HostAgentBirthAutoSync_step_clear(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        agent.newAgent();  # This creates the agent, we don't actually care about it's values at this point
    av.clear();
    # Test again after clear, to ensure it doesn't miss the host agent birth'd agents
    assert av.size() == 0;

class HostAgentBirthAutoSync_step_insert1(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Insert 4 agents at the end of the initial list
    av.insert(AGENT_COUNT, 4, av[1]);
    # Check the size has changed correctly
    assert av.size() == AGENT_COUNT  + 8;
    # Test again (need to test each with it's own update)
    for i in range(AGENT_COUNT + 8):
        a = av[i];
        if i < AGENT_COUNT:
            assert a.getVariableInt("int") == i;
        elif i < AGENT_COUNT + 4:
            assert a.getVariableInt("int") == 1;  # We inserted 4 copies of i
        else:
            assert a.getVariableInt("int")== -(i - (AGENT_COUNT + 4));  # Host new agents

class HostAgentBirthAutoSync_step_erase(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Remove 4 agents, 2 from end of initial list, 2 from start of new list
    av.erase(AGENT_COUNT - 2, AGENT_COUNT + 2);
    # Check the size has changed correctly
    assert av.size() == AGENT_COUNT;
    # Test again (need to test each with it's own update)
    for i in range(AGENT_COUNT):
        a = av[i];
        if i < AGENT_COUNT - 2:
            assert a.getVariableInt("int") == i;
        else:
            assert a.getVariableInt("int") == -(i + 4 - AGENT_COUNT);  # Host new agents



class HostAgentBirthAutoSync_step_push_back(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Insert 4 agents at the end of the initial list
    av.push_back();
    # Check the size has changed correctly
    assert av.size() == AGENT_COUNT + 5;
    # Test again (need to test each with it's own update)
    for i in range(AGENT_COUNT + 5):
        a = av[i];
        if i < AGENT_COUNT:
            assert a.getVariableInt("int") == i;
        elif i < AGENT_COUNT + 4:
            assert a.getVariableInt("int") == -(i - AGENT_COUNT);  # Host new agents
        else:
            assert a.getVariableInt("int") == -12;  # push_back



class HostAgentBirthAutoSync_step_pop_back(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Insert 4 agents at the end of the initial list
    av.pop_back();
    # Check the size has changed correctly
    assert av.size() == AGENT_COUNT + 3;
    # Test again (need to test each with it's own update)
    for i in range(AGENT_COUNT + 3):
        a = av[i];
        if i < AGENT_COUNT:
            assert a.getVariableInt("int") == i;
        else:
            assert a.getVariableInt("int") == -(i - AGENT_COUNT);  # Host new agents



class HostAgentBirthAutoSync_step_resize_up(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agent = FLAMEGPU.agent(AGENT_NAME);
    # It shouldn't matter whether this is called before/after
    av = agent.getPopulationData();
    # Agent begins with AGENT_COUNT agents, value 0- (AGENT_COUNT-1)
    assert av.size() == AGENT_COUNT;
    # Host agent birth 4 agents
    for i in range(4):
        a = agent.newAgent();
        a.setVariableInt("int", -i);

    # Insert 4 agents at the end of the initial list
    av.resize(AGENT_COUNT + 8);
    # Check the size has changed correctly
    assert av.size() == AGENT_COUNT + 8;
    # Test again (need to test each with it's own update)
    for i in range(AGENT_COUNT + 8):
        if i < AGENT_COUNT:
            assert av[i].getVariableInt("int") == i;
        elif i < AGENT_COUNT + 4:
            assert av[i].getVariableInt("int") == -(i - AGENT_COUNT);  # Host new agents
        else:
            assert av[i].getVariableInt("int") == -12;  # resize added agents should be default value


    
class AgentID_DeviceAgentVectorBirthMultiAgent(pyflamegpu.HostFunction):
  def run(self,FLAMEGPU):
    agt_a = FLAMEGPU.agent("agent");
    agt_b = FLAMEGPU.agent("agent2");
    vec_a = agt_a.getPopulationData();
    vec_b = agt_b.getPopulationData();
    birth_ct_a = vec_a.size();
    birth_ct_b = vec_b.size();
    vec_a.resize(birth_ct_a * 2);
    vec_b.resize(birth_ct_b * 2);
    for i in range(birth_ct_a, 2 * birth_ct_a):
        t = vec_a[i];
        t.setVariableID("id_copy", t.getID());

    for i in range(birth_ct_b, 2 * birth_ct_b):
        t = vec_b[i];
        t.setVariableID("id_copy", t.getID());

class DeviceAgentVectorTest(TestCase): 

    def test_setGet(self):
        # Initialise an agent population with values in a variable [0,1,2..N]
        # Inside a step function, retrieve the agent population as a DeviceAgentVector
        # Update all agents by adding 12 to their value
        # After model completion, retrieve the agent population and check their values are [12,13,14..N+12]
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 0);
        model.addStepFunction(setGet());

        # Init agent pop
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
          av[i].setVariableInt("int", i);

        # Create and step simulation
        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        for i in range(AGENT_COUNT):
          assert av[i].getVariableInt("int") == i + 12;

        # Step again
        sim.step();

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        for i in range(AGENT_COUNT):
          assert av[i].getVariableInt("int") == i + 24;
    
    def test_setGetHalf(self):
        # Initialise an agent population with values in a variable [0,1,2..N]
        # Inside a step function, retrieve the agent population as a DeviceAgentVector
        # Update half agents (contiguous block) by adding 12 to their value
        # After model completion, retrieve the agent population and check their values are [12,13,14..N+12]
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 0);
        model.addStepFunction(setGetHalf());

        # Init agent pop
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
          av[i].setVariableInt("int", i);

        # Create and step simulation
        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        assert av.size() == AGENT_COUNT;
        for i in range(AGENT_COUNT):
            if i < int(AGENT_COUNT/4) or i >= AGENT_COUNT - int(AGENT_COUNT/4):
                assert av[i].getVariableInt("int") == i
            else:
                assert av[i].getVariableInt("int") == i + 12;


        # Step again
        sim.step();

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        assert av.size() == AGENT_COUNT;
        for i in range(AGENT_COUNT):
            if i < int(AGENT_COUNT/4) or i >= AGENT_COUNT - int(AGENT_COUNT/4):
                assert av[i].getVariableInt("int") == i
            else:
                assert av[i].getVariableInt("int") == i + 24;

    def test_GetIndex(self):
        # Initialise an agent population with values in a variable [0,1,2..N]
        # Inside a step function, iterate the device agent vector
        # Assert that agent index matches the order in the vector.
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 0);
        model.addStepFunction(GetIndex());

        # Init agent pop
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        # Create and step simulation
        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();  # agent step function involved

    MasterIncrement = """
    FLAMEGPU_AGENT_FUNCTION(MasterIncrement, flamegpu::MessageNone, flamegpu::MessageNone) {
        FLAMEGPU->setVariable<unsigned int>("uint", FLAMEGPU->getVariable<unsigned int>("uint") + 1);
        return flamegpu::ALIVE;
    }
    """

    
    def test_SubmodelInsert(self):
        # In void CUDAFatAgentStateList::resize() (as of 2021-03-04)
        # The minimum buffer len is 1024 and resize grows by 25%
        # So to trigger resize, we can grow from 1024->2048

        # The intention of this test is to check that agent birth via DeviceAgentVector::insert works as expected
        # Specifically, that when the agent population is resized, unbound variabled in the master-model are default init
        sub_model = pyflamegpu.ModelDescription(SUBMODEL_NAME);
        sub_agent = sub_model.newAgent(AGENT_NAME);
        sub_agent.newVariableInt("int", 0);
        sub_model.addStepFunction(Insert());
        sub_model.addExitCondition(AlwaysExit());


        master_model = pyflamegpu.ModelDescription(MODEL_NAME);
        master_agent = master_model.newAgent(AGENT_NAME);
        master_agent.newVariableInt("int", 0);
        master_agent.newVariableUInt("uint", 12);
        mi_fn = master_agent.newRTCFunction("MasterIncrement", self.MasterIncrement);
        sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
        sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, True);
        master_model.newLayer().addAgentFunction(mi_fn);
        master_model.newLayer().addSubModel(sub_desc);

        # Init agent pop
        av = pyflamegpu.AgentVector(master_agent, AGENT_COUNT);
        vec_int = [];
        vec_uint = [];
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);
            av[i].setVariableUInt("uint", i);
            vec_int.append(i);
            vec_uint.append(i);


        # Create and step simulation
        sim = pyflamegpu.CUDASimulation(master_model);
        sim.setPopulationData(av);
        sim.step();
        # Update vectors to match
        for i in range(len(vec_uint)):
            vec_uint[i]+=1;
        vec_int_size = len(vec_int);
        vec_uint_size = len(vec_uint);
        for i in range(AGENT_COUNT):            
            vec_int.insert(vec_int_size - int(AGENT_COUNT / 2), vec_int[0]);
            vec_uint.insert(vec_uint_size - int(AGENT_COUNT / 2), 12);

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        assert av.size() == len(vec_int);
        for i in range(len(av)):
            assert av[i].getVariableInt("int") == vec_int[i];
            assert av[i].getVariableUInt("uint") == vec_uint[i];


        # Step again
        sim.step();
        # Update vectors to match
        for i in range(len(vec_uint)):
            vec_uint[i]+=1;
        vec_int_size = len(vec_int);
        vec_uint_size = len(vec_uint);
        for i in range(AGENT_COUNT):            
            vec_int.insert(vec_int_size - int(AGENT_COUNT / 2), vec_int[0]);
            vec_uint.insert(vec_uint_size - int(AGENT_COUNT / 2), 12);

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        assert av.size() == len(vec_int);
        for i in range(len(av)):
            assert av[i].getVariableInt("int") == vec_int[i];
            assert av[i].getVariableUInt("uint") == vec_uint[i];


    def test_SubmodelErase(self):
        # The intention of this test is to check that agent death via DeviceAgentVector::erase works as expected
        sub_model = pyflamegpu.ModelDescription(SUBMODEL_NAME);
        sub_agent = sub_model.newAgent(AGENT_NAME);
        sub_agent.newVariableInt("int", 0);
        sub_model.addStepFunction(Erase());
        sub_model.addExitCondition(AlwaysExit());


        master_model = pyflamegpu.ModelDescription(MODEL_NAME);
        master_agent = master_model.newAgent(AGENT_NAME);
        master_agent.newVariableInt("int", -1);
        master_agent.newVariableFloat("float", 12.0);
        sub_desc = master_model.newSubModel(SUBMODEL_NAME, sub_model);
        sub_desc.bindAgent(AGENT_NAME, AGENT_NAME, True);
        master_model.newLayer().addSubModel(sub_desc);

        # Init agent pop, and test vectors
        av = pyflamegpu.AgentVector(master_agent, AGENT_COUNT);
        vec_int = [];
        vec_flt = [];
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);
            vec_int.append(i);
            vec_flt.append(12.0);

        # Create and step simulation
        sim = pyflamegpu.CUDASimulation(master_model);
        sim.setPopulationData(av);
        sim.step();
        # Update vectors to match
        for i in range(int(AGENT_COUNT / 4), int(AGENT_COUNT / 2)):
            vec_int.pop(int(AGENT_COUNT / 4));
            vec_flt.pop(int(AGENT_COUNT / 4));
        vec_int.append(-2);
        vec_flt.append(12.0);

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        assert av.size() == len(vec_int);
        for i in range(len(vec_int)):
            assert av[i].getVariableInt("int") == vec_int[i];
            assert av[i].getVariableFloat("float") == vec_flt[i];


        # Step again
        sim.step();
        # Update vectors to match
        for i in range(int(AGENT_COUNT / 4), int(AGENT_COUNT / 2)):
            vec_int.pop(int(AGENT_COUNT / 4));
            vec_flt.pop(int(AGENT_COUNT / 4));
        vec_int.append(-2);
        vec_flt.append(12.0);

        # Retrieve and validate agents match
        sim.getPopulationData(av);
        assert av.size() == len(vec_int);
        for i in range(len(vec_int)):
            assert av[i].getVariableInt("int") == vec_int[i];
            assert av[i].getVariableFloat("float") == vec_flt[i];


    """
     The following tests all test the interaction between host agent birth and DeviceAgentVector
     All DeviceAgentVector methods are tested individually, to confirm they do apply host agent births before
     performing their actions.
    """
    def test_HostAgentBirthAutoSync_at(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_at());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT + 4;
    
    def test_HostAgentBirthAutoSync_front(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_front());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        sim = pyflamegpu.CUDASimulation(model);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == 1;

    def test_HostAgentBirthAutoSync_back(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_back());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT + 4;

    def test_HostAgentBirthAutoSync_begin(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_begin());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT + AGENT_COUNT;

    def test_HostAgentBirthAutoSync_empty(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_empty());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        sim = pyflamegpu.CUDASimulation(model);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == 1;

    def test_HostAgentBirthAutoSync_size(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_size());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();

    def test_HostAgentBirthAutoSync_step_shrink_to_fit(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_shrink_to_fit());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();

    def test_HostAgentBirthAutoSync_clear(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_clear());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();

    def test_HostAgentBirthAutoSync_insert1(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_insert1());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT + 8;

    def test_HostAgentBirthAutoSync_erase(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", 10);
        model.addStepFunction(HostAgentBirthAutoSync_step_erase());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT;

    def test_HostAgentBirthAutoSync_puck_back(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", -12);
        model.addStepFunction(HostAgentBirthAutoSync_step_push_back());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT + 5;

    def test_HostAgentBirthAutoSync_pop_back(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", -12);
        model.addStepFunction(HostAgentBirthAutoSync_step_pop_back());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT + 3;

    def test_HostAgentBirthAutoSync_resize_up(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", -12);
        model.addStepFunction(HostAgentBirthAutoSync_step_resize_up());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT + 8;

    def test_HostAgentBirthAutoSync_resize_down(self):
        model = pyflamegpu.ModelDescription(MODEL_NAME);
        agent = model.newAgent(AGENT_NAME);
        agent.newVariableInt("int", -12);
        model.addStepFunction(HostAgentBirthAutoSync_step_resize_down());
        av = pyflamegpu.AgentVector(agent, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            av[i].setVariableInt("int", i);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(av);
        sim.step();
        sim.getPopulationData(av);
        # Also confirm the agents weren't added twice
        assert av.size() == AGENT_COUNT - 4;

    def test_AgentID_MultipleStatesUniqueIDs(self):
        POP_SIZE = 100;
        # Create agents via AgentVector to two agent states
        # DeviceAgentVector Birth creates new agent in both states (at the end of the current agents)
        # Store agent IDs to an agent variable inside model
        # Export agents and check their IDs are unique
        # Also check that the id's copied during model match those at export

        model = pyflamegpu.ModelDescription("test_agentid");
        agent = model.newAgent("agent");
        agent.newVariableID("id_copy", pyflamegpu.ID_NOT_SET);
        agent.newState("a");
        agent.newState("b");

        layer_a = model.newLayer();
        layer_a.addHostFunction(AgentID_DeviceAgentVectorBirth());

        pop_in = pyflamegpu.AgentVector(agent, POP_SIZE);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop_in, "a");
        sim.setPopulationData(pop_in, "b");

        sim.step();

        pop_out_a = pyflamegpu.AgentVector(agent);
        pop_out_b = pyflamegpu.AgentVector(agent);

        sim.getPopulationData(pop_out_a, "a");
        sim.getPopulationData(pop_out_b, "b");

        ids = set();
        # Validate that there are no ID collisions
        for a in pop_out_a:
            ids.add(a.getID());
            if a.getVariableID("id_copy") != pyflamegpu.ID_NOT_SET:
                assert a.getID() == a.getVariableID("id_copy");  # ID is same as reported at birth
    

        for a in pop_out_b:
            ids.add(a.getID());
            if a.getVariableID("id_copy") != pyflamegpu.ID_NOT_SET:
                assert a.getID() == a.getVariableID("id_copy");  # ID is same as reported at birth
    

        assert len(ids) == 4 * POP_SIZE;  # No collisions

    def test_AgentID_MultipleAgents(self):
        POP_SIZE = 100;
        # Create agents via AgentVector to two agent types
        # DeviceAgentVector Birth creates new agent in both types
        # Store agent IDs to an agent variable inside model
        # Export agents and check their IDs are unique
        # Also check that the id's copied during model match those at export

        model = pyflamegpu.ModelDescription("test_agentid");
        agent = model.newAgent("agent");
        agent.newVariableID("id_copy", pyflamegpu.ID_NOT_SET);
        agent2 = model.newAgent("agent2");
        agent2.newVariableID("id_copy", pyflamegpu.ID_NOT_SET);

        layer_a = model.newLayer();
        layer_a.addHostFunction(AgentID_DeviceAgentVectorBirthMultiAgent());

        pop_in_a = pyflamegpu.AgentVector(agent, POP_SIZE);
        pop_in_b = pyflamegpu.AgentVector(agent2, POP_SIZE);

        sim = pyflamegpu.CUDASimulation(model);
        sim.setPopulationData(pop_in_a);
        sim.setPopulationData(pop_in_b);

        sim.step();

        pop_out_a = pyflamegpu.AgentVector(agent);
        pop_out_b = pyflamegpu.AgentVector(agent);

        sim.getPopulationData(pop_out_a);
        sim.getPopulationData(pop_out_b);

        ids_a = set();
        ids_b = set();
        # Validate that there are no ID collisions
        for a in pop_out_a:
            ids_a.add(a.getID());
            if a.getVariableID("id_copy") != pyflamegpu.ID_NOT_SET:
                assert a.getID() == a.getVariableID("id_copy");  # ID is same as reported at birth
    
        assert len(ids_a) == 2 * POP_SIZE;  # No collisions
        for a in pop_out_b:
            ids_b.add(a.getID());
            if a.getVariableID("id_copy") != pyflamegpu.ID_NOT_SET:
                assert a.getID() == a.getVariableID("id_copy");  # ID is same as reported at birth
    
        assert len(ids_b) == 2 * POP_SIZE;  # No collisions

 