import pytest
from unittest import TestCase
from pyflamegpu import *

class AgentVectorTest(TestCase):

    def test_constructor(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector constructors, size(), array operator
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableInt("int", 1);
        agent.newVariableUInt("uint", 2);
        agent.newVariableFloat("float", 3.0);
        agent.newVariableDouble("double", 4.0);

        # Create empty vector
        empty_pop = pyflamegpu.AgentVector(agent);
        assert empty_pop.size() == 0

        # Create vector with 10 agents, all default init
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            instance = pop[i];
            assert instance.getVariableInt("int") == 1
            assert instance.getVariableUInt("uint") == 2
            assert instance.getVariableFloat("float") == 3.0
            assert instance.getVariableDouble("double") == 4.0

    # def test_move_constructor(self): Not applicable to python

    #def test_copy_assignment_operator(self): Not applicable to python  (assignment always works by ref in python)    
    """The test does pass though
        POP_SIZE = 10;
        # Test correctness of AgentVector copy assignment, size(), array operator
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableInt("int", 1);
        agent.newVariableUInt("uint", 2);
        agent.newVariableFloat("float", 3.0);
        agent.newVariableDouble("double", 4.0);

        # Create empty vector
        base_empty_pop = pyflamegpu.AgentVector(agent);
        pop = pyflamegpu.AgentVector(agent, 2);  # Just some junk pop to be overwritten
        pop = base_empty_pop;
        assert pop.size() == 0

        # Create vector with 10 agents, all default init
        base_pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        pop = base_pop;
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            instance = pop[i];
            assert instance.getVariableInt("int") == 1
            assert instance.getVariableUInt("uint") == 2
            assert instance.getVariableFloat("float") == 3
            assert instance.getVariableDouble("double") == 4.0
    """

    # def test_move_assignment_operator(self): Not applicable to python

    def test_at(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector at(), synonymous with array operator
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableInt("int", 1);
        agent.newVariableUInt("uint", 2);
        agent.newVariableFloat("float", 3.0);
        agent.newVariableDouble("double", 4.0);

        # Create vector with 10 agents, all default init
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            instance = pop.at(i);
            assert instance.getVariableInt("int") == 1
            assert instance.getVariableUInt("uint") == 2
            assert instance.getVariableFloat("float") == 3
            assert instance.getVariableDouble("double") == 4.0


        # Create vector with 10 agents, all default init
        const_pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert const_pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            instance = const_pop.at(i);
            assert instance.getVariableInt("int") == 1
            assert instance.getVariableUInt("uint") == 2
            assert instance.getVariableFloat("float") == 3
            assert instance.getVariableDouble("double") == 4.0


        # Out of bounds exception
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.at(POP_SIZE)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            const_pop.at(POP_SIZE)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.at(POP_SIZE + 10)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            const_pop.at(POP_SIZE + 10)
        assert e.value.type() == "OutOfBoundsException"

    def test_array_operator(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector array operator (operator[]()), synonymous with at()
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableInt("int", 1);
        agent.newVariableUInt("uint", 2);
        agent.newVariableFloat("float", 3.0);
        agent.newVariableDouble("double", 4.0);

        # Create vector with 10 agents, all default init
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            instance = pop[i];
            assert instance.getVariableInt("int") == 1
            assert instance.getVariableUInt("uint") == 2
            assert instance.getVariableFloat("float") == 3
            assert instance.getVariableDouble("double") == 4.0


        # Create vector with 10 agents, all default init
        const_pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert const_pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            instance = const_pop[i];
            assert instance.getVariableInt("int") == 1
            assert instance.getVariableUInt("uint") == 2
            assert instance.getVariableFloat("float") == 3
            assert instance.getVariableDouble("double") == 4.0


        # Out of bounds exception
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop[POP_SIZE]
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            const_pop[POP_SIZE]
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop[POP_SIZE + 10]
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            const_pop[POP_SIZE + 10]
        assert e.value.type() == "OutOfBoundsException"

    def test_front(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector front(), synonymous with at(0)
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableInt("int", 1);

        # Create vector with 10 agents, all default init
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        pop[0].setVariableInt("int", 12);

        assert pop.front().getVariableInt("int") == 12
        assert pop.front().getVariableInt("int") == pop[0].getVariableInt("int")
        assert pop[1].getVariableInt("int") == 1  # Non-0th element is different

        # Out of bounds exception
        empty_pop = pyflamegpu.AgentVector(agent);
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            empty_pop.front()
        assert e.value.type() == "OutOfBoundsException"

    def test_back(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector back(), synonymous with at(size()-1)
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableInt("int", 1);

        # Create vector with 10 agents, all default init
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        pop[POP_SIZE-1].setVariableInt("int", 12);

        assert pop.back().getVariableInt("int") == 12
        assert pop.back().getVariableInt("int") == pop[pop.size() - 1].getVariableInt("int")
        assert pop[pop .size()-2].getVariableInt("int") == 1  # Non-0th element is different

        # Out of bounds exception
        empty_pop = pyflamegpu.AgentVector(agent);
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            empty_pop.back()
        assert e.value.type() == "OutOfBoundsException"

    # def test_data(self): Not applicable to python
 
    def test_iterator(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector python style list iterator, and the member functions for creating them.
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");

        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);


        # Iterate vector
        i = 0;
        for instance in pop:
            assert instance.getVariableUInt("uint") == i
            i += 1

        assert i == pop.size()

        # Test empty is empty
        empty_pop = pyflamegpu.AgentVector(agent);
        i = 0;
        for instance in empty_pop:
            i += 1

        assert i == 0

    # def test_const_iterator(self): Not applicable to python 
    # def test_reverse_iterator(self): Not applicable to python 
    # def test_const_reverse_iterator(self): Not applicable to python 

    def test_empty(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector empty
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");

        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.empty() == False
        pop2 = pyflamegpu.AgentVector(agent);
        assert pop2.empty() == True

    def test_size(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector size
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");

        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        pop2 = pyflamegpu.AgentVector(agent);
        assert pop2.size() == 0

    # def test_size(self): Nothing to test
    def test_reserve(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector reserve
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");

        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);


        # Init size
        assert pop.size() == POP_SIZE
        assert pop.capacity() >= POP_SIZE

        # Reserving up works as expected
        RESERVE_SIZE = pop.capacity() * 10;
        pop.reserve(RESERVE_SIZE);
        assert pop.size() == POP_SIZE
        assert pop.capacity() >= RESERVE_SIZE

        # Reserving down does nothing
        pop.reserve(POP_SIZE);
        assert pop.size() == POP_SIZE
        assert pop.capacity() >= RESERVE_SIZE

        # Data remains initialised
        for i in range(POP_SIZE):
            assert pop[i].getVariableUInt("uint") == i

    # def test_capacity(self): reserve contains best testing of this
    
    def test_shrink_to_fit(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector shrink_to_fit
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");

        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);

        # Init size
        assert pop.size() == POP_SIZE
        assert pop.capacity() >= POP_SIZE

        # Grow the vector's capacity
        RESERVE_SIZE = pop.capacity() * 10;
        pop.reserve(RESERVE_SIZE);
        assert pop.size() == POP_SIZE
        assert pop.capacity() >= RESERVE_SIZE

        #  Shrink to fit
        pop.shrink_to_fit();
        assert pop.size() == POP_SIZE
        assert pop.capacity() >= POP_SIZE

        # Data remains initialised
        for i in range(POP_SIZE):
            assert pop[i].getVariableUInt("uint") == i

    def test_clear(self): 
        POP_SIZE = 10;
        DEFAULT_VALUE = 12;
        # Test correctness of AgentVector array operator (operator[]()), synonymous with at()
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", DEFAULT_VALUE);

        # Create vector with 10 agents, init to non-default value
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);

        assert pop.size() == POP_SIZE
        capacity = pop.capacity();
        # Clear resets size, but does not affect capacity
        pop.clear();
        assert pop.size() == 0
        assert pop.capacity() == capacity

        # If items are added back, they are default init
        pop.push_back();
        pop.push_back();
        pop.push_back();
        pop.resize(POP_SIZE);
        for i in range(POP_SIZE):
            assert pop[i].getVariableUInt("uint") == DEFAULT_VALUE

    def test_insert(self):
        POP_SIZE = 10;
        # Test correctness of AgentVector insert
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");
        # AgentVector to insert
        insert_av = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert insert_av.size() == POP_SIZE
        for i in range(POP_SIZE):
            insert_av[i].setVariableUInt("uint", 23 + i);
        insert_ava = insert_av.front();
        # AgentInstance to insert
        insert_ai = pyflamegpu.AgentInstance(agent);
        insert_ai.setVariableUInt("uint", 24);

        ## Insert single item
        # insert(size_type pos, const AgentInstance& value)
        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert len(pop) == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);
        # Use iterator to insert item
        pop.insert(4, insert_ai);
        assert len(pop) == POP_SIZE + 1
        for i in range(pop.size()):
            if (i < 4) :
                assert pop[i].getVariableUInt("uint") == i
            elif (i == 4) :
                assert pop[i].getVariableUInt("uint") == 24
            else :
                assert pop[i].getVariableUInt("uint") == i - 1
        
        # insert(size_type pos, const Agent& value)
        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert len(pop) == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);
        # Use iterator to insert item
        pop.insert(4, insert_ava);
        assert len(pop) == POP_SIZE + 1
        for i in range(pop.size()):
            if (i < 4) :
                assert pop[i].getVariableUInt("uint") == i
            elif (i == 4) :
                assert pop[i].getVariableUInt("uint") == 23
            else :
                assert pop[i].getVariableUInt("uint") == i - 1
        
        # Insert multiple copies
        # insert(size_type pos, size_type count, const AgentInstance& value);
        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert len(pop) == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);
        # Use iterator to insert 3 items
        pop.insert(4, 3, insert_ai);
        assert pop.size() == POP_SIZE + 3;
        for i in range(pop.size()):
            if i < 4 :
                assert pop[i].getVariableUInt("uint") == i
            elif i < 7 :
                assert pop[i].getVariableUInt("uint") == 24
            else :
                assert pop[i].getVariableUInt("uint") == i - 3
            
        # insert(size_type pos, size_type count, const Agent& value);
        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert len(pop) == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);
        # Use iterator to insert 3 items
        pop.insert(4, 3, insert_ava);
        assert pop.size() == POP_SIZE + 3
        for i in range(pop.size()):
            if i < 4 :
                assert pop[i].getVariableUInt("uint") == i
            elif i < 7 :
                assert pop[i].getVariableUInt("uint") == 23;
            else :
                assert pop[i].getVariableUInt("uint") == i - 3

        # Wrong agents for exceptions
        agent2 = model.newAgent("agent2");
        agent2.newVariableFloat("float");
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        # AgentVector to insert
        wrong_insert_av = pyflamegpu.AgentVector(agent2, POP_SIZE);
        wrong_insert_ava = wrong_insert_av.front();
        # AgentInstance to insert
        wrong_insert_ai = pyflamegpu.AgentInstance(agent2);

        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.insert(4, wrong_insert_ai)
        assert e.value.type() == "InvalidAgent";
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.insert(4, wrong_insert_ava)
        assert e.value.type() == "InvalidAgent";
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.insert(4, 3, wrong_insert_ai)
        assert e.value.type() == "InvalidAgent";
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.insert(4, 3, wrong_insert_ava)
        assert e.value.type() == "InvalidAgent";    

    def test_erase_single(self):
        POP_SIZE = 10;
        # Test correctness of AgentVector erase (on single items)
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", POP_SIZE + 2);

        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(pop.size()):
            pop[i].setVariableUInt("uint", i);

        # Use index to remove item
        pop.erase(4);
        # Check 4 has gone missing from vector
        assert pop.size() == POP_SIZE - 1
        for i in range(pop.size()):
            if i < 4:
                assert pop[i].getVariableUInt("uint") == i
            else:
                assert pop[i].getVariableUInt("uint") == i + 1
        # If we add back an item, it is default init
        pop.push_back();
        assert pop.back().getVariableUInt("uint") == POP_SIZE + 2

        # Test exceptions
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.erase(POP_SIZE)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.erase(POP_SIZE + 2)
        assert e.value.type() == "OutOfBoundsException"
        pop.erase(POP_SIZE-1); # No throw

    def test_erase_range(self):
        POP_SIZE = 10;
        # Test correctness of AgentVector erase (on single items)
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");

        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);
        
        # Use index to remove item
        pop.erase(4, 8);
        # Check 4,5,6,7 has gone missing from vector
        assert pop.size() == POP_SIZE - 4
        for i in range(POP_SIZE - 4):
            if i < 4 :
                assert pop[i].getVariableUInt("uint") == i
            else :
                assert pop[i].getVariableUInt("uint") == i + 4
        

        # Test exceptions
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.erase(POP_SIZE, POP_SIZE + 2)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.erase(int(POP_SIZE/2), POP_SIZE + 2)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.erase(POP_SIZE + 2, POP_SIZE + 4)
        assert e.value.type() == "OutOfBoundsException"
        pop.erase(0, POP_SIZE); # No throw

    def test_push_back(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector push_back, and whether created item is default init
        # The impact on erase/clear/pop_back/etc functions on default init, is tested by their respective tests
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", 2);
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        pop.back().setVariableUInt("uint", 12);
        pop.push_back();
        assert pop.size() == POP_SIZE + 1
        assert pop.back().getVariableUInt("uint") == 2
        # Test alt-push_back
        ai = pyflamegpu.AgentInstance(agent);
        ai.setVariableUInt("uint", 22);
        pop.push_back(ai);
        assert pop.size() == POP_SIZE + 2
        assert pop.back().getVariableUInt("uint") == 22
        # Diff agent fail
        agent2 = model.newAgent("agent2");
        agent2.newVariableFloat("float", 2.0);
        ai2 = pyflamegpu.AgentInstance(agent2);
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            pop.push_back(ai2)
        assert e.value.type() == "InvalidAgent"

    def test_pop_back(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector pop_back
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", 2);
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);

        assert pop.size() == POP_SIZE
        # Pop back removes the last item
        pop.pop_back();
        assert pop.size() == POP_SIZE-1
        assert pop.back().getVariableUInt("uint") == POP_SIZE - 2
        # Adding back a new item is default init
        pop.push_back();
        assert pop.size() == POP_SIZE
        assert pop.back().getVariableUInt("uint") == 2;

        # Test that pop_back on empty has no effect
        pop2 = pyflamegpu.AgentVector(agent);
        assert pop2.size() == 0
        pop2.pop_back(); # No throw
        assert pop2.size() == 0

    def test_resize(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector resize
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", 2);
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);

        assert pop.size() == POP_SIZE
        # Resizing to a lower size, removes the trailing elements
        SMALL_POP_SIZE = POP_SIZE - int(POP_SIZE / 3);
        pop.resize(SMALL_POP_SIZE);
        for i in range(SMALL_POP_SIZE):
            assert pop[i].getVariableUInt("uint") == i

        assert pop.size() == SMALL_POP_SIZE
        # Resizing to a bigger size, adds back default items
        BIG_POP_SIZE = POP_SIZE + 3;
        pop.resize(BIG_POP_SIZE);
        for i in range(BIG_POP_SIZE):
            if (i < SMALL_POP_SIZE):
                assert pop[i].getVariableUInt("uint") == i
            else:
                assert pop[i].getVariableUInt("uint") == 2

        assert pop.size() == BIG_POP_SIZE

    def test_swap(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector swap
        # This can be applied to agents of different types, but not testing that
        # Should work effectively the same
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", 2);
        pop_default = pyflamegpu.AgentVector(agent, 2 * POP_SIZE);
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == POP_SIZE
        for i in range(POP_SIZE):
            pop[i].setVariableUInt("uint", i);

        # Check pops are expected
        assert pop.size() == POP_SIZE
        assert pop_default.size() == 2 * POP_SIZE
        for i in range(POP_SIZE):
            assert pop[i].getVariableUInt("uint") == i
            assert pop_default[i].getVariableUInt("uint") == 2
            assert pop_default[POP_SIZE + i].getVariableUInt("uint") == 2

        # Swap and check again
        pop.swap(pop_default);
        assert pop_default.size() == POP_SIZE
        assert pop.size() == 2 * POP_SIZE
        for i in range(POP_SIZE):
            assert pop_default[i].getVariableUInt("uint") == i
            assert pop[i].getVariableUInt("uint") == 2
            assert pop[POP_SIZE + i].getVariableUInt("uint") == 2

    def test_equality_operator(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector operator==
        model = pyflamegpu.ModelDescription("model");
        agent1 = model.newAgent("agent");
        agent1.newVariableUInt("uint", 2);
        model2 = pyflamegpu.ModelDescription("model2");
        agent2 = model2.newAgent("agent");
        agent2.newVariableUInt("uint", 2);
        model3 = pyflamegpu.ModelDescription("model3");
        agent3 = model3.newAgent("agent");
        agent3.newVariableUInt("uint", 2);
        agent3.newVariableFloat("float", 3);
        model4 = pyflamegpu.ModelDescription("model4");
        agent4 = model4.newAgent("agent");
        agent4.newVariableInt("int", 2);
        model5 = pyflamegpu.ModelDescription("model5");
        agent5 = model5.newAgent("agent");
        agent5.newVariableInt("uint", 2);
        agent6 = model.newAgent("agent2");
        agent6.newVariableUInt("uint", 2);
        pop = pyflamegpu.AgentVector(agent1, POP_SIZE);
        # Copy of the list is equal
        pop2 = pyflamegpu.AgentVector(pop);
        assert pop == pop2
        assert pop2 == pop
        # Different, but identical agentdesc is equal
        pop3 = pyflamegpu.AgentVector(agent2, POP_SIZE);
        assert pop == pop3
        assert pop3 == pop
        # But not if the lengths differ
        pop4 = pyflamegpu.AgentVector(agent2, POP_SIZE + 1);
        assert not(pop == pop4)
        assert not(pop4 == pop)
        # Not if we have additional vars
        pop5 = pyflamegpu.AgentVector(agent3, POP_SIZE);
        assert not(pop == pop5)
        assert not(pop5 == pop)
        # Or var has diff type
        pop6 = pyflamegpu.AgentVector(agent4, POP_SIZE);
        assert not(pop == pop6)
        assert not(pop6 == pop)
        # Or diff type, same name
        pop7 = pyflamegpu.AgentVector(agent5, POP_SIZE);
        assert not(pop == pop7)
        assert not(pop7 == pop)
        # Different agent name
        pop8 = pyflamegpu.AgentVector(agent6, POP_SIZE);
        assert not(pop == pop8)
        assert not(pop8 == pop)
        # Or if the value of the variable differs
        pop2.front().setVariableUInt("uint", 12);
        assert not(pop == pop2)
        assert not(pop2 == pop)

    def test_inequality_operator(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector operator==
        model = pyflamegpu.ModelDescription("model");
        agent1 = model.newAgent("agent");
        agent1.newVariableUInt("uint", 2);
        model2 = pyflamegpu.ModelDescription("model2");
        agent2 = model2.newAgent("agent");
        agent2.newVariableUInt("uint", 2);
        model3 = pyflamegpu.ModelDescription("model3");
        agent3 = model3.newAgent("agent");
        agent3.newVariableUInt("uint", 2);
        agent3.newVariableFloat("float", 3);
        model4 = pyflamegpu.ModelDescription("model4");
        agent4 = model4.newAgent("agent");
        agent4.newVariableInt("int", 2);
        model5 = pyflamegpu.ModelDescription("model5");
        agent5 = model5.newAgent("agent");
        agent5.newVariableInt("uint", 2);
        agent6 = model.newAgent("agent2");
        agent6.newVariableUInt("uint", 2);
        pop = pyflamegpu.AgentVector(agent1, POP_SIZE);
        # Copy of the list is equal
        pop2 = pyflamegpu.AgentVector(pop);
        assert not(pop != pop2)
        assert not(pop2 != pop)
        # Different, but identical agentdesc is equal
        pop3 = pyflamegpu.AgentVector(agent2, POP_SIZE);
        assert not(pop != pop3)
        assert not(pop3 != pop)
        # But not if the lengths differ
        pop4 = pyflamegpu.AgentVector(agent2, POP_SIZE + 1);
        assert pop != pop4
        assert pop4 != pop
        # Not if we have additional vars
        pop5 = pyflamegpu.AgentVector(agent3, POP_SIZE);
        assert pop != pop5
        assert pop5 != pop
        # Or var has diff type
        pop6 = pyflamegpu.AgentVector(agent4, POP_SIZE);
        assert pop != pop6
        assert pop6 != pop
        # Or diff type, same name
        pop7 = pyflamegpu.AgentVector(agent5, POP_SIZE);
        assert pop != pop7
        assert pop7 != pop
        # Different agent name
        pop8 = pyflamegpu.AgentVector(agent6, POP_SIZE);
        assert pop != pop8
        assert pop8 != pop
        # Or if the value of the variable differs
        pop2.front().setVariableUInt("uint", 12);
        assert pop != pop2
        assert pop2 != pop

    def test_getAgentName(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector getAgentName
        model = pyflamegpu.ModelDescription("model");
        agent1 = model.newAgent("agent");
        agent1.newVariableUInt("uint", 2);
        agent2 = model.newAgent("testtest");
        agent2.newVariableUInt("uint", 2);
        pop = pyflamegpu.AgentVector(agent1, POP_SIZE);
        assert pop.getAgentName() == "agent"
        pop2 = pyflamegpu.AgentVector(agent2, POP_SIZE);
        assert pop2.getAgentName() == "testtest"

    def test_matchesAgentType(self): 
        POP_SIZE = 10;
        # Test correctness of AgentVector::matchesAgentType(AgentDescription)
        model = pyflamegpu.ModelDescription("model");
        agent1 = model.newAgent("agent");
        agent1.newVariableUInt("uint", 2);
        agent2 = model.newAgent("testtest");
        agent2.newVariableInt("int", 2);
        pop = pyflamegpu.AgentVector(agent1, POP_SIZE);
        pop1 = pyflamegpu.AgentVector(agent1);
        assert pop.matchesAgentType(agent1)
        assert not(pop.matchesAgentType(agent2))
        assert pop1.matchesAgentType(agent1)
        assert not(pop1.matchesAgentType(agent2))
        pop2 = pyflamegpu.AgentVector(agent2, POP_SIZE);
        assert not(pop2.matchesAgentType(agent1))
        assert pop2.matchesAgentType(agent2)

    # def test_getVariableType(self):  Not applicable to python
    # def test_getVariableMetaData(self):  can't test this
    
    def test_getInitialState(self): 
        # Test correctness of AgentVector getInitialState
        # Though this is moreso testing how iniital state works
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");
        agent2 = model.newAgent("agent2");
        agent2.newState("test");
        agent3 = model.newAgent("agent3");
        agent3.newState("test");
        agent3.newState("test2");
        agent4 = model.newAgent("agent4");
        agent4.newState("test");
        agent4.newState("test2");
        agent4.setInitialState("test2");
        pop = pyflamegpu.AgentVector(agent);
        pop2 = pyflamegpu.AgentVector(agent2);
        pop3 = pyflamegpu.AgentVector(agent3);
        pop4 = pyflamegpu.AgentVector(agent4);
        assert pop.getInitialState(), pyflamegpu.DEFAULT_STATE
        assert pop2.getInitialState(), "test"
        assert pop3.getInitialState(), "test"
        assert pop4.getInitialState(), "test2"

    def test_AgentVector_Agent(self): 
      # Test that AgentVector::Agent provides set/get access to an agent of an AgentVector
        POP_SIZE = 10;
        # Test correctness of AgentVector getVariableType
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", 12);
        agent.newVariableArrayInt("int3", 3, [2, 3, 4]);
        agent.newVariableArrayInt("int2", 2, [5, 6]);
        agent.newVariableFloat("float", 15.0);

        # Create pop, variables are as expected
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        int3_ref = ( 2, 3, 4 );
        for i in range(POP_SIZE):
            ai = pop[i];
            assert ai.getVariableUInt("uint") == 12
            int3_check = ai.getVariableArrayInt("int3");
            assert int3_check == int3_ref
            assert ai.getVariableInt("int2", 0) == 5
            assert ai.getVariableInt("int2", 1) == 6
            assert ai.getVariableFloat("float") == 15.0
            # check index value is as expected
            assert ai.getIndex() == i

        # Update values
        for i in range(POP_SIZE):
            ai = pop[i];
            ai.setVariableUInt("uint", 12 + i);
            int3_set = [ 2 + i, 3 + i, 4 + i ];
            ai.setVariableArrayInt("int3", int3_set);
            ai.setVariableInt("int2", 0, 5 + i)
            ai.setVariableInt("int2", 1, 6 + i)
            ai.setVariableFloat("float", 15.0 + i)


        # Check vars now match as expected
        for i in range(POP_SIZE):
            ai = pop[i];
            assert ai.getVariableUInt("uint") == 12 + i
            int3_ref2 = ( 2 + i, 3 + i, 4 + i );
            int3_check = ai.getVariableArrayInt("int3");
            assert int3_check == int3_ref2
            assert ai.getVariableInt("int2", 0) == 5 + i
            assert ai.getVariableInt("int2", 1) == 6 + i
            assert ai.getVariableFloat("float") == 15.0 + i


        # Check various exceptions
        ai = pop.front();
        # setVariable(const std::string &variable_name, const T &value)
        # Bad name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableInt("wrong", 1)
        assert e.value.type() == "InvalidAgentVar"
        # Array passed to non-array method
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableInt("int2", 1)
        assert e.value.type() == "InvalidVarType"
        # Wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableInt("float", 1)
        assert e.value.type() == "InvalidVarType"

        # setVariable(const std::string &variable_name, const std::array<T, N> &value)
        int3_ref2 = [ 2, 3, 4 ];
        float3_ref = [ 2.0, 3.0, 4.0 ];
        # Bad name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableArrayInt("wrong", int3_ref2);
        assert e.value.type() == "InvalidAgentVar"
        # Array passed to non-array method
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableArrayInt("int2", int3_ref2);
        assert e.value.type() == "InvalidVarType"
        # Wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableArrayFloat("int3", float3_ref);
        assert e.value.type() == "InvalidVarType"

        # setVariable(const std::string &variable_name, const unsigned int &array_index, const T &value)
        # Bad name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableInt("wrong", 0, 1)
        assert e.value.type() == "InvalidAgentVar"
        # Index out of bounds
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableInt("int2", 2, 1)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableFloat("float", 1, 1)
        assert e.value.type() == "OutOfBoundsException"
        # Wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.setVariableInt("float", 0, 1)
        assert e.value.type() == "InvalidVarType"

        # getVariable(const std::string &variable_name) const
        # Bad name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableInt("wrong")
        assert e.value.type() == "InvalidAgentVar"
        # Array passed to non-array method
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableInt("int2")
        assert e.value.type() == "InvalidVarType"
        # Wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableInt("float")
        assert e.value.type() == "InvalidVarType"

        # getVariableArray(const std::string &variable_name)
        # Bad name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableArrayInt("wrong")
        assert e.value.type() == "InvalidAgentVar"
        # Wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableArrayFloat("int3")
        assert e.value.type() == "InvalidVarType"

        # getVariable(const std::string &variable_name, const unsigned int &array_index)
        # Bad name
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableInt("wrong", 0)
        assert e.value.type() == "InvalidAgentVar"
        # Index out of bounds
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableInt("int2", 2)
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableFloat("float", 1)
        assert e.value.type() == "OutOfBoundsException"
        # Wrong type
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            ai.getVariableInt("float", 0)
        assert e.value.type() == "InvalidVarType"

    def test_py__len__(self):
        POP_SIZE = 10;
        # Test correctness of AgentVector __len__
        # This mostly checks that it matches size()
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");

        # Create vector with 10 agents
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        assert pop.size() == len(pop)
        pop.resize(POP_SIZE * 4)
        assert pop.size() == len(pop)
        pop.push_back()
        assert pop.size() == len(pop)
        
        pop = pyflamegpu.AgentVector(agent);
        assert pop.size() == len(pop)
    
    def test_py_negative_index(self):
        POP_SIZE = 10;
        # Test pythonic behaviour that you can negatively index into AgentVector to access it in reverse order
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint");
        
        # Create vector with 10 agents, init to their index
        pop = pyflamegpu.AgentVector(agent, POP_SIZE);
        pop.front().setVariableUInt("uint", 200);
        pop.back().setVariableUInt("uint", 100);
        
        assert pop[-len(pop)].getVariableUInt("uint") == pop.front().getVariableUInt("uint")
        assert pop[-1].getVariableUInt("uint")
      