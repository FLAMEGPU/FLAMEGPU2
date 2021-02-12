import pytest
from unittest import TestCase
from pyflamegpu import *

class AgentInstanceTest(TestCase):

    def test_constructor(self): 
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableInt("int", 1);
        agent.newVariableUInt("uint", 2);
        ai = pyflamegpu.AgentInstance(agent);
        # New AgentInstance is default init
        assert ai.getVariableInt("int") == 1
        assert ai.getVariableUInt("uint") == 2
    
    def test_copy_constructor(self): 
      model = pyflamegpu.ModelDescription("model");
      agent = model.newAgent("agent");
      agent.newVariableInt("int", 1);
      agent.newVariableArrayUInt("uint3", 3, [2, 3, 4]);
      ai_uint3_ref = (0, 1, 2);
      # Copying an agent instance retains the values
      ai = pyflamegpu.AgentInstance(agent);
      ai.setVariableInt("int", 12);
      ai.setVariableArrayUInt("uint3", ai_uint3_ref);
      ai2 = pyflamegpu.AgentInstance(ai);
      assert ai2.getVariableInt("int") == 12
      ai2_uint3_check = ai2.getVariableArrayUInt("uint3");
      assert ai2_uint3_check == ai_uint3_ref
      # Copying an agent instance from an AgentVector::Agent retains values
      av = pyflamegpu.AgentVector(agent, 1);
      ava = av.front();
      ava.setVariableInt("int", 12);
      ava.setVariableArrayUInt("uint3", ai_uint3_ref);
      ai3 = pyflamegpu.AgentInstance(ava);
      assert ai3.getVariableInt("int") == 12
      ai2_uint3_check2 = ai3.getVariableArrayUInt("uint3");
      assert ai2_uint3_check2 == ai_uint3_ref
    
    # def test_move_constructor(self): Not applicable to python
    
    # def test_copy_assignment_operator(self):  Not applicable to python (assignment always works by ref in python)
    
    # def test_move_assignment_operator(self):  Not applicable to python
    
    def test_getsetVariable(self): 
        i = 15;  # This is a stripped down version of AgentVectorTest::AgentVector_Agent
        # Test correctness of AgentVector getVariableType
        model = pyflamegpu.ModelDescription("model");
        agent = model.newAgent("agent");
        agent.newVariableUInt("uint", 12);
        agent.newVariableArrayInt("int3", 3, [2, 3, 4] );
        agent.newVariableArrayInt("int2", 2, [5, 6] );
        agent.newVariableFloat("float", 15.0);

        # Create pop, variables are as expected
        ai = pyflamegpu.AgentInstance(agent);
        int3_ref = ( 2, 3, 4 );
        assert ai.getVariableUInt("uint") == 12
        int3_check = ai.getVariableArrayInt("int3");
        assert int3_check == int3_ref
        assert ai.getVariableInt("int2", 0) == 5
        assert ai.getVariableInt("int2", 1) == 6
        assert ai.getVariableFloat("float") == 15.0

        # Update value
        ai.setVariableUInt("uint", 12 + i);
        int3_set = ( 2 + i, 3 + i, 4 + i );
        ai.setVariableArrayInt("int3", int3_set);
        ai.setVariableInt("int2", 0, 5 + i);
        ai.setVariableInt("int2", 1, 6 + i);
        ai.setVariableFloat("float", 15.0 + i);
        

        # Check vars now match as expected
        assert ai.getVariableUInt("uint") == 12 + i
        int3_ref2 = ( 2 + i, 3 + i, 4 + i );
        int3_check = ai.getVariableArrayInt("int3");
        assert int3_check == int3_ref2
        assert ai.getVariableInt("int2", 0) == 5 + i
        assert ai.getVariableInt("int2", 1) == 6 + i
        assert ai.getVariableFloat("float") == 15.0 + i
        

        # Check various exceptions
        # setVariable(const std::string &variable_name, const T &value)
        # Bad name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableInt("wrong", 1)
        assert e.value.type() == "InvalidAgentVar"
        # Array passed to non-array method
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableInt("int2", 1)
        assert e.value.type() == "InvalidVarType"
        # Wrong type
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableInt("float", 1)
        assert e.value.type() == "InvalidVarType"
        
        # setVariableArray(const std::string &variable_name, const std::vector<T> &value)
        int3_ref2 = ( 2, 3, 4 );
        float3_ref = ( 2.0, 3.0, 4.0 );
        # Bad name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableArrayInt("wrong", int3_ref2);
        assert e.value.type() == "InvalidAgentVar"
        # Array passed to non-array method
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableArrayInt("int2", int3_ref2);
        assert e.value.type() == "InvalidVarType"
        # Wrong type
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableArrayFloat("int3", float3_ref);
        assert e.value.type() == "InvalidVarType"
        
        # setVariable(const std::string &variable_name, const unsigned int &array_index, const T &value)
        # Bad name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableInt("wrong", 0, 1);
        assert e.value.type() == "InvalidAgentVar"
        # Index out of bounds
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableInt("int2", 2, 1);
        assert e.value.type() == "OutOfBoundsException"
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableFloat("float", 1, 1);
        assert e.value.type() == "OutOfBoundsException"
        # Wrong type
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableInt("float", 0, 1);
        assert e.value.type() == "InvalidVarType"
        
        # getVariable(const std::string &variable_name) const
        # Bad name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.getVariableInt("wrong");
        assert e.value.type() == "InvalidAgentVar"
        # Array passed to non-array method
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.getVariableInt("int2");
        assert e.value.type() == "InvalidVarType"
        # Wrong type
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.getVariableInt("float");
        assert e.value.type() == "InvalidVarType"
        
        # getVariable(const std::string &variable_name) const
        # Bad name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.getVariableArrayInt("wrong");
        assert e.value.type() == "InvalidAgentVar"
        # Wrong type
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.getVariableArrayFloat("int3");
        assert e.value.type() == "InvalidVarType"
        
        # getVariable(const std::string &variable_name, const unsigned int &array_index) const
        # Bad name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.getVariableInt("wrong", 0);
        assert e.value.type() == "InvalidAgentVar"
        # Index out of bounds
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.getVariableInt("int2", 2);
        assert e.value.type() == "OutOfBoundsException"
        # Wrong type
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:
            ai.setVariableArrayFloat("int3", float3_ref);
        assert e.value.type() == "InvalidVarType"
        
    
