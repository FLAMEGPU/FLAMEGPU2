import pytest
from unittest import TestCase
from pyflamegpu import *

class ExitAlways(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
      return pyflamegpu.EXIT;

class SubAgentDescriptionTest(TestCase):

    def test_RequiresExitCondition(self):
        sm = pyflamegpu.ModelDescription("sub");
        # Define SubModel
        sm.newAgent("a");
        m = pyflamegpu.ModelDescription("host");
        ma = m.newAgent("b");
        # Define Model
        ma.newVariableFloat("b_float");
        ma.newVariableUInt("b_uint");
        ma.newState("b");
        # Missing exit condition        
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidSubModel exception
            m.newSubModel("sub", sm);
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        m.newSubModel("sub", sm);
        # Submodel name already exists
        sm2 = pyflamegpu.ModelDescription("sub2");
        # Define SubModel
        sm2.newAgent("a");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidSubModelName exception
            m.newSubModel("sub", sm2);

    def test_InvalidAgentName(self):
        sm = pyflamegpu.ModelDescription("sub");
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        # Define SubModel
        sm.newAgent("a");
        m = pyflamegpu.ModelDescription("host");
        m.newAgent("b");
        smd = m.newSubModel("sub", sm);
        # Invalid agent
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidSubAgentName exception
            smd.bindAgent("c", "b", False, False);
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentName exception
            smd.bindAgent("a", "c", False, False);
        # Good
        smd.bindAgent("a", "b", False, False)

    def test_InvalidAgentState(self):
        sm = pyflamegpu.ModelDescription("sub");
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        # Define SubModel
        a = sm.newAgent("a");
        a.newVariableFloat("a_float");
        a.newVariableUInt("a_uint");
        a.newState("a");
        a.newState("a2");
        m = pyflamegpu.ModelDescription("host");
        ma = m.newAgent("b");
        # Define Model
        ma.newVariableFloat("b_float");
        ma.newVariableUInt("b_uint");
        ma.newState("b");
        ma.newState("b2");
        smd = m.newSubModel("sub", sm);
        agent_map = smd.bindAgent("a", "b", False, False);
        # Invalid name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentState exception
            agent_map.mapState("c", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentState exception
            agent_map.mapState("a", "c");
        # Good
        agent_map.mapState("a", "b");
        # Already Bound
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentState exception
            agent_map.mapState("a2", "b");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentState exception
            agent_map.mapState("a", "b2");
        # Good
        agent_map.mapState("a2", "b2");

    def test_InvalidAgentVariable(self):
        sm = pyflamegpu.ModelDescription("sub");
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        # Define SubModel
        a = sm.newAgent("a");
        a.newVariableFloat("a_float");
        a.newVariableUInt("a_uint");
        a.newVariableUIntArray2("a_uint2");
        a.newVariableFloat("a_float2");
        a.newState("a");
        m = pyflamegpu.ModelDescription("host");
        ma = m.newAgent("b");
        # Define Model
        ma.newVariableFloat("b_float");
        ma.newVariableUInt("b_uint");
        ma.newVariableUIntArray2("b_uint2");
        ma.newVariableFloat("b_float2");
        ma.newState("b");
        smd = m.newSubModel("sub", sm);
        agent_map = smd.bindAgent("a", "b", False, False);
        # Bad name
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("c", "b_float");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("a_float", "c");
        # Bad data type
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("a_uint", "b_float");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("a_float", "a_uint");
        # Bad array length
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("a_uint", "b_uint2");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("a_uint2", "b_uint");
        # Good
        agent_map.mapVariable("a_float", "b_float");
        agent_map.mapVariable("a_uint", "b_uint");
        agent_map.mapVariable("a_uint2", "b_uint2");
        # Already bound
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("a_float2", "b_float");
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentVar exception
            agent_map.mapVariable("a_float", "b_float2");
        # Good
        agent_map.mapVariable("a_float2", "b_float2");

    def test_AlreadyBound(self):
        sm = pyflamegpu.ModelDescription("sub");
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        # Define SubModel
        sm.newAgent("a");
        sm.newAgent("a2");
        m = pyflamegpu.ModelDescription("host");
        m.newAgent("b");
        m.newAgent("b2");
        smd = m.newSubModel("sub", sm);
        # Good
        smd.bindAgent("a", "b", False, False);
        # Already Bound
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidAgentName exception
            smd.bindAgent("a2", "b", False, False);
        with pytest.raises(pyflamegpu.FGPURuntimeException) as e:  # InvalidSubAgentName exception
            smd.bindAgent("a", "b2", False, False);
        # Good
        smd.bindAgent("a2", "b2", False, False);
