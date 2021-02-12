import pytest
from unittest import TestCase
from pyflamegpu import *
from random import randint

AGENT_COUNT = 100;
SUB_MODEL_NAME = "SubModel";
PROXY_SUB_MODEL_NAME = "ProxySubModel";
MODEL_NAME = "Model";
AGENT_NAME = "Agent";
AGENT_VAR1_NAME = "AVar1";
AGENT_VAR2_NAME = "AVar2";
SUB_VAR1_NAME = "SubVar1";
AGENT_VAR_i = "i";
AGENT_VAR_t = "t";

MAPPED_STATE1 = "mapped1";
MAPPED_STATE2 = "mapped2";
UNMAPPED_STATE1 = "unmapped1";
UNMAPPED_STATE2 = "unmapped2";

AddT = """
FLAMEGPU_AGENT_FUNCTION(AddT, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    const unsigned int t = FLAMEGPU->getVariable<unsigned int>("t");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + t);
    FLAMEGPU->setVariable<unsigned int>("t", t + 1);
    return ALIVE;
}
"""
AddOne = """
FLAMEGPU_AGENT_FUNCTION(AddOne, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    const unsigned int sub_v = FLAMEGPU->getVariable<unsigned int>("SubVar1");
    if (sub_v == 12) {
        // sub_v should always be it's default value 12 if created in submodel, we never change it
        FLAMEGPU->setVariable<unsigned int>("AVar1", v + 1);
    } else if (sub_v == 599) {
        // sub_v Agents created byproxysubmodel or above will have this value, so original agents set this
        FLAMEGPU->setVariable<unsigned int>("AVar1", v + 1);
    } else {
        FLAMEGPU->setVariable<unsigned int>("AVar1", v + 100000);
    }
    return ALIVE;
}
"""
AddSubVar = """
FLAMEGPU_AGENT_FUNCTION(AddSubVar, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    const unsigned int sub_v = FLAMEGPU->getVariable<unsigned int>("SubVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + sub_v);
    FLAMEGPU->setVariable<unsigned int>("SubVar1", sub_v * 2);
    return ALIVE;
}
"""
AddOne2 = """
FLAMEGPU_AGENT_FUNCTION(AddOne2, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + 1);
    return ALIVE;
}
"""
AddTen = """
FLAMEGPU_AGENT_FUNCTION(AddTen, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    FLAMEGPU->setVariable<unsigned int>("AVar1", v + 10);
    const unsigned int v2 = FLAMEGPU->getVariable<unsigned int>("AVar2");
    FLAMEGPU->setVariable<unsigned int>("AVar2", v2 - 1000);
    return ALIVE;
}
"""
KillEven = """
FLAMEGPU_AGENT_FUNCTION(KillEven, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    FLAMEGPU->setVariable<unsigned int>("i", v * 3);
    if (FLAMEGPU->getVariable<unsigned int>("AVar2") > UINT_MAX-1000) {
        // First iteration
        if (v % 4 == 0)
            return DEAD;
    } else {
        // Second iteration
        if (v % 2 == 0)
            return DEAD;
    }
    return ALIVE;
}
"""
BirthEven = """
FLAMEGPU_AGENT_FUNCTION(BirthEven, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    FLAMEGPU->setVariable<unsigned int>("i", v * 3);
    if (FLAMEGPU->getVariable<unsigned int>("AVar2") > UINT_MAX - 2000) {
        // First iteration
        if (v % 4 == 0) {
            FLAMEGPU->agent_out.setVariable("i", v * 3);
            FLAMEGPU->agent_out.setVariable("AVar2", 4000 + v);
        }
    } else if (FLAMEGPU->getVariable<unsigned int>("AVar2") > UINT_MAX - 4000) {
        // Second iteration
        if ((v / 3) % 4 == 0) {
            FLAMEGPU->agent_out.setVariable("i", v * 3);
            FLAMEGPU->agent_out.setVariable("AVar2", 4000 + v);
        }
    }
    return ALIVE;
}
"""
AllowEven = """
FLAMEGPU_AGENT_FUNCTION_CONDITION(AllowEven) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    // First iteration
    if (v % 4 == 0) {
        return true;
    }
    return false;
}
"""
UpdateId100 = """
FLAMEGPU_AGENT_FUNCTION(UpdateId100, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("i");
    FLAMEGPU->setVariable<unsigned int>("i", v + 100);
    return ALIVE;
}
"""
class ExitAlways(pyflamegpu.HostFunctionConditionCallback):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
      return pyflamegpu.EXIT;

HostBirth = """
FLAMEGPU_HOST_FUNCTION(HostBirth) {
    auto a = FLAMEGPU->newAgent(AGENT_NAME);
    a.setVariable<unsigned int>(AGENT_VAR1_NAME, 5);
    a.setVariable<unsigned int>(AGENT_VAR2_NAME, 500);
}
"""
HostBirth2 = """
FLAMEGPU_HOST_FUNCTION(HostBirth2) {
    auto a = FLAMEGPU->newAgent(AGENT_NAME);
    a.setVariable<unsigned int>(AGENT_VAR1_NAME, 5);
}
"""
HostBirthUpdate = """
FLAMEGPU_AGENT_FUNCTION(HostBirthUpdate, MsgNone, MsgNone) {
    const unsigned int v = FLAMEGPU->getVariable<unsigned int>("AVar1");
    if (v == 5) {
        FLAMEGPU->setVariable<unsigned int>("AVar1", 500);
    }
    return ALIVE;
}
"""

def UINT_MAX():
  return pow(2, 32)-1;
  
class TestCUDASubAgent(TestCase):
    def test_simple(self):
        # Tests whether a sub model is capable of changing an agents variable
        # Agents in same named state, with matching variables
        sm = pyflamegpu.ModelDescription(SUB_MODEL_NAME);
        
        # Define SubModel      
        a = sm.newAgent(AGENT_NAME);
        a.newVariableUInt(AGENT_VAR1_NAME, 0);
        a.newVariableUInt(AGENT_VAR_t, 1);
        a.newVariableUInt(SUB_VAR1_NAME, 12);
        fn_1 = a.newRTCFunction("1", AddT);
        fn_2 = a.newRTCFunction("2", AddOne);
        sm.newLayer().addAgentFunction(fn_1);
        sm.newLayer().addAgentFunction(fn_2);
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        ma = m.newAgent(AGENT_NAME);
        # Define Model
        ma.newVariableUInt(AGENT_VAR1_NAME, 1);
        ma.newVariableUInt(AGENT_VAR2_NAME, UINT_MAX());
        fn_3 = ma.newRTCFunction("3", AddTen);
        smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, True, True);  # auto map vars and states
        m.newLayer().addAgentFunction(fn_3);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(fn_3);
        
        # Init Agents
        pop = pyflamegpu.AgentVector(ma, AGENT_COUNT);

        # Init Model
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 1;
        c.applyConfig();
        c.setPopulationData(pop);
        # Run Model
        c.step();
        # Check result
        # Mapped var = init + af + submodel af + af
        mapped_result = 1 + 10 + 1 + 1 + 10;
        # Unmapped var = init + af + af
        unmapped_result = UINT_MAX() - 1000 - 1000;
        c.getPopulationData(pop);
        for ai in pop:
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == mapped_result
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result

        # Run Model
        c.step();
        # Check result
        # Mapped var = mapped_result + af + submodel af + af
        mapped_result2 = mapped_result + 10 + 1 + 1 + 10;
        # Unmapped var = unmapped_result + af + af
        unmapped_result2 = unmapped_result - 1000 - 1000;
        c.getPopulationData(pop);
        for ai in pop:
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == mapped_result2
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result2
            
    def test_AgentDeath_BeforeSubModel(self):
        sm = pyflamegpu.ModelDescription(SUB_MODEL_NAME);
        # Define SubModel
        a = sm.newAgent(AGENT_NAME);
        a.newVariableUInt(AGENT_VAR1_NAME, 0);
        a.newVariableUInt(SUB_VAR1_NAME, 12);
        fn_1 = a.newRTCFunction("1", AddOne);
        sm.newLayer().addAgentFunction(fn_1);
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        ma = m.newAgent(AGENT_NAME);
        # Define Model
        ma.newVariableUInt(AGENT_VAR1_NAME, 1);
        ma.newVariableUInt(AGENT_VAR2_NAME, UINT_MAX());
        ma.newVariableUInt(AGENT_VAR_i);
        fn_2 = ma.newRTCFunction("2", KillEven);
        fn_2.setAllowAgentDeath(True);
        fn_3 = ma.newRTCFunction("3", AddTen);
        smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, True, True);  # auto map vars and states
        m.newLayer().addAgentFunction(fn_2);
        m.newLayer().addAgentFunction(fn_3);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(fn_3);

        # Init Agents
        pop = pyflamegpu.AgentVector(ma, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            ai = pop[i]
            ai.setVariableUInt(AGENT_VAR_i, i);
            ai.setVariableUInt(AGENT_VAR1_NAME, i);
            ai.setVariableUInt(AGENT_VAR2_NAME, UINT_MAX() - i);
            # Other vars all default init

        # Init Model
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 1;
        c.applyConfig();
        c.setPopulationData(pop);
        # Run Model
        c.step();
        # Check result
        # Mapped var = init + af + submodel af + af
        mapped_result = 10 + 1 + 10;
        # Unmapped var = init + af + af
        unmapped_result = UINT_MAX() - 1000 - 1000;
        c.getPopulationData(pop);
        assert len(pop) == int(AGENT_COUNT*0.75) # if AGENT_COUNT > 1000 this test will fail
        for ai in pop:
            _i = ai.getVariableUInt(AGENT_VAR_i);
            assert _i % 3 == 0  # Var divides cleanly by 3
            __i = int(_i/3);  # Calculate original value of AGENT_VAR_i
            assert __i % 4 != 0  # Agent doesn't have original AGENT_VAR_i that was supposed to be killed
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == __i + mapped_result;
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result - __i;

        # Run Model
        c.step();
        # Check result
        # Mapped var = mapped_result + af + submodel af + af
        mapped_result2 = mapped_result + 10 + 1 + 10;
        # Unmapped var = unmapped_result + af + af
        unmapped_result2 = unmapped_result - 1000 - 1000;
        c.getPopulationData(pop);
        assert len(pop) == int(AGENT_COUNT/2)
        for ai in pop:
            _i = ai.getVariableUInt(AGENT_VAR_i);
            assert _i % 9 == 0  # Var divides cleanly by 3
            __i = _i/9;  # Calculate original value of AGENT_VAR_i
            assert __i % 2 != 0  # Agent doesn't have original AGENT_VAR_i that was supposed to be killed
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == __i + mapped_result2;
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result2 - __i;
    
    def test_AgentDeath_InSubModel(self):
        sm = pyflamegpu.ModelDescription(SUB_MODEL_NAME);
        # Define SubModel
        a = sm.newAgent(AGENT_NAME);
        a.newVariableUInt(AGENT_VAR1_NAME, 0);
        a.newVariableUInt(AGENT_VAR2_NAME, 0);
        a.newVariableUInt(AGENT_VAR_i, 0);
        fn_1 = a.newRTCFunction("1", KillEven);
        fn_1.setAllowAgentDeath(True);
        sm.newLayer().addAgentFunction(fn_1);
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        ma = m.newAgent(AGENT_NAME);
        # Define Model
        ma.newVariableUInt(AGENT_VAR1_NAME, 1);
        ma.newVariableUInt(AGENT_VAR2_NAME, UINT_MAX());
        ma.newVariableUInt(AGENT_VAR_i);
        fn_2 = ma.newRTCFunction("2", AddOne2);
        fn_3 = ma.newRTCFunction("3", AddTen);
        smd = m.newSubModel("sub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, True, True);  # auto map vars and states
        m.newLayer().addAgentFunction(fn_2);
        m.newLayer().addSubModel("sub");
        m.newLayer().addAgentFunction(fn_3);
        
        # Init Agents
        pop = pyflamegpu.AgentVector(ma, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            ai = pop[i]
            ai.setVariableUInt(AGENT_VAR_i, i);
            ai.setVariableUInt(AGENT_VAR1_NAME, i);
            ai.setVariableUInt(AGENT_VAR2_NAME, UINT_MAX() - i);
            # Other vars all default init

        # Init Model
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 1;
        c.applyConfig();
        c.setPopulationData(pop);
        # Run Model
        c.step();
        # Check result
        # Mapped var = init + af + submodel af + af
        mapped_result = 1 + 10;
        # Unmapped var = init + af + af
        unmapped_result = UINT_MAX() - 1000;
        c.getPopulationData(pop);
        assert len(pop) == int(AGENT_COUNT*0.75) # if AGENT_COUNT > 1000 this test will fail
        for ai in pop:
            _i = ai.getVariableUInt(AGENT_VAR_i);
            assert _i % 3 == 0  # Var divides cleanly by 3
            __i = int(_i/3);  # Calculate original value of AGENT_VAR_i
            assert __i % 4 != 0  # Agent doesn't have original AGENT_VAR_i that was supposed to be killed
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == __i + mapped_result;
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result - __i;

        # Run Model
        c.step();
        # Check result
        # Mapped var = mapped_result + af + submodel af + af
        mapped_result2 = mapped_result + 1 + 10;
        # Unmapped var = unmapped_result + af + af
        unmapped_result2 = unmapped_result - 1000;
        c.getPopulationData(pop);
        assert len(pop) == int(AGENT_COUNT/2)
        for ai in pop:
            _i = ai.getVariableUInt(AGENT_VAR_i);
            assert _i % 9 == 0  # Var divides cleanly by 3
            __i = _i/9;  # Calculate original value of AGENT_VAR_i
            assert __i % 2 != 0  # Agent doesn't have original AGENT_VAR_i that was supposed to be killed
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == __i + mapped_result2;
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result2 - __i;
    
    def test_AgentDeath_InNestedSubModel(self):
        sm = pyflamegpu.ModelDescription(SUB_MODEL_NAME);
        # Define SubModel
        a = sm.newAgent(AGENT_NAME);
        a.newVariableUInt(AGENT_VAR1_NAME, 0);
        a.newVariableUInt(AGENT_VAR2_NAME, 0);
        a.newVariableUInt(AGENT_VAR_i, 0);
        fn_1 = a.newRTCFunction("1", KillEven);
        fn_1.setAllowAgentDeath(True);
        sm.newLayer().addAgentFunction(fn_1);
        exitcdn = ExitAlways()
        sm.addExitConditionCallback(exitcdn);
        
        # Define Proxy SubModel
        psm = pyflamegpu.ModelDescription(SUB_MODEL_NAME);
        pa = psm.newAgent(AGENT_NAME);
        pa.newVariableUInt(AGENT_VAR1_NAME, 0);
        pa.newVariableUInt(AGENT_VAR2_NAME, 0);
        pa.newVariableUInt(AGENT_VAR_i, 0);
        psmd = psm.newSubModel("sub", sm);
        psmd.bindAgent(AGENT_NAME, AGENT_NAME, True, True);  # auto map vars and states
        psm.newLayer().addSubModel("sub");
        psm.addExitConditionCallback(exitcdn);
        
        m = pyflamegpu.ModelDescription(MODEL_NAME);
        ma = m.newAgent(AGENT_NAME);
        # Define Model
        ma.newVariableUInt(AGENT_VAR1_NAME, 1);
        ma.newVariableUInt(AGENT_VAR2_NAME, UINT_MAX());
        ma.newVariableUInt(AGENT_VAR_i);
        fn_2 = ma.newRTCFunction("2", AddOne2);
        fn_3 = ma.newRTCFunction("3", AddTen);
        smd = m.newSubModel("proxysub", sm);
        smd.bindAgent(AGENT_NAME, AGENT_NAME, True, True);  # auto map vars and states
        m.newLayer().addAgentFunction(fn_2);
        m.newLayer().addSubModel("proxysub");
        m.newLayer().addAgentFunction(fn_3);

        # Init Agents
        pop = pyflamegpu.AgentVector(ma, AGENT_COUNT);
        for i in range(AGENT_COUNT):
            ai = pop[i];
            ai.setVariableUInt(AGENT_VAR_i, i);
            ai.setVariableUInt(AGENT_VAR1_NAME, i);
            ai.setVariableUInt(AGENT_VAR2_NAME, UINT_MAX() - i);
            # Other vars all default init

        # Init Model
        c = pyflamegpu.CUDASimulation(m)
        c.SimulationConfig().steps = 1;
        c.applyConfig();
        c.setPopulationData(pop);
        # Run Model
        c.step();
        # Check result
        # Mapped var = init + af + submodel af + af
        mapped_result = 1 + 10;
        # Unmapped var = init + af + af
        unmapped_result = UINT_MAX() - 1000;
        c.getPopulationData(pop);
        assert len(pop) == int(AGENT_COUNT*0.75) # if AGENT_COUNT > 1000 this test will fail
        for ai in pop:
            _i = ai.getVariableUInt(AGENT_VAR_i);
            assert _i % 3 == 0  # Var divides cleanly by 3
            __i = int(_i/3);  # Calculate original value of AGENT_VAR_i
            assert __i % 4 != 0  # Agent doesn't have original AGENT_VAR_i that was supposed to be killed
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == __i + mapped_result;
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result - __i;

        # Run Model
        c.step();
        # Check result
        # Mapped var = mapped_result + af + submodel af + af
        mapped_result2 = mapped_result + 1 + 10;
        # Unmapped var = unmapped_result + af + af
        unmapped_result2 = unmapped_result - 1000;
        c.getPopulationData(pop);
        assert len(pop) == int(AGENT_COUNT/2)
        for ai in pop:
            _i = ai.getVariableUInt(AGENT_VAR_i);
            assert _i % 9 == 0  # Var divides cleanly by 3
            __i = _i/9;  # Calculate original value of AGENT_VAR_i
            assert __i % 2 != 0  # Agent doesn't have original AGENT_VAR_i that was supposed to be killed
            assert ai.getVariableUInt(AGENT_VAR1_NAME) == __i + mapped_result2;
            assert ai.getVariableUInt(AGENT_VAR2_NAME) == unmapped_result2 - __i;

