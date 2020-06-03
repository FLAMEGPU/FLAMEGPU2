# To run this test invoke the virtual env in ../../build/lib/<platform>/python/venv
# E.g. ../../build/lib/windows-x64/python/venv/scripts/activate
# TODO: This is not an actual testing framework yet

from pyflamegpu import *

rtc_empty_agent_func = """
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    return ALIVE;
}
"""

AGENT_COUNT = 64


# Test an empty agent function to ensure that the RTC library can successful build and run a minimal example

model = pyflamegpu.ModelDescription("model");
agent = model.newAgent("agent_name");
agent.newVariableFloat("x");

# messaging
message = model.newMessage("location");
message.newVariableInt("id");
message

# add RTC agent function
func = agent.newRTCFunction("rtc_test_func", rtc_empty_agent_func);
func.setAllowAgentDeath(True);
model.newLayer().addAgentFunction(func);

# Init pop
init_population = pyflamegpu.AgentPopulation(agent, AGENT_COUNT);
for i in range(AGENT_COUNT):
    instance = init_population.getNextInstance("default");
    instance.setVariableFloat("x", float(i));
# }
# // Setup Model
# CUDAAgentModel cuda_model(model);
# cuda_model.setPopulationData(init_population);
# // Run 1 step to ensure agent function compiles and runs
# cuda_model.step();
