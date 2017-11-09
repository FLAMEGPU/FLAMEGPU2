
#include "device_functions.h"


FLAMEGPU_AGENT_FUNCTION(add_func)
{
	// should've returned error if the type was not correct. Needs type check
	double x = FLAMEGPU->getVariable<double>("x");

	FLAMEGPU->setVariable<double>("x", x + 2);

	return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(subtract_func)
{

	double x = FLAMEGPU->getVariable<double>("x");
	double y = FLAMEGPU->getVariable<double>("y");

	FLAMEGPU->setVariable<double>("y", x - y);

	return ALIVE;
}

void attach_add_func(AgentFunctionDescription& func) {
	func.setFunction(&add_func);
}

void attach_subtract_func(AgentFunctionDescription& func) {
	func.setFunction(&subtract_func);
}