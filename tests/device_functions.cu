
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

FLAMEGPU_AGENT_FUNCTION(output_func)
{
	float x = FLAMEGPU->getVariable<float>("x");
	FLAMEGPU->setVariable<float>("x", x + 2);
  x = FLAMEGPU->getVariable<float>("x");
  printf("x after set = %f\n", x);

	return ALIVE;
}

//FLAMEGPU_AGENT_FUNCTION(move_func)
//{
//	float x = FLAMEGPU->getVariable<float>("x");
//	FLAMEGPU->setVariable<float>("x", x + 2);
//	x = FLAMEGPU->getVariable<float>("x");
//
//	//??
//
//
//
//	return ALIVE;
//}

FLAMEGPU_AGENT_FUNCTION(input_func)
{
	return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(move_func)
{
	return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(stay_func)
{
	return ALIVE;
}

void attach_add_func(AgentFunctionDescription& func) {
	func.setFunction(&add_func);
}

void attach_subtract_func(AgentFunctionDescription& func) {
	func.setFunction(&subtract_func);
}

void attach_input_func(AgentFunctionDescription& func) {
	func.setFunction(&input_func);
}

void attach_move_func(AgentFunctionDescription& func) {
	func.setFunction(&move_func);
}

void attach_stay_func(AgentFunctionDescription& func) {
	func.setFunction(&stay_func);
}

void attach_output_func(AgentFunctionDescription& func) {
	func.setFunction(&output_func);
}


/**
 * test_actor_random.h
 */

FLAMEGPU_AGENT_FUNCTION(random1_func)
{
    FLAMEGPU->setVariable<float>("a", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("b", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("c", FLAMEGPU->random.uniform<float>());

    return ALIVE;
}
void attach_random1_func(AgentFunctionDescription& func)
{
    func.setFunction(&random1_func);
}
FLAMEGPU_AGENT_FUNCTION(random2_func)
{
    FLAMEGPU->setVariable<float>("uniform_float", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<double>("uniform_double", FLAMEGPU->random.uniform<double>());

    FLAMEGPU->setVariable<float>("normal_float", FLAMEGPU->random.normal<float>());
    FLAMEGPU->setVariable<double>("normal_double", FLAMEGPU->random.normal<double>());

    FLAMEGPU->setVariable<float>("logNormal_float", FLAMEGPU->random.logNormal<float>(0,1));
    FLAMEGPU->setVariable<double>("logNormal_double", FLAMEGPU->random.logNormal<double>(0, 1));

    // char
    FLAMEGPU->setVariable<char>("uniform_char", FLAMEGPU->random.uniform<char>(CHAR_MIN, CHAR_MAX));
    FLAMEGPU->setVariable<unsigned char>("uniform_u_char", FLAMEGPU->random.uniform<unsigned char>(0, UCHAR_MAX));
    // short
    FLAMEGPU->setVariable<short>("uniform_short", FLAMEGPU->random.uniform<short>(SHRT_MIN, SHRT_MAX));
    FLAMEGPU->setVariable<unsigned short>("uniform_u_short", FLAMEGPU->random.uniform<unsigned short>(0, USHRT_MAX));
    // int
    FLAMEGPU->setVariable<int>("uniform_int", FLAMEGPU->random.uniform<int>(INT_MIN, INT_MAX));
    FLAMEGPU->setVariable<unsigned int>("uniform_u_int", FLAMEGPU->random.uniform<unsigned int>(0, UINT_MAX));
    // long
    FLAMEGPU->setVariable<long>("uniform_long", FLAMEGPU->random.uniform<long>(LONG_MIN, LONG_MAX));
    FLAMEGPU->setVariable<unsigned long>("uniform_u_long", FLAMEGPU->random.uniform<unsigned long>(0, ULONG_MAX));
    // long long
    FLAMEGPU->setVariable<long long>("uniform_longlong", FLAMEGPU->random.uniform<long long>(LLONG_MIN, LLONG_MAX));
    FLAMEGPU->setVariable<unsigned long long>("uniform_u_longlong", FLAMEGPU->random.uniform<unsigned long long>(0, ULLONG_MAX));

    return ALIVE;
}
void attach_random2_func(AgentFunctionDescription& func)
{
    func.setFunction(&random1_func);
}