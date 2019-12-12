#include "helpers/device_test_functions.h"

FLAMEGPU_AGENT_FUNCTION(add_func) {
    // should've returned error if the type was not correct. Needs type check
    double x = FLAMEGPU->getVariable<double>("x");

    FLAMEGPU->setVariable<double>("x", x + 2);

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(subtract_func) {
    double x = FLAMEGPU->getVariable<double>("x");
    double y = FLAMEGPU->getVariable<double>("y");

    FLAMEGPU->setVariable<double>("y", x - y);

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(output_func) {
    float x = FLAMEGPU->getVariable<float>("x");
    FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");

    return ALIVE;
}

// FLAMEGPU_AGENT_FUNCTION(move_func) {
//     float x = FLAMEGPU->getVariable<float>("x");
//    FLAMEGPU->setVariable<float>("x", x + 2);
//    x = FLAMEGPU->getVariable<float>("x");
//
//    // ??
//
//
//
//    return ALIVE;
// }

FLAMEGPU_AGENT_FUNCTION(input_func) {
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(move_func) {
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(stay_func) {
    return ALIVE;
}

AgentFunctionDescription& attach_add_func(AgentDescription& agent) {
    return agent.newFunction("func", add_func);
}

AgentFunctionDescription& attach_subtract_func(AgentDescription& agent) {
    return agent.newFunction("func", subtract_func);
}

AgentFunctionDescription& attach_input_func(AgentDescription& agent) {
    return agent.newFunction("func", input_func);
}

AgentFunctionDescription& attach_move_func(AgentDescription& agent) {
    return agent.newFunction("move", move_func);
}

AgentFunctionDescription& attach_stay_func(AgentDescription& agent) {
    return agent.newFunction("stay", stay_func);
}

AgentFunctionDescription& attach_output_func(AgentDescription& agent) {
    return agent.newFunction("output", output_func);
}


/**
 * test_actor_random.h
 */

FLAMEGPU_AGENT_FUNCTION(random1_func) {
    FLAMEGPU->setVariable<float>("a", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("b", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<float>("c", FLAMEGPU->random.uniform<float>());

    return ALIVE;
}
AgentFunctionDescription& attach_random1_func(AgentDescription& agent) {
    return agent.newFunction("func", random1_func);
}
FLAMEGPU_AGENT_FUNCTION(random2_func) {
    FLAMEGPU->setVariable<float>("uniform_float", FLAMEGPU->random.uniform<float>());
    FLAMEGPU->setVariable<double>("uniform_double", FLAMEGPU->random.uniform<double>());

    FLAMEGPU->setVariable<float>("normal_float", FLAMEGPU->random.normal<float>());
    FLAMEGPU->setVariable<double>("normal_double", FLAMEGPU->random.normal<double>());

    FLAMEGPU->setVariable<float>("logNormal_float", FLAMEGPU->random.logNormal<float>(0, 1));
    FLAMEGPU->setVariable<double>("logNormal_double", FLAMEGPU->random.logNormal<double>(0, 1));

    // char
    FLAMEGPU->setVariable<char>("uniform_char", FLAMEGPU->random.uniform<char>(CHAR_MIN, CHAR_MAX));
    FLAMEGPU->setVariable<unsigned char>("uniform_u_char", FLAMEGPU->random.uniform<unsigned char>(0, UCHAR_MAX));
    // short
    FLAMEGPU->setVariable<int16_t>("uniform_short", FLAMEGPU->random.uniform<int16_t>(INT16_MIN, INT16_MAX));
    FLAMEGPU->setVariable<uint16_t>("uniform_u_short", FLAMEGPU->random.uniform<uint16_t>(0, UINT16_MAX));
    // int
    FLAMEGPU->setVariable<int32_t>("uniform_int", FLAMEGPU->random.uniform<int32_t>(INT32_MIN, INT32_MAX));
    FLAMEGPU->setVariable<uint32_t>("uniform_u_int", FLAMEGPU->random.uniform<uint32_t>(0, UINT32_MAX));
    // long long
    FLAMEGPU->setVariable<int64_t>("uniform_longlong", FLAMEGPU->random.uniform<int64_t>(INT64_MIN, INT64_MAX));
    FLAMEGPU->setVariable<uint64_t>("uniform_u_longlong", FLAMEGPU->random.uniform<uint64_t>(0, UINT64_MAX));

    return ALIVE;
}
AgentFunctionDescription& attach_random2_func(AgentDescription& agent) {
    return agent.newFunction("func", random2_func);
}
