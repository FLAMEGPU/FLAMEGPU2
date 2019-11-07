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
  printf("x after set = %f\n", x);

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
