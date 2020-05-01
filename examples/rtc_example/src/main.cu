/******************************************************************************
 * main.cu is a host function that prepares data array and passes it to the CUDA kernel.
 * This main.cu would either be specified by a user or automatically generated from the model.xml.
 * Each of the API functions will have a 121 mapping with XML elements
 * The API is very similar to FLAME 2. The directory structure and general project is set out very similarly.

 * Single Agent model example

 ******************************************************************************
 * Author  Paul Richmond, Mozhgan Kabiri Chimeh
 * Date    Feb 2017
 *****************************************************************************/

#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "flamegpu/flame_api.h"


/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */

#define AGENT_COUNT 32

const char* rtc_test_func_str = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_test_func, MsgNone, MsgNone) {
    printf("Hello from rtc_test_func\n");

    float x = FLAMEGPU->getVariable<float>("x");
    printf("x = %f\n", x);
    //FLAMEGPU->setVariable<float>("x", x + 2);
    //x = FLAMEGPU->getVariable<float>("x");
    // printf("x after set = %f\n", x);
    if (threadIdx.x %2 == 0)
        return ALIVE;
    else
        return DEAD;
}
)###";

const char* rtc_msg_out_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_msg_out_func, MsgNone, MsgBruteForce) {
    printf("x(in)=%d,", FLAMEGPU->getVariable<int>("x"));
    FLAMEGPU->message_out.setVariable("x", FLAMEGPU->getVariable<int>("x"));
    return ALIVE;
}
)###";

const char* rtc_msg_in_func = R"###(
FLAMEGPU_AGENT_FUNCTION(rtc_msg_in_func, MsgBruteForce, MsgNone) {
    int sum = 0;
    for (auto& message : FLAMEGPU->message_in) {
        const int x = message.getVariable<int>("x");
        printf("x(msg)=%d,", x);
        sum += x;
    }
    FLAMEGPU->setVariable<int>("sum", sum);
    return ALIVE;
}
)###";


int main(int argc, const char* argv[]) {
    ModelDescription m("RTC TEst Model");
    MsgBruteForce::Description& msg = m.newMessage("message_x");
    msg.newVariable<int>("x");
    AgentDescription& a = m.newAgent("agent");
    a.newVariable<int>("x");
    a.newVariable<int>("sum");
    a.newVariable<int>("product");
    AgentFunctionDescription& fo = a.newRTCFunction("rtc_msg_out_func", rtc_msg_out_func);
    fo.setMessageOutput(msg);
    AgentFunctionDescription& fi = a.newRTCFunction("rtc_msg_in_func", rtc_msg_in_func);
    fi.setMessageInput(msg);

    std::default_random_engine rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_int_distribution<int> dist(-3, 3);
    AgentPopulation pop(a, (unsigned int)AGENT_COUNT);
    int sum = 0;
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getNextInstance();
        const int x = dist(rng);
        sum += x;
        ai.setVariable<int>("x", x);
    }
    LayerDescription& lo = m.newLayer("output_layer");
    lo.addAgentFunction(fo);
    LayerDescription& li = m.newLayer("input_layer");
    li.addAgentFunction(fi);
    CUDAAgentModel c(m);
    c.initialise(argc, argv);
    c.SimulationConfig().steps = 1;
    c.setPopulationData(pop);
    c.simulate();
    c.getPopulationData(pop);
    // Validate each agent has same result
    for (unsigned int i = 0; i < AGENT_COUNT; ++i) {
        AgentInstance ai = pop.getInstanceAt(i);
        if (ai.getVariable<int>("sum") != sum)
            printf("Error: Agent sum (%d) not equal to host sum (%d)\n", ai.getVariable<int>("sum"), sum);
    }
    return 0;
}
