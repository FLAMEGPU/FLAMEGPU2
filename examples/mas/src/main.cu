/******************************************************************************
 * main_MAS.cu is a host function that prepares data array and passes it to the CUDA kernel.
 * This main_MAS.cu would either be specified by a user or automatically generated from the model.xml.
 * Each of the API functions will have a 121 mapping with XML elements
 * The API is very similar to FLAME 2. The directory structure and general project is set out very similarly.

 * Multi Agent model example

 ******************************************************************************
 * Author  Paul Richmond, Mozhgan Kabiri Chimeh
 * Date    Feb 2017
 *****************************************************************************/

#include <cstdio>
#include <cstdlib>
#include <iostream>

#include "flamegpu/flame_api.h"  // @todo rename flamegpu2.h or similar?

#define SIZE 10  // default value for the pop and msg size
#define enable_read 0

/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */


FLAMEGPU_AGENT_FUNCTION(output_func, MsgNone, MsgBruteForce) {
    const int s = FLAMEGPU->environment.get<int>("step");
    FLAMEGPU->setVariable<float>("x", s + 10.0f);
    FLAMEGPU->setVariable<float>("y", s + 11.0f);
    if ((blockDim.x * blockIdx.x + threadIdx.x) % 2 == s % 2) {
        // Optional output, only from even thread ids
        FLAMEGPU->message_out.setVariable<float>("x", s + 12.0f);
        printf("outp: %d\n", s + (blockDim.x * blockIdx.x + threadIdx.x));
        FLAMEGPU->message_out.setVariable<float>("y", s + (blockDim.x * blockIdx.x + threadIdx.x));
    }

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(input_func, MsgBruteForce, MsgNone) {
    // printf("Hello from input_func\n");
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");

    // printf("[get (x,y)]: x = %f, y = %f\n", x, y);

    FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");

    // printf("[set (x)]: x = %f, y = %f\n", x, y);

    // 0) not interested - need to remove.
    // float x1 = FLAMEGPU->getMessageVariable<float>("x");
    // float y1 = FLAMEGPU->getMessageVariable<float>("y");
    // printf("(input func - get msg): x = %f, y = %f\n", x1, y1);
    
    // Multiple options for iterating messages

    // 1) First method: iterator loop.
    // for (MessageList::iterator iterator = messageList.begin(); iterator != messageList.end(); ++iterator)
    for (auto iterator = FLAMEGPU->message_in.begin(); iterator != FLAMEGPU->message_in.end(); ++iterator) {
        // De-reference the iterator to get to the message object
        auto message = *iterator;

        float x0 = message.getVariable<float>("x");
        printf("(input func - for loop, get msg variable): x = %f\n", x0);
        float y0 = message.getVariable<float>("y");
        printf("(input func - for loop, get msg variable): y = %f\n", y0);
    }

    // 2) Second method: Range based for loop
    for (auto &message : FLAMEGPU->message_in) {
        float x0 = message.getVariable<float>("x");
        // printf("(input func - for-range, get msg variables): x = %f \n", x0);

        float y0 = message.getVariable<float>("y");
        // printf("(input func - for-range, get msg variables): y = %f \n", y0);
    }

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(add_func, MsgNone, MsgNone) {
    // printf("Hello from add_func\n");
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
    // printf("-y = %f, x = %f\n", y, x);
    FLAMEGPU->setVariable<float>("y", y + x);
    y = FLAMEGPU->getVariable<float>("y");
    // printf("-y after set = %f\n", y);
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(subtract_func, MsgNone, MsgNone) {
    // printf("Hello from subtract_func\n");
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
    // printf("y = %f, x = %f\n", y, x);
    FLAMEGPU->setVariable<float>("y", x - y);
    y = FLAMEGPU->getVariable<float>("y");
    // printf("y after set = %f\n", y);
    return ALIVE;
}
FLAMEGPU_HOST_FUNCTION(increment_step) {
    int s = FLAMEGPU->environment.get<int>("step");
    s++;
    FLAMEGPU->environment.set<int>("step", s);
}

int main(int argc, const char* argv[]) {
    /* Multi agent model */
    ModelDescription flame_model("circles_model");
    flame_model.Environment().add<int>("step", 0);

    AgentDescription &circle1_agent = flame_model.newAgent("circle1");
    circle1_agent.newVariable<float>("x");
    circle1_agent.newVariable<float>("y");

    AgentDescription &circle2_agent = flame_model.newAgent("circle2");
    circle2_agent.newVariable<float>("x");
    circle2_agent.newVariable<float>("y");

    // same name ?
    MessageDescription &location1_message = flame_model.newMessage("location1");
    location1_message.newVariable<float>("x");
    location1_message.newVariable<float>("y");
    /*
        MessageDescription &location2_message = flame_model.newMessage("location2");
        location2_message.newVariable<float>("x");
        location2_message.newVariable<float>("y");
    */

    AgentFunctionDescription &output_data = circle1_agent.newFunction("output_data", output_func);
    output_data.setMessageOutput(location1_message);
    output_data.setMessageOutputOptional(true);

    AgentFunctionDescription &input_data = circle2_agent.newFunction("input_data", input_func);
    input_data.setMessageInput(location1_message);

    AgentFunctionDescription &add_data = circle1_agent.newFunction("add_data", add_func);
    // add_data.setMessageOutput(location1_message);

    AgentFunctionDescription &subtract_data = circle2_agent.newFunction("subtract_data", subtract_func);
    // subtract_data.setMessageOutput(location1_message);

    LayerDescription &output_layer = flame_model.newLayer("output_layer");
    output_layer.addAgentFunction(output_data);

    LayerDescription &input_layer = flame_model.newLayer("input_layer");
    input_layer.addAgentFunction(input_data);

    // multiple functions per simulation layer (from different agents)
    LayerDescription &concurrent_layer = flame_model.newLayer("concurrent_layer");
    concurrent_layer.addAgentFunction(add_data);
    concurrent_layer.addAgentFunction(subtract_data);
    flame_model.addStepFunction(increment_step);
    CUDAAgentModel cuda_model(flame_model);

#ifdef enable_read
        AgentPopulation population1(circle1_agent);
        AgentPopulation population2(circle2_agent);

        cuda_model.setPopulationData(population1);
        cuda_model.setPopulationData(population2);

        // If there is not enough arguments bail.
        if (argc < 2) {
            fprintf(stderr, "Error not enough arguments.\n");
            return EXIT_FAILURE;
        }

        cuda_model.initialise(argc, argv);  // argv[1]
      // cuda_model.initialise(new StateReader(flame_model,argv[1]));
#else
        // 2)
        AgentPopulation population1(circle1_agent, SIZE);
        for (int i = 0; i < SIZE; i++) {
            AgentInstance instance = population1.getNextInstance("default");
            instance.setVariable<float>("x", i*0.1f);
            instance.setVariable<float>("y", i*0.1f);
        }

        AgentPopulation population2(circle2_agent, SIZE);
        for (int i = 0; i < SIZE; i++) {
            AgentInstance instance = population2.getNextInstance("default");
            instance.setVariable<float>("x", i*0.2f);
            instance.setVariable<float>("y", i*0.2f);
        }

        cuda_model.setPopulationData(population1);
        cuda_model.setPopulationData(population2);
#endif

    cuda_model.SimulationConfig().steps = 2;  // steps>1 --> does not work for now

    /* Run the model */
    cuda_model.simulate();

    cuda_model.getPopulationData(population1);
    cuda_model.getPopulationData(population2);

    cuda_model.output(argc, argv);  // argv[1]

    return 0;
}

