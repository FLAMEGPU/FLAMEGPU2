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

/*
FLAMEGPU_AGENT_FUNCTION(output_func, MsgNone, MsgNone) {
    // printf("Hello from output_func\n");

    float x = FLAMEGPU->getVariable<float>("x");
    // printf("x = %f\n", x);
    FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");
    // printf("x after set = %f\n", x);

    return ALIVE;
}
*/
FLAMEGPU_AGENT_FUNCTION(output_func, MsgNone, MsgNone) {
// some thoughts on how to use messages
    // FLAMEGPU->outputMessage<float>("location","x");
    // FLAMEGPU->outputMessage<float>("location","x",FLAMEGPU->getVariable<float>("x"));
    // FLAMEGPU->outputMessage<float>("location","x",FLAMEGPU->getVariable<float>("x"),"y",FLAMEGPU->getVariable<float>("y"));
    // FLAMEGPU->outputMessage("location");

    float x = FLAMEGPU->getVariable<float>("x");
    // printf("x = %f\n", x);
    FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(input_func, MsgNone, MsgNone) {
    float x = FLAMEGPU->getVariable<float>("x");
    // printf("x = %f\n", x);
    FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(move_func, MsgNone, MsgNone) {
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(stay_func, MsgNone, MsgNone) {
    return ALIVE;
}


int main(void) {
    /* MODEL */
    /* The model is the description of the actual model that is equivalent to that described by model.xml*/
    /* Nothing with GPUs it is only about building the model description in memory */
    ModelDescription flame_model("circles_model");

    // circle agent
    AgentDescription &circle_agent = flame_model.newAgent("circle");
    circle_agent.newVariable<float>("x");
    circle_agent.newVariable<float>("y");
    circle_agent.newVariable<float>("dx");
    circle_agent.newVariable<float>("dy");


    // circle add states
    // circle_agent.addState("state1");
    // circle_agent.addState("state2");


    // location message
    MessageDescription &location_message = flame_model.newMessage("location");
    location_message.newVariable<float>("x");
    location_message.newVariable<float>("y");

    // circle agent output_data function
    // Do not specify the state. As their are no states in the system it is assumed that this model is stateless.
    AgentFunctionDescription &output_data = circle_agent.newFunction("output_data", output_func);
    output_data.setMessageOutput(location_message);
    // output_data.setInitialState("state1");

    // circle agent input_data function
    AgentFunctionDescription &input_data = circle_agent.newFunction("input_data", input_func);
    input_data.setMessageInput(location_message);


    // circle agent move function
    AgentFunctionDescription &move = circle_agent.newFunction("move", move_func);

    // TODO: At some point the model should be validated and then become read only. You should not be bale to add new agent variables once you have instances of the population for example.
    // flame_model.validate();


    // TODO: globals

    // POPULATION (FLAME2 mem)
    /* Population is an instantiation of the model. It is equivalent to the data from 0.xml or any other state of the model. It requires a model description to know what the agent variables and states are. */
    /* Data in populations and instances are only on the host. No concept of GPUs at this stage. */

    AgentPopulation population(circle_agent, 10);
    // TODO: Set maximum population size if known in advance
    for (int i=0; i < 10; i++) {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
        instance.setVariable<float>("y", i*0.1f);
        instance.setVariable<float>("dx", 0);
        instance.setVariable<float>("dy", 0);

        // get function would look like
        // int x = instance.getVariable<int>("x");
    }

    /* GLOBALS */
    /* TODO: We will consider this later. Not in the circles model. */


    // SIMULATION
    /* This is essentially the function layers from a model.xml */
    /* Currently this is specified by the user. In the future this could be generated automatically through dependency analysis like with FLAME HPC */

    LayerDescription &output_layer = flame_model.newLayer("output_layer");  // in the original schema layers are not named
    output_layer.addAgentFunction(output_data);

    LayerDescription &input_layer = flame_model.newLayer("input_layer");
    input_layer.addAgentFunction(input_data);

    LayerDescription &move_layer = flame_model.newLayer("input_layer");
    move_layer.addAgentFunction(move);

    // TODO: simulation.getLayerPosition("layer name") - returns an int
    // Simulation.addFunctionToLayer(0,"lmove")
    // This would come from the program arguments. Would be argv[2] if this file had been generated from model.xml


    /* CUDA agent model */
    /* Instantiate the model with a set of data (agents) on the device */
    /* Run the model */
    CUDAAgentModel cuda_model(flame_model);

    cuda_model.SimulationConfig().steps = 1;

    cuda_model.setPopulationData(population);

    cuda_model.simulate();

    // cuda_model.simulate(simulation);

    cuda_model.step();

    cuda_model.getPopulationData(population);

    /* This is not to be done yet. We want to first replicate the functionality of FLAMEGPU on a single device */
    /*
    // EXECUTION
    HardwareDescription hardware_config();
    hardware_config.addGPUResource(SM_30);
    hardware_config.addGPUResource(SM_20);

    // SCHEDULER and MAPPING
    // dynamic scheduling is not possible without considering the implications of communication later in the iteration (dynamic only suitable for shared memory systems unless lookahead is used)
    // mapping for now should be simple but in future a generic is based on either
    // 1) memory constraints
    // 2) occupancy
    // scheduler should use occupancy calculator to determine best thread block size
    GPUScheduler scheduler(&hardware_config);    //GPUWorkerThread
    scheduler.addSimulation(&simulation);
    scheduler.map();

    scheduler.simulationIteration();
    */



    //  system("pause");
    return 0;
}
