#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "model/ModelDescription.h"
#include "pop/AgentPopulation.h"
#include "sim/Simulation.h"
#include "gpu/CUDAAgentModel.h"

using namespace std;

/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */
//#include "agent_functions.h"

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 * This main.cpp would either be specified by a user or automatically generated from the model.xml.
 * Each of the API functions will have a 121 mapping with XML elements
 * The API is very similar to FLAME 2. The directory structure and general project is set out very similarly.
 */
int main(void)
{

    /* MODEL */
    /* The model is the description of the actual model that is equivalent to that described by model.xml*/
    /* Nothing with GPUs it is only about building the model description in memory */
    ModelDescription flame_model("circles_model");

    //circle agent
    AgentDescription circle_agent("circle");
    circle_agent.addAgentVariable<float>("x");
    circle_agent.addAgentVariable<float>("y");
    circle_agent.addAgentVariable<float>("dx");
    circle_agent.addAgentVariable<float>("dy");

// TODO (mozhgan#1#05/12/16): write some tests that check the model object (model folder)
// TODO (mozhgan#1#05/12/16): Write some tests for population objects. Check that the data is correct.


    //circle add states
    //circle_agent.addState("state1");
    //circle_agent.addState("state2");


    //location message
    MessageDescription location_message("location");
    location_message.addVariable<float>("x");
    location_message.addVariable<float>("y");

    //circle agent output_data function
    //Do not specify the state. As their are no states in the system it is assumed that this model is stateless.
    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
    //output_data.setInitialState("state1");
    circle_agent.addAgentFunction(output_data);

    //circle agent input_data function
    AgentFunctionDescription input_data("input_data");
    AgentFunctionInput input_location("location");
    input_data.addInput(input_location);
    circle_agent.addAgentFunction(input_data);


    //circle agent move function
    AgentFunctionDescription move("move");
    circle_agent.addAgentFunction(move);

    //model
    flame_model.addMessage(location_message);
    flame_model.addAgent(circle_agent);

    //TODO: At some point the model should be validated and then become read only. You should not be bale to add new agent variables once you have instances of the population for example.
    //flame_model.validate();



    //TODO: globals

    // POPULATION (FLAME2 mem)
    /* Population is an instantiation of the model. It is equivalent to the data from 0.xml or any other state of the model. It requires a model description to know what the agent variables and states are. */
    /* Data in populations and instances are only on the host. No concept of GPUs at this stage. */

    AgentPopulation population(flame_model, "circle");
	//TODO: Set maximum population size if known in advance
    for (int i=0; i< 100; i++)
    {
        AgentInstance instance = population.addInstance("default");
        instance.setVariable<float>("x", i*0.1f);
        instance.setVariable<float>("y", i*0.1f);
        instance.setVariable<float>("dx", 0);
        instance.setVariable<float>("dy", 0);

        //get function would look like
        //int x = instance.getVariable<int>("x");

    }


    /* GLOBALS */
    /* TODO: We will consider this later. Not in the circles model. */


    // SIMULATION
    /* This is essentially the function layers from a model.xml */
    /* Currently this is specified by the user. In the future this could be generated automatically through dependency analysis like with FLAME HPC */

    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer"); //in the original schema layers are not named
    output_layer.addAgentFunction("output_data");			  //equivalent of layerfunction in FLAMEGPU
    simulation.addSimulationLayer(output_layer);
    //TOD: simulation.insertFunctionLayerAt(layer, int index) //Should inster at the layer position and move all other layer back

    SimulationLayer input_layer(simulation, "input_layer");
    input_layer.addAgentFunction("input_data");
    simulation.addSimulationLayer(input_layer);

    SimulationLayer move_layer(simulation, "move_layer");
    move_layer.addAgentFunction("move");
    simulation.addSimulationLayer(move_layer);

    //TODO: simulation.getLayerPosition("layer name") - returns an int
    //Simulation.addFunctionToLayer(0,"lmove")
    //This would come from the program arguments. Would be argv[2] if this file had been generated from model.xml
    simulation.setSimulationSteps(10);

    /* CUDA agent model */
    /* Instantiate the model with a set of data (agents) on the device */
    /* Run the model */
    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setPopulationData(population);

    cuda_model.simulate(simulation);

    //cuda_model.step(simulation);

    //cuda_model.getPopulation(population);

    /* This is not to be done yet. We want to first replicate the functionality of FLAMEGPU on a single device */
    /*
    // EXECUTION
    HardwareDescription hardware_config();
    hardware_config.addGPUResource(SM_30);
    hardware_config.addGPUResource(SM_20);

    // SCHEDULER and MAPPING
    //dynamic scheduling is not possible without considering the implications of communication later in the iteration (dynamic only suitable for shared memory systems unless lookahead is used)
    //mapping for now should be simple but in future a generic is based on either
    //1) memory constraints
    //2) occupancy
    //scheduler should use occupancy calculator to determine best thread block size
    GPUScheduler scheduler(&hardware_config);	//GPUWorkerThread
    scheduler.addSimulation(&simulation);
    scheduler.map();

    scheduler.simulationIteration();
    */



    system("pause");
    return 0;
}
