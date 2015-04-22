/* *
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>

#include "model/ModelDescription.h"
#include "pop/AgentPopulation.h"

/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */
//#include "agent_functions.h"

/**
 * Host function that prepares data array and passes it to the CUDA kernel.
 */
int main(void) {

	/* MODEL */
	ModelDescription flame_model("circles_model");

	//circle agent
	AgentDescription circle_agent("circle");
	circle_agent.addAgentVariable<float>("x");
	circle_agent.addAgentVariable<float>("y");
	circle_agent.addAgentVariable<float>("dx");
	circle_agent.addAgentVariable<float>("dy");

	
	//location message
	MessageDescription location_message("location");
	location_message.addVariable<float>("x");
	location_message.addVariable<float>("y");
	
	//circle agent output_data function
	AgentFunctionDescription output_data("output_data");
	AgentFunctionOutput output_location("location");
	output_data.addOutput(output_location);
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
	
	//TODO: globals

	// POPULATION (FLAME2 mem) 
	
	AgentPopulation population(flame_model, "circle");
	for (int i=0; i< 100; i++){
		AgentInstance instance = population.addInstance("default");
		instance.setVariable<float>("x", i*0.1f);
		instance.setVariable<float>("y", i*0.1f);
		instance.setVariable<float>("dx", 0);
		instance.setVariable<float>("dy", 0);
	}
	

	/* GLOBALS */

/*
	// SIMULATION
	GPUSimulation simulation(&flame_model, &population, &globals);

	SimulationLayer output_layer("output_layer");
	output_layer.addAgentFunction("output_data");
	simulation.addLayer(&input_layer);

	SimulationLayer input_layer("input_layer");
	input_layer.addAgentFunction("input_data");
	simulation.addLayer(&input_layer);

	SimulationLayer move_layer("move_layer");
	move_layer.addAgentFunction("move");
	simulation.addLayer(&move_layer);

	simulation.simulationSteps(10);

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




	return 0;
}
