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

#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "runtime/flame_api.h"


using namespace std;
#define SIZE 10  // default value for the pop and msg size

/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */


FLAMEGPU_AGENT_FUNCTION(output_func)
{
    printf("Hello from output_func\n");
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
    
	printf("[get (x,y)]: x = %f, y = %f\n", x, y);

    FLAMEGPU->setVariable<float>("x", x + 3);
    x = FLAMEGPU->getVariable<float>("x");
   
	printf("[set (x)]: x = %f, y = %f\n", x, y);

	FLAMEGPU->addMessage<float>("x", x);
	FLAMEGPU->addMessage<float>("y", y);

	/* Possible ways of adding message. Chose number 2)
	1)
	FLAMEGPU->addMessage("location1", x, y);  using variadic functions instead?
	
	2)
    FLAMEGPU->addMessage("x", x);
	FLAMEGPU->addMessage("y", y);

	3)
	FLAMEGPU->addMessage("location1", "x", x);
	FLAMEGPU->addMessage("location1", "y", y); 
	
	4)
	FLAMEGPU->setMessageVariable<float>("x", x);
	FLAMEGPU->setMessageVariable<float>("y", y);
	*/

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(input_func)
{
   printf("Hello from input_func\n");
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");

    printf("[get (x,y)]: x = %f, y = %f\n", x, y);
    
	FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");

	printf("[set (x)]: x = %f, y = %f\n", x, y);

	// 0) not interested
	float x1 = FLAMEGPU->getMessageVariable<float>("x");
	float y1 = FLAMEGPU->getMessageVariable<float>("y");
	printf("(input func - get msg): x = %f, y = %f\n", x1, y1);

	MessageList ml = FLAMEGPU->GetMessageIterator("location1");

	// 1) First method	
	for (MessageList::iterator it = ml.begin(); it != ml.end(); ++it)
	{
		float m_x = ml.getVariable<float>(it, "x");
		float m_x2 = it.getVariable<float>("x");
		//auto m = *it; 
		//m.getVariable<float>(ml, "x");
		printf("(input func - for loop, get msg variable): x = %f  %f\n", m_x, m_x2);
	}
/*
	for (auto i = mi.begin(); i != mi.end(); i++)
	{
		float m_x = mi.getVariable<float>("x");
		printf("(input func - for loop, get msg variable): x = %f\n", m_x);
	}
	*/
	// 2) Second method
	/*
	for(Message m : ml) {
	// If the Message class needs to know about the message list (ML) and the message list needs to know about the Message class - @todo - forward declaration or similar to avoid dependancy
		float m_x = m.getVariable<float>(ml, "x");
	}
	*/

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(add_func)
{
   //printf("Hello from add_func\n");
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
   printf("-y = %f, x = %f\n", y, x);
    FLAMEGPU->setVariable<float>("y", y + x);
    y = FLAMEGPU->getVariable<float>("y");
   printf("-y after set = %f\n", y);
    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(subtract_func)
{
    //printf("Hello from subtract_func\n");
    float x = FLAMEGPU->getVariable<float>("x");
    float y = FLAMEGPU->getVariable<float>("y");
    printf("y = %f, x = %f\n", y, x);
    FLAMEGPU->setVariable<float>("y", x - y);
    y = FLAMEGPU->getVariable<float>("y");
    printf("y after set = %f\n", y);
    return ALIVE;
}

int main(void)
{
    /* Multi agent model */
    ModelDescription flame_model("circles_model");
	
    AgentDescription circle1_agent("circle1");
    circle1_agent.addAgentVariable<float>("x");
    circle1_agent.addAgentVariable<float>("y");

    AgentDescription circle2_agent("circle2");
    circle2_agent.addAgentVariable<float>("x");
    circle2_agent.addAgentVariable<float>("y");

    //same name ?
    MessageDescription location1_message("location1",SIZE);
    location1_message.addVariable<float>("x");
    location1_message.addVariable<float>("y");
/*
    MessageDescription location2_message("location2");
    location2_message.addVariable<float>("x");
    location2_message.addVariable<float>("y");
*/

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location1");
    output_data.addOutput(output_location);
    output_data.setFunction(&output_func);
    circle1_agent.addAgentFunction(output_data);

    AgentFunctionDescription input_data("input_data");
    AgentFunctionInput input_location("location1");
    input_data.addInput(input_location);
    input_data.setFunction(&input_func);
    circle2_agent.addAgentFunction(input_data);

    AgentFunctionDescription add_data("add_data");
    //add_data.addInput(input_location);
    add_data.setFunction(&add_func);
    circle1_agent.addAgentFunction(add_data);

    AgentFunctionDescription subtract_data("subtract_data");
    //subtract_data.addInput(input_location);
    subtract_data.setFunction(&subtract_func);
    circle2_agent.addAgentFunction(subtract_data);


    //model
    flame_model.addMessage(location1_message);
    flame_model.addAgent(circle1_agent);

    //flame_model.addMessage(location2_message);
    flame_model.addAgent(circle2_agent);


	//1)
	flame_model.initialise(inputfile, circle1_agent);
	//2)
    AgentPopulation population1(circle1_agent, SIZE);
    for (int i=0; i< SIZE; i++)
    {
        AgentInstance instance = population1.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
        instance.setVariable<float>("y", i*0.1f);
    }

    AgentPopulation population2(circle2_agent, SIZE);
    for (int i=0; i< SIZE; i++)
    {
        AgentInstance instance = population2.getNextInstance("default");
        instance.setVariable<float>("x", i*0.2f);
        instance.setVariable<float>("y", i*0.2f);
    }

    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer");
    output_layer.addAgentFunction("output_data");
    simulation.addSimulationLayer(output_layer);

    SimulationLayer input_layer(simulation, "input_layer");
    input_layer.addAgentFunction("input_data");
    simulation.addSimulationLayer(input_layer);

    //multiple functions per simulation layer (from different agents)
    SimulationLayer concurrent_layer(simulation, "concurrent_layer");
    concurrent_layer.addAgentFunction("add_data");
    concurrent_layer.addAgentFunction("subtract_data");
    simulation.addSimulationLayer(concurrent_layer);

    simulation.setSimulationSteps(2); // steps>1 --> does not work for now

    /* Run the model */
    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population1);
    cuda_model.setInitialPopulationData(population2);

	cuda_model.setMessageData(location1_message);

    //cuda_model.addSimulation(simulation); 

    //cuda_model.step(simulation); // replaced by below
	cuda_model.simulate(simulation);

    cuda_model.getPopulationData(population1);
    cuda_model.getPopulationData(population2);

    return 0;
}

