/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       VS_test_sim_validation.h
 * @authors    Mozhgan Kabiri Chimeh, Paul Richmond
 * @date       16 Oct 2017
 * @brief      Test suite for validating methods in sim folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */
#include "../flame_api.h"

FLAMEGPU_AGENT_FUNCTION(output_func)
{
	//printf("Hello from output_func\n");

	// should've returned error if the type was not correct. Needs type check
	float x = FLAMEGPU->getVariable<float>("x");
	FLAMEGPU->setVariable<float>("x", x + 2);
    x = FLAMEGPU->getVariable<float>("x");
    printf("x after set = %f\n", x);

	return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(input_func)
{
	//printf("Hello from input_func\n");
	return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(move_func)
{
	//printf("Hello from move_func\n");
	return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(stay_func)
{
	//printf("Hello from stay_func\n");
	return ALIVE;
}

using namespace std;

/**
 * @brief      To verify the correctness of simulation functions
 *
 * This test checks whether functions of executing on device by printing 'Hello'.
 * This test should pass.
*/
bool sim_test_1()
{
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<float>("x");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
    output_data.setFunction(&output_func);
    //output_data.setInitialState("state1");
    circle_agent.addAgentFunction(output_data);

    AgentFunctionDescription move("move");
    move.setFunction(&move_func);
    circle_agent.addAgentFunction(move);

    AgentFunctionDescription stay("stay");
    stay.setFunction(&stay_func);
    circle_agent.addAgentFunction(stay);

    flame_model.addAgent(circle_agent);


    AgentPopulation population(circle_agent);

    for (int i=0; i< 10; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<float>("x", i*0.1f);
    }


    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer");
    output_layer.addAgentFunction("output_data");
    simulation.addSimulationLayer(output_layer);


    SimulationLayer moveStay_layer(simulation, "move_layer");
    moveStay_layer.addAgentFunction("move");
    moveStay_layer.addAgentFunction("stay");
    simulation.addSimulationLayer(moveStay_layer);


    // check number of function layers
    if(simulation.getLayerCount()!=2){
    printf("Error:Number of layers is wrong");
    }

    //for each each sim layer
    for (unsigned int i = 0; i < simulation.getLayerCount(); i++)
    {
        const FunctionDescriptionVector& functions = simulation.getFunctionsAtLayer(i);

        //for each func function
        for (AgentFunctionDescription func_des : functions)
        {
            // check functions - printing function name only
            printf("Calling agent function %s at layer %d!\n",func_des.getName(),i);
        }
    }


    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population);

    cuda_model.addSimulation(simulation);

    cuda_model.step(simulation);

    return 1;
    }

/**
 * @brief      { function_description }
 *
 * @param[in]  <unnamed>  { parameter_description }
 * @todo to test the mismatch var type
 */
bool sim_test_2()
{

    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");

    circle_agent.addAgentVariable<int>("x");

    AgentFunctionDescription output_data("output_data");
    AgentFunctionOutput output_location("location");
    output_data.addOutput(output_location);
    output_data.setFunction(&output_func);
    //output_data.setInitialState("state1");
    circle_agent.addAgentFunction(output_data);

    flame_model.addAgent(circle_agent);


    AgentPopulation population(circle_agent);

    for (int i=0; i< 10; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<int>("x", i);
    }


    Simulation simulation(flame_model);

    SimulationLayer output_layer(simulation, "output_layer");
    output_layer.addAgentFunction("output_data");
    simulation.addSimulationLayer(output_layer);


    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population);

    cuda_model.addSimulation(simulation);

    cuda_model.step(simulation);

    return 1;

}

