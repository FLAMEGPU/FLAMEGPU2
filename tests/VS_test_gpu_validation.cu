#include "../flame_api.h"

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


using namespace std;

/**
 * @brief      To verify the correctness of set and get variable function without
 * simulating any function
 *
 * To ensure initial values for agent population is transferred correctly onto
 * the GPU, this test compares the values copied back from the device with the initial
 * population data. This test should pass.
*/
bool gpu_test_1()
{
bool equal = true;
     ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");


    circle_agent.addAgentVariable<int>("id");

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent, 100);
    for (int i = 0; i< 100; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<int>("id", i);
    }

    CUDAAgentModel cuda_model(flame_model);
    cuda_model.setInitialPopulationData(population);

    AgentPopulation population2(circle_agent, 100);
    cuda_model.getPopulationData(population2);


while(equal){
    //check values are the same
    for (int i = 0; i < 100; i++)
    {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        if(i1.getVariable<int>("id") != i2.getVariable<int>("id"))
        equal=false;
    }
    }
    return equal;

}

/**
 * @brief      To verify the correctness of ::AgentInstance::setVariable  and
 *  ::AgentInstance::getVariable variable function and hashing after simulating a function
 *
 * To ensure initial values for agent population is transferred correctly onto
 * the GPU, this test checks the correctness of the values copied back from the device
 * after being updated/changed during the simulation of an agent function.
 * This test should pass.
*/
bool gpu_test_2()
{
bool equal = true;
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");


    circle_agent.addAgentVariable<double>("x");

    AgentFunctionDescription add_data("add_data");
    AgentFunctionOutput add_location("location");
    add_data.addOutput(add_location);
    add_data.setFunction(&add_func);
    circle_agent.addAgentFunction(add_data);

    flame_model.addAgent(circle_agent);

    AgentPopulation population(circle_agent,10);
    for (int i = 0; i< 10; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        instance.setVariable<double>("x", i);
    }

    Simulation simulation(flame_model);

    SimulationLayer add_layer(simulation, "add_layer");
    add_layer.addAgentFunction("add_data");

    simulation.addSimulationLayer(add_layer);

    //simulation.setSimulationSteps(10);

    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population);

    cuda_model.addSimulation(simulation);

    cuda_model.step(simulation);


// Moz: I don't think we need this, we can simply compare it with i. See gpu_test3
    AgentPopulation population2(circle_agent, 10);
    cuda_model.getPopulationData(population2);

while(equal){
    //check values are the same
    for (int i = 0; i < 10; i++)
    {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        if(i1.getVariable<double>("x") + 2 != i2.getVariable<double>("x"))
        equal = false;
    }
    }
    retrun equal;
}

/**
 * @brief      To verify the correctness of running multiple functions simultaneously
 *
 * To test CUDA streams for overlapping host and device operations
 * This test should pass.
*/
bool gpu_test_3()
{
bool equal = true;
 /* Multi agent model */
    ModelDescription flame_model("circles_model");

    AgentDescription circle1_agent("circle1");
    circle1_agent.addAgentVariable<double>("x");

    AgentDescription circle2_agent("circle2");
    circle2_agent.addAgentVariable<double>("x");
    circle2_agent.addAgentVariable<double>("y");

    AgentFunctionDescription add_data("add_data");
    //add_data.addInput(input_location);
    add_data.setFunction(&add_func);
    circle1_agent.addAgentFunction(add_data);

    AgentFunctionDescription subtract_data("subtract_data");
    //subtract_data.addInput(input_location);
    subtract_data.setFunction(&subtract_func);
    circle2_agent.addAgentFunction(subtract_data);


    flame_model.addAgent(circle1_agent);
    flame_model.addAgent(circle2_agent);

    #define SIZE 10
    AgentPopulation population1(circle1_agent, SIZE);
    for (int i=0; i< SIZE; i++)
    {
        AgentInstance instance = population1.getNextInstance("default");
        instance.setVariable<double>("x", i);
    }

    AgentPopulation population2(circle2_agent, SIZE);
    for (int i=0; i< SIZE; i++)
    {
        AgentInstance instance = population2.getNextInstance("default");
        instance.setVariable<double>("x", i);
        instance.setVariable<double>("y", i);
    }

    Simulation simulation(flame_model);

    //multiple functions per simulation layer (from different agents)
    SimulationLayer concurrent_layer(simulation, "concurrent_layer");
    concurrent_layer.addAgentFunction("add_data");
    concurrent_layer.addAgentFunction("subtract_data");
    simulation.addSimulationLayer(concurrent_layer);

    simulation.setSimulationSteps(1);

    /* Run the model */
    CUDAAgentModel cuda_model(flame_model);

    cuda_model.setInitialPopulationData(population1);
    cuda_model.setInitialPopulationData(population2);

    cuda_model.addSimulation(simulation);

    cuda_model.step(simulation);


    cuda_model.getPopulationData(population1);
    cuda_model.getPopulationData(population2);


while(equal){
    //check values are the same
    for (int i = 0; i < SIZE; i++)
    {
        AgentInstance i1 = population1.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        if( (i2.getVariable<double>("y") != 0) || (i1.getVariable<double>("x")) != i+2)
        equal = false;
    }
    }
    retrun equal;
}
