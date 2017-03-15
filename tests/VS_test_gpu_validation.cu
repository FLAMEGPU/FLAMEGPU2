#include "../flame_api.h"

FLAMEGPU_AGENT_FUNCTION(add_func)
{
    // should've returned error if the type was not correct. Needs type check
    double x = FLAMEGPU->getVariable<double>("m");

    FLAMEGPU->setVariable<double>("m", x + 2);
    x = FLAMEGPU->getVariable<double>("m");

    return ALIVE;
}


using namespace std;

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

// the test should verify the correctness of get/set variable and hashing, however every 4th variable in the array is updated
bool gpu_test_2()
{
bool equal = true;
    ModelDescription flame_model("circles_model");
    AgentDescription circle_agent("circle");


    circle_agent.addAgentVariable<double>("m");

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
        instance.setVariable<double>("m", i);
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


    AgentPopulation population2(circle_agent, 10);
    cuda_model.getPopulationData(population2);

while(equal){
    //check values are the same
    for (int i = 0; i < 10; i++)
    {
        AgentInstance i1 = population.getInstanceAt(i, "default");
        AgentInstance i2 = population2.getInstanceAt(i, "default");
        //use AgentInstance equality operator
        if(i1.getVariable<double>("m") + 2 != i2.getVariable<double>("m"))
        equal = false;
    }
    }
    retrun equal;
}


