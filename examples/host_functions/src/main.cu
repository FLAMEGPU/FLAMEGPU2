#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */
#pragma warning(disable : 4100)
FLAMEGPU_AGENT_FUNCTION(device_function) {
    return ALIVE;
}
FLAMEGPU_HOST_FUNCTION(init_function) {
    printf("Init Function!\n");
}
FLAMEGPU_HOST_FUNCTION(step_function) {
    printf("Step Function!\n");
}
FLAMEGPU_EXIT_FUNCTION(exit_function) {
    printf("Exit Function!\n");
}
FLAMEGPU_HOST_FUNCTION(host_function) {
    printf("Host Function!\n");
}
FLAMEGPU_EXIT_CONDITION(exit_condition) {
    printf("Host Condition!\n");
    return CONTINUE;
}
#pragma warning(op)

int main(void) {
    const unsigned int AGENT_COUNT = 1024;
    ModelDescription flame_model("host_functions_example");

    // {//circle agent
        AgentDescription agent("agent");
        agent.addAgentVariable<float>("x");
        agent.addAgentVariable<float>("y");

        // {// Device fn
            AgentFunctionDescription deviceFn("device_function");
            deviceFn.setFunction(&device_function);
            agent.addAgentFunction(deviceFn);
        // }

        flame_model.addAgent(agent);

        // Init pop
        AgentPopulation population(agent, AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            AgentInstance instance = population.getNextInstance();
            instance.setVariable<float>("x", i*0.1f);
            instance.setVariable<float>("y", i*0.1f);
        }
    // }

    /**
     * GLOBALS
     */

     /**
     * Simulation
     */

    Simulation simulation(flame_model);

    // Attach init/step/exit functions and exit condition
    // {
        simulation.addInitFunction(&init_function);
        simulation.addStepFunction(&step_function);
        simulation.addExitFunction(&exit_function);
        simulation.addExitCondition(&exit_condition);
        // Run until exit condition triggers
        simulation.setSimulationSteps(5);
    // }

    // {
        SimulationLayer devicefn_layer(simulation, "devicefn_layer");
        devicefn_layer.addAgentFunction("device_function");
        simulation.addSimulationLayer(devicefn_layer);
        // TODO: simulation.insertFunctionLayerAt(layer, int index) //Should insert at the layer position and move all other layer back
    // }

    // {
        SimulationLayer hostfn_layer(simulation, "hostfn_layer");
    // }


    /**
     * Execution
     */
    CUDAAgentModel cuda_model(flame_model);
    cuda_model.setInitialPopulationData(population);
    cuda_model.simulate(simulation);

    cuda_model.simulate(simulation);

    cuda_model.getPopulationData(population);

    getchar();
    return 0;
}
