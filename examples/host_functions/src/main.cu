#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */
#ifdef _MSVC
#pragma warning(disable : 4100)
#endif
FLAMEGPU_AGENT_FUNCTION(device_function) {
    return ALIVE;
}
FLAMEGPU_INIT_FUNCTION(init_function) {
    float min_a = FLAMEGPU->agent("agent").min<float>("x");
    printf("Init Function! (Min: %g)\n", min_a);
}
FLAMEGPU_STEP_FUNCTION(step_function) {
    int sum_a = FLAMEGPU->agent("agent").sum<int>("a");
    printf("Step Function! (Sum: %d)\n", sum_a);
}
FLAMEGPU_EXIT_FUNCTION(exit_function) {
    float max_a = FLAMEGPU->agent("agent").max<float>("x");
    printf("Exit Function! (Max: %g)\n", max_a);
}
FLAMEGPU_HOST_FUNCTION(host_function) {
    printf("Host Function!\n");
}
FLAMEGPU_EXIT_CONDITION(exit_condition) {
    printf("Host Condition!\n");
    return CONTINUE;
}
#ifdef _MSVC
#pragma warning(op)
#endif

int main(void) {
    const unsigned int AGENT_COUNT = 1024;
    ModelDescription flame_model("host_functions_example");

    // {//circle agent
        AgentDescription agent("agent");
        agent.addAgentVariable<float>("x");
        agent.addAgentVariable<int>("a");

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
            instance.setVariable<float>("x", i);
            instance.setVariable<int>("a", 1);
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
        hostfn_layer.addHostFunction(&host_function);
    // }


    /**
     * Execution
     */
    CUDAAgentModel cuda_model(flame_model);
    cuda_model.setInitialPopulationData(population);
    cuda_model.simulate(simulation);

    cuda_model.getPopulationData(population);

    getchar();
    return 0;
}
