#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"


/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */
FLAMEGPU_AGENT_FUNCTION(device_function) {
    return ALIVE;
}
FLAMEGPU_INIT_FUNCTION(init_function) {
    float min_x = FLAMEGPU->agent("agent").min<float>("x");
    float max_x = FLAMEGPU->agent("agent").max<float>("x");
    printf("Init Function! (Min: %g, Max: %g)\n", min_x, max_x);
}
FLAMEGPU_CUSTOM_REDUCTION(customSum, a, b) {
    return a + b;
}
FLAMEGPU_STEP_FUNCTION(step_function) {
    int sum_a = FLAMEGPU->agent("agent").sum<int>("a");
    int custom_sum_a = FLAMEGPU->agent("agent").reduce<int>("a", customSum, 0);
    printf("Step Function! (Sum: %d, CustomSum: %d)\n", sum_a, custom_sum_a);
}
FLAMEGPU_EXIT_FUNCTION(exit_function) {
    float uniform_real = FLAMEGPU->random.uniform<float>();
    int uniform_int = FLAMEGPU->random.uniform<int>(1, 10);
    float normal = FLAMEGPU->random.normal<float>();
    float logNormal = FLAMEGPU->random.logNormal<float>(1, 1);
    printf("Exit Function! (%g, %i, %g, %g)\n",
        uniform_real, uniform_int, normal, logNormal);
}
FLAMEGPU_HOST_FUNCTION(host_function) {
    std::vector<int> hist_x = FLAMEGPU->agent("agent").histogramEven<float>("x", 8, -0.5, 1023.5);
    printf("Host Function! (Hist: [%d, %d, %d, %d, %d, %d, %d, %d]\n",
        hist_x[0], hist_x[1], hist_x[2], hist_x[3], hist_x[4], hist_x[5], hist_x[6], hist_x[7]);
}
#if defined(_MSC_VER)
#pragma warning(push)
#pragma warning(disable : 4100)
#endif
FLAMEGPU_EXIT_CONDITION(exit_condition) {
    printf("Host Condition!\n");
    return CONTINUE;
}
#if defined(_MSC_VER)
#pragma warning(pop)
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
            instance.setVariable<float>("x", static_cast<float>(i));
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
        simulation.addSimulationLayer(hostfn_layer);
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
