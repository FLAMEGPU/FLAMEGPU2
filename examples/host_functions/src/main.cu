#include "flamegpu/flamegpu.h"

const unsigned int AGENT_COUNT = 1024;

FLAMEGPU_AGENT_FUNCTION(device_function, flamegpu::MessageNone, flamegpu::MessageNone) {
    const float prop_float = FLAMEGPU->environment.getProperty<float>("float");
    const int16_t prop_int16 = FLAMEGPU->environment.getProperty<int16_t>("int16_t");
    const uint64_t prop_uint64_0 = FLAMEGPU->environment.getProperty<uint64_t, 3>("uint64_t", 0);
    const uint64_t prop_uint64_1 = FLAMEGPU->environment.getProperty<uint64_t, 3>("uint64_t", 1);
    const uint64_t prop_uint64_2 = FLAMEGPU->environment.getProperty<uint64_t, 3>("uint64_t", 2);
    if (blockIdx.x * blockDim.x + threadIdx.x == 0) {
        printf("Agent Function[Thread 0]! Properties(Float: %g, int16: %hd, uint64[3]: {%llu, %llu, %llu})\n", prop_float, prop_int16, prop_uint64_0, prop_uint64_1, prop_uint64_2);
    }
    return flamegpu::ALIVE;
}
FLAMEGPU_INIT_FUNCTION(init_function) {
    flamegpu::HostAgentAPI agent = FLAMEGPU->agent("agent");
    float min_x = agent.min<float>("x");
    float max_x = agent.max<float>("x");
    printf("Init Function! (AgentCount: %u, Min: %g, Max: %g)\n", FLAMEGPU->agent("agent").count(), min_x, max_x);
    for (unsigned int i = AGENT_COUNT / 2; i < AGENT_COUNT; i++) {
        flamegpu::HostNewAgentAPI instance = agent.newAgent();
        instance.setVariable<float>("x", static_cast<float>(i));
        instance.setVariable<int>("a", i % 2 == 0 ? 1 : 0);
    }
    printf("Init Function! Created %u more agents\n", AGENT_COUNT / 2);
}
FLAMEGPU_CUSTOM_REDUCTION(customSum, a, b) {
    return a + b;
}
FLAMEGPU_CUSTOM_TRANSFORM(customTransform, a) {
    return (a == 0 || a == 1) ? 1 : 0;
}
FLAMEGPU_STEP_FUNCTION(step_function) {
    auto agent = FLAMEGPU->agent("agent");
    int sum_a = agent.sum<int>("a");
    int custom_sum_a = agent.reduce<int>("a", customSum, 0);
    unsigned int count_a = agent.count<int>("a", 1);
    unsigned int countif_a = agent.transformReduce<int, unsigned int>("a", customTransform, customSum, 0u);
    printf("Step Function! (AgentCount: %u, Sum: %d, CustomSum: %d, Count: %u, CustomCountIf: %u)\n", agent.count(), sum_a, custom_sum_a, count_a, countif_a);
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
    std::vector<unsigned int> hist_x = FLAMEGPU->agent("agent").histogramEven<float>("x", 8, -0.5, 1023.5);
    printf("Host Function! (Hist: [%u, %u, %u, %u, %u, %u, %u, %u]\n",
        hist_x[0], hist_x[1], hist_x[2], hist_x[3], hist_x[4], hist_x[5], hist_x[6], hist_x[7]);
    FLAMEGPU->environment.setProperty<int16_t>("int16_t", FLAMEGPU->environment.getProperty<int16_t>("int16_t") + 1);
}
FLAMEGPU_EXIT_CONDITION(exit_condition) {
    const float CHANCE = 0.15f;
    float uniform_real = FLAMEGPU->random.uniform<float>();
    printf("Exit Condition! (Rolled: %g)\n", uniform_real);
    if (uniform_real < CHANCE) {
        printf("Rolled number is less than %g, exiting!\n", CHANCE);
        return flamegpu::EXIT;
    }
    return flamegpu::CONTINUE;
}


int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("host_functions_example");

    {  // agent
        flamegpu::AgentDescription  agent = model.newAgent("agent");
        agent.newVariable<float>("x");
        agent.newVariable<int>("a");
        agent.newFunction("device_function", device_function);
    }

    /**
     * GLOBALS
     */
    {
        flamegpu::EnvironmentDescription  envProperties = model.Environment();
        envProperties.newProperty<float>("float", 12.0f);
        envProperties.newProperty<int16_t>("int16_t", 0);
        envProperties.newProperty<uint64_t, 3>("uint64_t", {11llu, 12llu, 13llu});
    }
    /**
     * Control flow
     */     
     {  // Attach init/step/exit functions and exit condition
        model.addInitFunction(init_function);
        model.addStepFunction(step_function);
        model.addExitFunction(exit_function);
        model.addExitCondition(exit_condition);
     }

     {
        flamegpu::LayerDescription devicefn_layer = model.newLayer("devicefn_layer");
        devicefn_layer.addAgentFunction(device_function);
     }

     {
        flamegpu::LayerDescription hostfn_layer = model.newLayer("hostfn_layer");
        hostfn_layer.addHostFunction(host_function);
     }

    /**
     * Initialisation
     */
    flamegpu::AgentVector population(model.Agent("agent"), AGENT_COUNT/2);
    for (unsigned int i = 0; i < AGENT_COUNT/2; i++) {
        flamegpu::AgentVector::Agent instance = population[i];
        instance.setVariable<float>("x", static_cast<float>(i));
        instance.setVariable<int>("a", i % 2 == 0 ? 1 : 0);
    }

    /**
     * Execution
     */
    flamegpu::CUDASimulation cudaSimulation(model);
    cudaSimulation.SimulationConfig().steps = 0;
    cudaSimulation.setPopulationData(population);
    cudaSimulation.initialise(argc, argv);
    cudaSimulation.simulate();

    cudaSimulation.getPopulationData(population);

    // Ensure profiling / memcheck work correctly
    flamegpu::util::cleanup();

    return 0;
}
