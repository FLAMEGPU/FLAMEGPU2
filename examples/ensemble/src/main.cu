#include "flamegpu/flamegpu.h"

FLAMEGPU_AGENT_FUNCTION(AddOffset, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Output each agents publicly visible properties.
    FLAMEGPU->setVariable<int>("x", FLAMEGPU->getVariable<int>("x") + FLAMEGPU->environment.getProperty<int>("offset"));
    return flamegpu::ALIVE;
}
FLAMEGPU_INIT_FUNCTION(Init) {
    const unsigned int POPULATION_TO_GENERATE = FLAMEGPU->environment.getProperty<unsigned int>("POPULATION_TO_GENERATE");
    const int init = FLAMEGPU->environment.getProperty<int>("init");
    const int init_offset = FLAMEGPU->environment.getProperty<int>("init_offset");
    auto agent = FLAMEGPU->agent("Agent");
    for (unsigned int i = 0; i < POPULATION_TO_GENERATE; ++i) {
        agent.newAgent().setVariable<int>("x", init + i * init_offset);
    }
}
std::atomic<unsigned int> atomic_init = {0};
std::atomic<uint64_t> atomic_result = {0};
FLAMEGPU_EXIT_FUNCTION(Exit) {
    atomic_init += FLAMEGPU->environment.getProperty<int>("init");
    atomic_result += FLAMEGPU->agent("Agent").sum<int>("x");
}
int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("boids_spatial3D");
    const unsigned int POPULATION_TO_GENERATE = 100000;
    const unsigned int STEPS = 10;
    /**
     * GLOBALS
     */
     {
        flamegpu::EnvironmentDescription  &env = model.Environment();

        env.newProperty<unsigned int>("POPULATION_TO_GENERATE", POPULATION_TO_GENERATE, true);

        env.newProperty<int>("init", 0);
        env.newProperty<int>("init_offset", 0);
        env.newProperty<int>("offset", 1);
    }
    {   // Agent
        flamegpu::AgentDescription  agent = model.newAgent("Agent");
        agent.newVariable<int>("x");
        agent.newFunction("AddOffset", AddOffset);
    }

    /**
     * Control flow
     */     
    {   // Layer #1
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(AddOffset);

        model.addInitFunction(Init);
        model.addExitFunction(Exit);
    }

    /**
     * Create a run plan
     */
    flamegpu::RunPlanVector runs(model, 100);
    {
        runs.setSteps(STEPS);
        runs.setRandomSimulationSeed(12, 1);
        runs.setPropertyUniformDistribution<int>("init", 0, 9);
        runs.setPropertyUniformDistribution<int>("init_offset", 1, 0);
        runs.setPropertyUniformDistribution<int>("offset", 0, 99);
    }

    /**
     * Create Model Runner
     */
    flamegpu::CUDAEnsemble cuda_ensemble(model, argc, argv);

    cuda_ensemble.simulate(runs);

    // Check result
    // Don't currently have logging
    unsigned int init_sum = 0;
    uint64_t result_sum = 0;
    for (int i = 0 ; i < 100; ++i) {
        const int init = i/10;
        const int init_offset = 1 - i/50;
        init_sum += init;
        result_sum += POPULATION_TO_GENERATE * init + init_offset * ((POPULATION_TO_GENERATE-1)*POPULATION_TO_GENERATE/2);  // Initial agent values
        result_sum += POPULATION_TO_GENERATE * STEPS * i;  // Agent values added by steps
    }
    printf("Ensemble init: %u, calculated init %u\n", atomic_init.load(), init_sum);
    printf("Ensemble result: %zu, calculated result %zu\n", atomic_result.load(), result_sum);

    // Ensure profiling / memcheck work correctly
    flamegpu::util::cleanup();

    return 0;
}
