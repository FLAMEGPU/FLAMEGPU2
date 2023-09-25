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
    FLAMEGPU->environment.setProperty<int>("result", FLAMEGPU->agent("Agent").sum<int>("x"));
}
int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("boids_spatial3D");
    const unsigned int POPULATION_TO_GENERATE = 100000;
    const unsigned int STEPS = 10;
    /**
     * GLOBALS
     */
     {
        flamegpu::EnvironmentDescription  env = model.Environment();

        env.newProperty<unsigned int>("POPULATION_TO_GENERATE", POPULATION_TO_GENERATE, true);

        env.newProperty<int>("init", 0);
        env.newProperty<int>("init_offset", 0);
        env.newProperty<int>("offset", 1);
        env.newProperty<int>("result", 0);
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
        runs.setPropertyLerpRange<int>("init", 0, 9);
        runs.setPropertyLerpRange<int>("init_offset", 1, 0);
        runs.setPropertyLerpRange<int>("offset", 0, 99);
    }
    /**
     * Create a logging config
     */
    flamegpu::LoggingConfig exit_log_cfg(model);
    exit_log_cfg.logEnvironment("init");
    exit_log_cfg.logEnvironment("result");
    /**
     * Create Model Runner
     */
    flamegpu::CUDAEnsemble cuda_ensemble(model, argc, argv);
    cuda_ensemble.setExitLog(exit_log_cfg);
    cuda_ensemble.simulate(runs);

    /**
     * Check result for each log
     */
    const std::map<unsigned int, flamegpu::RunLog> &logs = cuda_ensemble.getLogs();
    unsigned int init_sum = 0, expected_init_sum = 0;
    uint64_t result_sum = 0, expected_result_sum = 0;

    for (const auto &[i, log] : logs) {
        const int init = i/10;
        const int init_offset = 1 - i/50;
        expected_init_sum += init;
        expected_result_sum += POPULATION_TO_GENERATE * init + init_offset * ((POPULATION_TO_GENERATE-1)*POPULATION_TO_GENERATE/2);  // Initial agent values
        expected_result_sum += POPULATION_TO_GENERATE * STEPS * i;  // Agent values added by steps
        const flamegpu::ExitLogFrame &exit_log = log.getExitLog();
        init_sum += exit_log.getEnvironmentProperty<int>("init");
        result_sum += exit_log.getEnvironmentProperty<int>("result");
    }
    printf("Ensemble init: %u, calculated init %u\n", expected_init_sum, init_sum);
    printf("Ensemble result: %zu, calculated result %zu\n", expected_result_sum, result_sum);

    /**
     * Report if MPI was enabled
     */
    if (cuda_ensemble.Config().mpi) {
        printf("Local MPI runner completed %u/%u runs.\n", static_cast<unsigned int>(logs.size()), static_cast<unsigned int>(runs.size()));
    }

    // Ensure profiling / memcheck work correctly
    flamegpu::util::cleanup();

    return 0;
}
