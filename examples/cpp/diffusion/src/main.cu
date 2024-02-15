#include "flamegpu/flamegpu.h"

/**
 * This example reimplements the heat equation
 * Based on: https://enccs.github.io/OpenACC-CUDA-beginners/2.02_cuda-heat-equation/
 */

FLAMEGPU_AGENT_FUNCTION(output, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<float>("value", FLAMEGPU->getVariable<float>("value"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(update, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int i = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int j = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    const float dx2 = FLAMEGPU->environment.getProperty<float>("dx2");
    const float dy2 = FLAMEGPU->environment.getProperty<float>("dy2");
    const float old_value = FLAMEGPU->getVariable<float>("value");

    const float left = FLAMEGPU->message_in.at(i == 0 ? FLAMEGPU->message_in.getDimX() - 1 : i - 1, j).getVariable<float>("value");
    const float up = FLAMEGPU->message_in.at(i, j == 0 ? FLAMEGPU->message_in.getDimY() - 1 : j - 1).getVariable<float>("value");
    const float right = FLAMEGPU->message_in.at(i + 1 >= FLAMEGPU->message_in.getDimX() ? 0 : i + 1, j).getVariable<float>("value");
    const float down = FLAMEGPU->message_in.at(i, j + 1 >= FLAMEGPU->message_in.getDimY() ? 0 : j + 1).getVariable<float>("value");

    // Explicit scheme
    float new_value = (left - 2.0 * old_value + right) / dx2 + (up - 2.0 * old_value + down) / dy2;

    const float a = FLAMEGPU->environment.getProperty<float>("a");
    const float dt = FLAMEGPU->environment.getProperty<float>("dt");

    new_value *= a * dt;
    new_value += old_value;

    FLAMEGPU->setVariable<float>("value", new_value);
    return flamegpu::ALIVE;
}
FLAMEGPU_EXIT_CONDITION(stable_temperature) {
    // Exit when standard deviation of temperature across agents goes below 0.006
    // (At this point it looks kind of uniform to the eye)
    const double sd = FLAMEGPU->agent("cell").meanStandardDeviation<float>("value").second;
    return sd < 0.006 ? flamegpu::EXIT : flamegpu::CONTINUE;
}
int main(int argc, const char ** argv) {
    const unsigned int SQRT_AGENT_COUNT = 200;
    const unsigned int AGENT_COUNT = SQRT_AGENT_COUNT * SQRT_AGENT_COUNT;
    flamegpu::util::nvtx::Range range{"main"};
    flamegpu::util::nvtx::push("ModelDescription");
    flamegpu::ModelDescription model("Heat Equation");

    {   // Message
        flamegpu::MessageArray2D::Description message = model.newMessage<flamegpu::MessageArray2D>("temperature");
        message.newVariable<float>("value");
        message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    }
    {   // Cell agent
        flamegpu::AgentDescription agent = model.newAgent("cell");
        agent.newVariable<unsigned int, 2>("pos");
        agent.newVariable<float>("value");
#ifdef FLAMEGPU_VISUALISATION
        // Redundant separate floating point position vars for vis
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
#endif
        agent.newFunction("output", output).setMessageOutput("temperature");
        agent.newFunction("update", update).setMessageInput("temperature");
    }

    /**
     * GLOBALS
     */
    {
        flamegpu::EnvironmentDescription  env = model.Environment();
        // Diffusion constant
        const float a = 0.5f;
        env.newProperty<float>("a", a);
        // Grid spacing
        const float dx = 0.01f;
        env.newProperty<float>("dx", dx);
        const float dy = 0.01f;
        env.newProperty<float>("dy", dy);
        // Grid spacing squared (pre-computed)
        const float dx2 = powf(dx, 2);
        env.newProperty<float>("dx2", dx2);
        const float dy2 = powf(dy, 2);
        env.newProperty<float>("dy2", dy2);
        // Largest stable timestep
        const float dt = dx2 * dy2 / (2.0f * a * (dx2 + dy2));
        env.newProperty<float>("dt", dt);
    }

    /**
     * Control flow
     */
    {   // Layer #1
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(output);
    }
    {   // Layer #2
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(update);
    }
    model.addExitCondition(stable_temperature);
    flamegpu::util::nvtx::pop();

    /**
     * Create Model Runner
     */
    flamegpu::util::nvtx::push("CUDASimulation creation");
    flamegpu::CUDASimulation cudaSimulation(model, argc, argv);
    flamegpu::util::nvtx::pop();

    /**
     * Initialisation
     */
    if (cudaSimulation.getSimulationConfig().input_file.empty()) {
        // Currently population has not been init, so generate an agent population on the fly
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        flamegpu::AgentVector init_pop(model.Agent("cell"));
        init_pop.reserve(AGENT_COUNT);
        for (unsigned int x = 0; x < SQRT_AGENT_COUNT; ++x) {
            for (unsigned int y = 0; y < SQRT_AGENT_COUNT; ++y) {
                init_pop.push_back();
                flamegpu::AgentVector::Agent instance = init_pop.back();
                instance.setVariable<unsigned int, 2>("pos", { x, y });
                instance.setVariable<float>("value", dist(rng));
#ifdef FLAMEGPU_VISUALISATION
// Redundant separate floating point position vars for vis
                instance.setVariable<float>("x", static_cast<float>(x));
                instance.setVariable<float>("y", static_cast<float>(y));
#endif
            }
        }
        cudaSimulation.setPopulationData(init_pop);
    }

    /**
     * Create visualisation
     */
#ifdef FLAMEGPU_VISUALISATION
    flamegpu::visualiser::ModelVis visualisation = cudaSimulation.getVisualisation();
    {
        visualisation.setBeginPaused(true);
        visualisation.setSimulationSpeed(5);
        visualisation.setInitialCameraLocation(SQRT_AGENT_COUNT / 2.0f, SQRT_AGENT_COUNT / 2.0f, 450.0f);
        visualisation.setInitialCameraTarget(SQRT_AGENT_COUNT / 2.0f, SQRT_AGENT_COUNT / 2.0f, 0.0f);
        visualisation.setCameraSpeed(0.001f * SQRT_AGENT_COUNT);
        visualisation.setViewClips(0.01f, 2500);
        visualisation.setClearColor(0.0f, 0.0f, 0.0f);
        visualisation.setOrthographic(true);
        visualisation.setOrthographicZoomModifier(0.284f);
        auto agt = visualisation.addAgent("cell");
        // Position vars are named x, y, z; so they are used by default
        agt.setModel(flamegpu::visualiser::Stock::Models::CUBE);  // 5 unwanted faces!
        agt.setModelScale(1.0f);
        // Assume that midpoint will be 0.5f, and any values outside this range will be lost in early steps
        agt.setColor(flamegpu::visualiser::ViridisInterpolation("value", 0.35f, 0.65f));
    }
    visualisation.activate();
#endif

    /**
     * Execution
     */
    cudaSimulation.simulate();

    /**
     * Export Pop
     */
#ifdef FLAMEGPU_VISUALISATION
    visualisation.join();
#endif

    // Ensure profiling / memcheck work correctly
    flamegpu::util::cleanup();

    return 0;
}
