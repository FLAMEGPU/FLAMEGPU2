#include "flamegpu/flamegpu.h"

FLAMEGPU_AGENT_FUNCTION(output, flamegpu::MsgNone, flamegpu::MsgArray2D) {
    FLAMEGPU->message_out.setVariable<char>("is_alive", FLAMEGPU->getVariable<unsigned int>("is_alive"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(update, flamegpu::MsgArray2D, flamegpu::MsgNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    unsigned int living_neighbours = 0;
    // Iterate 3x3 Moore neighbourhood (this does no include the central cell)
    for (auto &msg : FLAMEGPU->message_in(my_x, my_y)) {
        living_neighbours += msg.getVariable<char>("is_alive") ? 1 : 0;
    }
    // Using count, decide and output new value for is_alive
    char is_alive = FLAMEGPU->getVariable<unsigned int>("is_alive");
    if (is_alive) {
        if (living_neighbours < 2)
            is_alive = 0;
        else if (living_neighbours > 3)
            is_alive = 0;
        else  // exactly 2 or 3 living_neighbours
            is_alive = 1;
    } else {
        if (living_neighbours == 3)
            is_alive = 1;
    }
    FLAMEGPU->setVariable<unsigned int>("is_alive", is_alive);
    return flamegpu::ALIVE;
}
int main(int argc, const char ** argv) {
    const unsigned int SQRT_AGENT_COUNT = 1000;
    const unsigned int AGENT_COUNT = SQRT_AGENT_COUNT * SQRT_AGENT_COUNT;
    NVTX_RANGE("main");
    NVTX_PUSH("ModelDescription");
    flamegpu::ModelDescription model("Game_of_Life_example");

    {   // Location message
        flamegpu::MsgArray2D::Description &message = model.newMessage<flamegpu::MsgArray2D>("is_alive_msg");
        message.newVariable<char>("is_alive");
        message.setDimensions(SQRT_AGENT_COUNT, SQRT_AGENT_COUNT);
    }
    {   // Cell agent
        flamegpu::AgentDescription  &agent = model.newAgent("cell");
        agent.newVariable<unsigned int, 2>("pos");
        agent.newVariable<unsigned int>("is_alive");
#ifdef VISUALISATION
        // Redundant separate floating point position vars for vis
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
#endif
        agent.newFunction("output", output).setMessageOutput("is_alive_msg");
        agent.newFunction("update", update).setMessageInput("is_alive_msg");
    }

    /**
     * GLOBALS
     */
    {
        flamegpu::EnvironmentDescription  &env = model.Environment();
        env.newProperty("repulse", 0.05f);
        env.newProperty("radius", 1.0f);
    }

    /**
     * Control flow
     */
    {   // Layer #1
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(output);
    }
    {   // Layer #2
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(update);
    }
    NVTX_POP();

    /**
     * Create Model Runner
     */
    NVTX_PUSH("CUDASimulation creation");
    flamegpu::CUDASimulation  cuda_model(model, argc, argv);
    NVTX_POP();

    /**
     * Initialisation
     */
    if (cuda_model.getSimulationConfig().input_file.empty()) {
        // Currently population has not been init, so generate an agent population on the fly
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        flamegpu::AgentVector init_pop(model.Agent("cell"));
        init_pop.reserve(AGENT_COUNT);
        for (unsigned int x = 0; x < SQRT_AGENT_COUNT; ++x) {
            for (unsigned int y = 0; y < SQRT_AGENT_COUNT; ++y) {
                init_pop.push_back();
                flamegpu::AgentVector::Agent instance = init_pop.back();
                instance.setVariable<unsigned int, 2>("pos", { x, y });
                char is_alive = dist(rng) < 0.4f ? 1 : 0;
                instance.setVariable<unsigned int>("is_alive", is_alive);  // 40% Chance of being flamegpu::ALIVE
#ifdef VISUALISATION
// Redundant separate floating point position vars for vis
                instance.setVariable<float>("x", static_cast<float>(x));
                instance.setVariable<float>("y", static_cast<float>(y));
#endif
            }
        }
        cuda_model.setPopulationData(init_pop);
    }

    /**
     * Create visualisation
     * @note FGPU2 doesn't currently have proper support for discrete/2d visualisations
     */
#ifdef VISUALISATION
    flamegpu::visualiser::ModelVis & visualisation = cuda_model.getVisualisation();
    {
        visualisation.setBeginPaused(true);
        visualisation.setSimulationSpeed(5);
        visualisation.setInitialCameraLocation(SQRT_AGENT_COUNT / 2.0f, SQRT_AGENT_COUNT / 2.0f, 450.0f);
        visualisation.setInitialCameraTarget(SQRT_AGENT_COUNT / 2.0f, SQRT_AGENT_COUNT / 2.0f, 0.0f);
        visualisation.setCameraSpeed(0.001f * SQRT_AGENT_COUNT);
        visualisation.setViewClips(0.01f, 2500);
        visualisation.setClearColor(0.6f, 0.6f, 0.6f);
        auto& agt = visualisation.addAgent("cell");
        // Position vars are named x, y, z; so they are used by default
        agt.setModel(flamegpu::visualiser::Stock::Models::CUBE);  // 5 unwanted faces!
        agt.setModelScale(1.0f);
        flamegpu::visualiser::DiscreteColor<unsigned int> cell_colors = flamegpu::visualiser::DiscreteColor<unsigned int>("is_alive", flamegpu::visualiser::Color{ "#666" });
        cell_colors[0] = flamegpu::visualiser::Stock::Colors::BLACK;
        cell_colors[1] = flamegpu::visualiser::Stock::Colors::WHITE;
        agt.setColor(cell_colors);
    }
    visualisation.activate();
#endif

    /**
     * Execution
     */
    cuda_model.simulate();

    /**
     * Export Pop
     */
#ifdef VISUALISATION
    visualisation.join();
#endif
    return 0;
}
