#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>


#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"
#include "flamegpu/io/factory.h"
#include "flamegpu/util/nvtx.h"

void printPopulation(AgentPopulation &pop);

FLAMEGPU_AGENT_FUNCTION(output, MsgNone, MsgArray2D) {
    FLAMEGPU->message_out.setVariable<char>("is_alive", FLAMEGPU->getVariable<char>("is_alive"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(update, MsgArray2D, MsgNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    unsigned int living_neighbours = 0;
    // Iterate 3x3 grid
    for (auto &msg : FLAMEGPU->message_in(my_x, my_y, 2)) {
        living_neighbours += msg.getVariable<char>("is_alive") ? 1 : 0;
    }
    // Using count, decide and output new value for is_alive
    char is_alive = FLAMEGPU->getVariable<char>("is_alive");
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
    FLAMEGPU->setVariable<char>("is_alive", is_alive);
    return ALIVE;
}
int main(int argc, const char ** argv) {
    NVTX_RANGE("main");
    NVTX_PUSH("ModelDescription");
    ModelDescription model("Game_of_Life_example");

    {   // Location message
        MsgArray2D::Description &message = model.newMessage<MsgArray2D>("is_alive_msg");
        message.newVariable<char>("is_alive");
        message.setDimensions(10, 10);
    }
    {   // Cell agent
        AgentDescription &agent = model.newAgent("cell");
        agent.newVariable<unsigned int, 2>("pos");
        agent.newVariable<char>("is_alive");
        agent.newFunction("output", output).setMessageOutput("is_alive_msg");
        agent.newFunction("update", update).setMessageInput("is_alive_msg");
    }

    /**
     * GLOBALS
     */
    {
        EnvironmentDescription &env = model.Environment();
        env.add("repulse", 0.05f);
        env.add("radius", 1.0f);
    }

    /**
     * Control flow
     */
    {   // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(output);
    }
    {   // Layer #2
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(update);
    }
    NVTX_POP();

    /**
     * Create Model Runner
     */
    NVTX_PUSH("CUDAAgentModel creation");
    CUDAAgentModel cuda_model(model);
    NVTX_POP();

    /**
     * Initialisation
     */
    cuda_model.initialise(argc, argv);
    if (cuda_model.getSimulationConfig().xml_input_file.empty()) {
        // Currently population has not been init, so generate an agent population on the fly
        const unsigned int SQRT_AGENT_COUNT = 10;
        const unsigned int AGENT_COUNT = SQRT_AGENT_COUNT * SQRT_AGENT_COUNT;
        std::default_random_engine rng;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        AgentPopulation init_pop(model.Agent("cell"), AGENT_COUNT);
        for (unsigned int x = 0; x < SQRT_AGENT_COUNT; ++x) {
            for (unsigned int y = 0; y < SQRT_AGENT_COUNT; ++y) {
                AgentInstance instance = init_pop.getNextInstance();
                instance.setVariable<unsigned int, 2>("pos", { x, y });
                char is_alive = dist(rng) < 0.4f ? 1 : 0;
                instance.setVariable<char>("is_alive", is_alive);  // 40% Chance of being alive
            }
        }
        printPopulation(init_pop);
        cuda_model.setPopulationData(init_pop);
    }

    /**
     * Execution
     */
    AgentPopulation cell_pop(model.Agent("cell"));
     while (cuda_model.getStepCounter() < cuda_model.getSimulationConfig().steps && cuda_model.step()) {
        cuda_model.getPopulationData(cell_pop);
        printPopulation(cell_pop);
        getchar();
     }

    /**
     * Export Pop
     */
    // TODO
    return 0;
}
/**
 * Only works on square grids
 * Assumes grid is always in same order as output
 */
void printPopulation(AgentPopulation &pop) {
    const unsigned int dim = static_cast<unsigned int>(sqrt(pop.getCurrentListSize()));
    unsigned int i = 0;
    for (unsigned int x = 0; x < dim; ++x) {
        for (unsigned int y = 0; y < dim; ++y) {
            AgentInstance instance = pop.getInstanceAt(i++);
            printf("%s", instance.getVariable<char>("is_alive") ? "#" : " ");
        }
        printf("\n");
    }
    for (unsigned int x = 0; x < dim; ++x) {
        printf("-");
    }
    printf("\n");
}
