#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <filesystem>
#include "flamegpu/flamegpu.h"
#include "flamegpu/stock/subModels/SingleAgentDiscreteMovement.h"

#define ENV_DIM 100

using flamegpu::ModelDescription;
using flamegpu::AgentDescription;
using flamegpu::AgentFunctionDescription;
using flamegpu::LayerDescription;
using flamegpu::CUDASimulation;
using flamegpu::StepLoggingConfig;
using flamegpu::MessageNone;
using flamegpu::ALIVE;
using flamegpu::EnvironmentDescription;

CUDASimulation *global_sim = nullptr;
const char *global_out_dir = "output/";

FLAMEGPU_AGENT_FUNCTION(calculate_priority, MessageNone, MessageNone) {
    float current_nectar = FLAMEGPU->getVariable<float>("current_cell_score");
    float hunger_level = FLAMEGPU->getVariable<float>("hunger_level");

    // If at a flower and still hungry, stay put (priority 0)
    if (current_nectar > 0.01f && hunger_level > 0.0f) {
        FLAMEGPU->setVariable<float>("priority", 0.0f);
        // Ensure submodel doesn't move us if we want to stay
        FLAMEGPU->setVariable<float>("current_cell_score", 1000.0f);
        return ALIVE;
    }

    int wait = FLAMEGPU->getVariable<int>("wait");
    float wh = FLAMEGPU->environment.getProperty<float>("WH");
    float ww = FLAMEGPU->environment.getProperty<float>("WW");

    // Priority for movement (higher = more likely to win a cell)
    float priority = hunger_level * wh + (float)wait * ww + FLAMEGPU->random.uniform<float>(0.0f, 1.0f);
    FLAMEGPU->setVariable<float>("priority", priority);

    // Force movement by setting current_cell_score to a low value.
    // This ensures any neighbor with score >= 0.0 will be considered a valid move.
    FLAMEGPU->setVariable<float>("current_cell_score", -1.0f);

    return ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(update_hunger_wait, MessageNone, MessageNone) {
    float current_nectar = FLAMEGPU->getVariable<float>("current_cell_score");
    float hunger_level = FLAMEGPU->getVariable<float>("hunger_level");
    int wait = FLAMEGPU->getVariable<int>("wait");

    if (current_nectar > 0.01f && hunger_level > 0.0f) {
        // Feed: decrease hunger_level
        hunger_level -= 5.0f;
        if (hunger_level <= 0.0f) {
            hunger_level = 0.0f;
        }
        wait = 0;
    } else {
        // Hunger increases over time
        hunger_level += 2.0f;
        wait += 1;
    }

    FLAMEGPU->setVariable<float>("hunger_level", hunger_level);
    FLAMEGPU->setVariable<int>("wait", wait);

    return ALIVE;
}

FLAMEGPU_INIT_FUNCTION(createAgent) {
    const int GRID_DIM = 100;
    const int FLOWER_SPACING = 5;

    // Create bees at random unique positions first
    const int NUM_BEES = 100;
    auto bee_api = FLAMEGPU->agent("bee");

    std::vector<int> available_indices(GRID_DIM * GRID_DIM);
    std::iota(available_indices.begin(), available_indices.end(), 0);

    std::mt19937 g(std::random_device {}());
    std::shuffle(available_indices.begin(), available_indices.end(), g);

    std::vector<bool> is_bee_at(GRID_DIM * GRID_DIM, false);

    for (int i = 0; i < NUM_BEES; ++i) {
        int index = available_indices[i];
        int x = index / GRID_DIM;
        int y = index % GRID_DIM;
        is_bee_at[index] = true;

        auto bee = bee_api.newAgent();
        bee.setVariable<int>("x", x);
        bee.setVariable<int>("y", y);
        bee.setVariable<float>("hunger_level", FLAMEGPU->random.uniform<float>(0.0f, 100.0f));
        bee.setVariable<int>("wait", 0);
        bee.setVariable<float>("priority", 0.0f);
        bee.setVariable<float>("current_cell_score", 0.0f);
    }

    // Create a 100x100 grid of cells and set occupancy
    auto cell_api = FLAMEGPU->agent("flower_cell");
    for (int i = 0; i < GRID_DIM; ++i) {
        for (int j = 0; j < GRID_DIM; ++j) {
            int index = i * GRID_DIM + j;
            auto cell = cell_api.newAgent();
            cell.setVariable<int>("x", i);
            cell.setVariable<int>("y", j);
            cell.setVariable<int>("is_occupied", is_bee_at[index] ? 1 : 0);

            float nectar = 0.0f;
            if (i % FLOWER_SPACING == 0 && j % FLOWER_SPACING == 0) {
                nectar = FLAMEGPU->random.uniform<float>(10.0f, 50.0f);
            }
            cell.setVariable<float>("nectar", nectar);
        }
    }
}

FLAMEGPU_STEP_FUNCTION(stepLogger) {
    auto bees = FLAMEGPU->agent("bee");
    unsigned int step = FLAMEGPU->getStepCounter();

    float avg_hunger = bees.sum<float>("hunger_level") / (float)bees.count();
    std::cout << "Step: " << step
              << " | Bee count: " << bees.count()
              << " | Avg Hunger: " << avg_hunger << std::endl;

    // Export data per step using global pointer
    if (global_sim) {
        global_sim->exportData(global_out_dir + "step_" + std::to_string(step) + ".xml");
    }
}

void define_model(ModelDescription &model) {
    // Environment variables
    EnvironmentDescription env = model.Environment();
    env.newProperty<float>("WH", 0.6f);
    env.newProperty<float>("WW", 0.4f);

    // Cell Agent
    AgentDescription cell = model.newAgent("flower_cell");
    cell.newVariable<int>("x");
    cell.newVariable<int>("y");
    cell.newVariable<int>("is_occupied", 0);
    cell.newVariable<float>("nectar", 0.0f);

    // Bee Agent
    AgentDescription bee = model.newAgent("bee");
    bee.newVariable<int>("x");
    bee.newVariable<int>("y");
    bee.newVariable<float>("hunger_level");
    bee.newVariable<int>("wait", 0);
    bee.newVariable<float>("priority", 0.0f);
    bee.newVariable<float>("current_cell_score", 0.0f);


    // Initialize the submodel using the constructor
    flamegpu::stock::submodels::SingleAgentDiscreteMovement move_sub_logic(model, ENV_DIM, ENV_DIM);

    // Bind parent agents to submodel
    move_sub_logic.setMovingAgent("bee",
        {
            {"x", "x"},
            {"y", "y"},
            {"priority", "priority"},
            {"current_cell_score", "current_cell_score"}
        },
        {});

    move_sub_logic.setEnvironmentAgent("flower_cell",
        {
            {"x", "x"},
            {"y", "y"},
            {"is_occupied", "is_occupied"},
            {"cell_score", "nectar"}
        },
        {});


    bee.newFunction("calculate_priority", calculate_priority);
    bee.newFunction("update_hunger_wait", update_hunger_wait);

    LayerDescription l0 = model.newLayer();
    l0.addAgentFunction(calculate_priority);

    LayerDescription l1 = model.newLayer();
    l1.addSubModel(move_sub_logic.getSubModelDescription());

    LayerDescription l2 = model.newLayer();
    l2.addAgentFunction(update_hunger_wait);

    model.addInitFunction(createAgent);
    model.addStepFunction(stepLogger);
}

int main(int argc, const char ** argv) {
    // Determine output directory based on execution context
    if (std::filesystem::exists("examples/cpp/beesandflowers")) {
        global_out_dir = "examples/cpp/beesandflowers/output/";
    }
    std::filesystem::create_directories(global_out_dir);

    ModelDescription model("OneAgentMovingModel");

    define_model(model);

    CUDASimulation simulation(model);

    // Assign global pointer for step export
    global_sim = &simulation;

    // Set defaults before initialising (allows CLI to override)
    simulation.SimulationConfig().steps = 100;
    simulation.SimulationConfig().truncate_log_files = true;

    // Configure logging using the FLAMEGPU API
    StepLoggingConfig step_log(model);
    step_log.agent("bee").logCount();
    step_log.agent("bee").logMean<float>("hunger_level");
    step_log.agent("bee").logMean<int>("wait");
    simulation.setStepLog(step_log);

    simulation.initialise(argc, argv);

    // Export initial state
    simulation.exportData(global_out_dir + "initial_state.xml");

    // Run the simulation normally
    simulation.simulate();

    // Export the summary log
    simulation.exportLog(global_out_dir + "simulation_log.json", true, true, false, false);

    return EXIT_SUCCESS;
}
