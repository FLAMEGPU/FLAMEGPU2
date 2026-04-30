#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include "flamegpu/flamegpu.h"
#include "flamegpu/stockAgent/subModels/SingleAgentDiscreteMovement.h"

#define ENV_DIM 100
#define SIMULATION_STEPS 100

using flamegpu::ModelDescription;
using flamegpu::AgentDescription;
using flamegpu::AgentFunctionDescription;
using flamegpu::LayerDescription;
using flamegpu::CUDASimulation;
using flamegpu::MessageNone;
using flamegpu::ALIVE;
using flamegpu::EnvironmentDescription;

// Global log file
std::ofstream agents_log;

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
        bee.setVariable<int>("last_x", -1);
        bee.setVariable<int>("last_y", -1);
        bee.setVariable<int>("last_resources_x", -1);
        bee.setVariable<int>("last_resources_y", -1);
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


FLAMEGPU_INIT_FUNCTION(initLog) {
    agents_log.open("bees_log.csv");
    agents_log << "step,id,x,y,hunger_level,wait" << std::endl;
}

FLAMEGPU_STEP_FUNCTION(stepLogger) {
    auto bees = FLAMEGPU->agent("bee");
    auto& bee_pop = bees.getPopulationData();
    unsigned int step = FLAMEGPU->getStepCounter();

    int count = 0;
    for (const auto& bee : bee_pop) {
        if (count < 5) {
            std::cout << "Bee " << bee.getID() << " at (" << bee.getVariable<int>("x") << ", " << bee.getVariable<int>("y") << ")" << std::endl;
            count++;
        }
        agents_log << step << ","
                 << bee.getID() << ","
                 << bee.getVariable<int>("x") << ","
                 << bee.getVariable<int>("y") << ","
                 << bee.getVariable<float>("hunger_level") << ","
                 << bee.getVariable<int>("wait") << "\n";
    }

    // Log cells with nectar once at the start
    if (step == 0) {
        std::ofstream flower_log("flowers_log.csv");
        flower_log << "id,x,y,nectar" << std::endl;
        auto cells = FLAMEGPU->agent("flower_cell");
        auto& cell_pop = cells.getPopulationData();
        for (const auto& cell : cell_pop) {
            float nectar = cell.getVariable<float>("nectar");
            if (nectar > 0.0f) {
                flower_log << cell.getID() << ","
                           << cell.getVariable<int>("x") << ","
                           << cell.getVariable<int>("y") << ","
                           << nectar << "\n";
            }
        }
        flower_log.close();
    }

    float avg_hunger = bees.sum<float>("hunger_level") / (float)bees.count();
    std::cout << "Step: " << step
              << " | Bee count: " << bees.count()
              << " | Avg Hunger: " << avg_hunger << std::endl;
}

FLAMEGPU_EXIT_FUNCTION(exitLog) {
    if (agents_log.is_open()) {
        agents_log.close();
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
    bee.newVariable<int>("last_x", -1);
    bee.newVariable<int>("last_y", -1);
    bee.newVariable<int>("last_resources_x", -1);
    bee.newVariable<int>("last_resources_y", -1);
    bee.newVariable<float>("hunger_level");
    bee.newVariable<int>("wait", 0);
    bee.newVariable<float>("priority", 0.0f);
    bee.newVariable<float>("current_cell_score", 0.0f);


    // Declared on the stack - uses empty constructor
    flamegpu::stockAgent::submodels::SingleAgentDiscreteMovement move_sub_logic;

    // Initialize the submodel
    auto move_sub_desc = move_sub_logic.addSingleAgentDiscreteMovementSubmodel(model, ENV_DIM, ENV_DIM);

    // Bind parent agents to submodel
    move_sub_logic.setMovingAgent("bee",
        {
            {"x", "x"},
            {"y", "y"},
            {"last_x", "last_x"},
            {"last_y", "last_y"},
            {"last_resources_x", "last_resources_x"},
            {"last_resources_y", "last_resources_y"},
            {"priority", "priority"},
            {"current_cell_score", "current_cell_score"}
        },
        {
            {"active", flamegpu::ModelData::DEFAULT_STATE}
        });

    move_sub_logic.setEnvironmentAgent("flower_cell",
        {
            {"x", "x"},
            {"y", "y"},
            {"is_occupied", "is_occupied"},
            {"cell_score", "nectar"}
        },
        {
            {"active", flamegpu::ModelData::DEFAULT_STATE}
        });


    bee.newFunction("calculate_priority", calculate_priority);
    bee.newFunction("update_hunger_wait", update_hunger_wait);

    LayerDescription l0 = model.newLayer();
    l0.addAgentFunction(calculate_priority);

    LayerDescription l1 = model.newLayer();
    l1.addSubModel(move_sub_desc);

    LayerDescription l2 = model.newLayer();
    l2.addAgentFunction(update_hunger_wait);

    model.addInitFunction(createAgent);
    model.addInitFunction(initLog);
    model.addStepFunction(stepLogger);
    model.addExitFunction(exitLog);
}

int main(int argc, const char ** argv) {
    ModelDescription model("OneAgentMovingModel");

    define_model(model);

    CUDASimulation simulation(model);
    simulation.SimulationConfig().steps = SIMULATION_STEPS;

    simulation.simulate();

    return EXIT_SUCCESS;
}
