#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>

#include "flamegpu/flamegpu.h"
#include "flamegpu/stockAgent/subModels/SingleAgentDiscreteMovement.h"

// Grid Size
#define GRID_WIDTH 256
#define GRID_HEIGHT 256

// Growback variables
#define SUGAR_GROWBACK_RATE 1.0f
#define SUGAR_MAX_CAPACITY 7.0f

/**
 * Agent Functions
 */

// 1. Metabolise & Harvest: Bug eats the sugar at its new location and consumes energy
// This runs AFTER movement, so current_cell_score is already updated by the submodel.
FLAMEGPU_AGENT_FUNCTION(metabolise, flamegpu::MessageNone, flamegpu::MessageNone) {
    float sugar = FLAMEGPU->getVariable<float>("sugar");
    float metabolism = FLAMEGPU->getVariable<float>("metabolism");
    float harvested = FLAMEGPU->getVariable<float>("current_cell_score");

    // Add what we found and subtract what we used
    sugar += harvested;
    sugar -= metabolism;

    // Death check
    if (sugar <= 0.0f) {
        return flamegpu::DEAD;
    }

    FLAMEGPU->setVariable<float>("sugar", sugar);
    return flamegpu::ALIVE;
}

// 2. Growback: SugarCell grows sugar or is emptied if a bug is currently standing on it
FLAMEGPU_AGENT_FUNCTION(growback, flamegpu::MessageNone, flamegpu::MessageNone) {
    float sugar = FLAMEGPU->getVariable<float>("sugar");
    float max_sugar = FLAMEGPU->getVariable<float>("max_sugar");
    int is_occupied = FLAMEGPU->getVariable<int>("is_occupied");

    if (is_occupied) {
        // A bug is here, so it has eaten the sugar
        sugar = 0.0f;
    } else {
        // Grow back
        sugar += SUGAR_GROWBACK_RATE;
        if (sugar > max_sugar) {
            sugar = max_sugar;
        }
    }

    FLAMEGPU->setVariable<float>("sugar", sugar);
    return flamegpu::ALIVE;
}

/**
 * Step function to log simulation state to CSV
 */
FLAMEGPU_STEP_FUNCTION(step_logger) {
    unsigned int step = FLAMEGPU->getStepCounter();
    unsigned int bug_count = FLAMEGPU->agent("bug").count();
    printf("Step %u: bugs=%u\n", step, bug_count);

    // Log bugs every step
    static std::ofstream bug_log;
    if (step == 0) {
        bug_log.open("bugs_log.csv");
        bug_log << "step,x,y,sugar,metabolism" << std::endl;
    }
    // getPopulationData returns a reference to a DeviceAgentVector_impl, so we must use a reference
    auto& bug_pop = FLAMEGPU->agent("bug").getPopulationData();
    for (const auto& bug : bug_pop) {
        bug_log << step << "," << bug.getVariable<int>("x") << "," << bug.getVariable<int>("y") << "," << bug.getVariable<float>("sugar") << "," << bug.getVariable<float>("metabolism") << std::endl;
    }

    // Log cells every 10 steps to save space (and step 0)
    static std::ofstream cell_log;
    if (step == 0) {
        cell_log.open("cells_log.csv");
        cell_log << "step,x,y,sugar,max_sugar" << std::endl;
    }
    if (step % 10 == 0) {
        auto& cell_pop = FLAMEGPU->agent("sugar_cell").getPopulationData();
        for (const auto& cell : cell_pop) {
            cell_log << step << "," << cell.getVariable<int>("x") << "," << cell.getVariable<int>("y") << "," << cell.getVariable<float>("sugar") << "," << cell.getVariable<float>("max_sugar") << std::endl;
        }
    }
}

/**
 * Main
 */
int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("Sugarscape");

    /**
     * Agents
     */
    // Bug Agent (The moving agent)
    flamegpu::AgentDescription bug = model.newAgent("bug");
    bug.newVariable<float>("sugar");
    bug.newVariable<float>("metabolism");
    bug.newVariable<int>("x");
    bug.newVariable<int>("y");
    bug.newVariable<int>("last_x", -1);
    bug.newVariable<int>("last_y", -1);
    bug.newVariable<int>("last_resources_x", -1);
    bug.newVariable<int>("last_resources_y", -1);
    bug.newVariable<float>("current_cell_score", 0.0f);

    // SugarCell Agent (The environment agent)
    flamegpu::AgentDescription sugar_cell = model.newAgent("sugar_cell");
    sugar_cell.newVariable<int>("x");
    sugar_cell.newVariable<int>("y");
    sugar_cell.newVariable<float>("sugar");
    sugar_cell.newVariable<float>("max_sugar");
    sugar_cell.newVariable<int>("is_occupied", 0);

    /**
     * Submodel Configuration
     */
    flamegpu::stockAgent::submodels::SingleAgentDiscreteMovement move_sub_logic;
    auto move_sub_desc = move_sub_logic.addSingleAgentDiscreteMovementSubmodel(model, GRID_WIDTH, GRID_HEIGHT);

    // Bind Bug to the submodel's moving agent
    move_sub_logic.setMovingAgent("bug",
        {
            {"x", "x"},
            {"y", "y"},
            {"last_x", "last_x"},
            {"last_y", "last_y"},
            {"last_resources_x", "last_resources_x"},
            {"last_resources_y", "last_resources_y"},
            {"current_cell_score", "current_cell_score"}
        },
        {
            {"active", flamegpu::ModelData::DEFAULT_STATE}
        });

    // Bind SugarCell to the submodel's environment agent
    move_sub_logic.setEnvironmentAgent("sugar_cell",
        {
            {"x", "x"},
            {"y", "y"},
            {"is_occupied", "is_occupied"},
            {"cell_score", "sugar"}
        },
        {
            {"active", flamegpu::ModelData::DEFAULT_STATE}
        });

    /**
     * Functions and Layers
     */
    bug.newFunction("metabolise", metabolise).setAllowAgentDeath(true);
    sugar_cell.newFunction("growback", growback);

    //  Layer 1: Movement
    model.newLayer().addSubModel(move_sub_desc);

    // Layer 2: Life logic (Metabolism and Growback can happen in parallel)
    {
        auto l = model.newLayer();
        l.addAgentFunction(metabolise);
        l.addAgentFunction(growback);
    }

    model.addStepFunction(step_logger);

    /**
     * Simulation Setup
     */
    flamegpu::CUDASimulation cudaSimulation(model);
    cudaSimulation.initialise(argc, argv);
    cudaSimulation.SimulationConfig().steps = 100;

    // If no input file, generate a random starting state
    if (cudaSimulation.getSimulationConfig().input_file.empty()) {
        std::mt19937_64 rng(42);
        // Define sugar hotspots (spatial distribution of max_sugar)
        std::vector<std::array<unsigned int, 4>> sugar_hotspots;
        {
            std::uniform_int_distribution<unsigned int> width_dist(0, GRID_WIDTH - 1);
            std::uniform_int_distribution<unsigned int> height_dist(0, GRID_HEIGHT - 1);
            std::uniform_int_distribution<unsigned int> radius_dist(15, 45);
            float hotspot_area = 0;
            while (hotspot_area < (GRID_WIDTH * GRID_HEIGHT) * 0.6f) {
                unsigned int rad = radius_dist(rng);
                std::array<unsigned int, 4> hs = {width_dist(rng), height_dist(rng), rad, (unsigned int)SUGAR_MAX_CAPACITY};
                sugar_hotspots.push_back(hs);
                hotspot_area += 3.141f * rad * rad;
            }
        }

        // Generate a shuffled list of all grid indices to place bugs uniquely
        std::vector<int> indices(GRID_WIDTH * GRID_HEIGHT);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);

        const float bug_density = 0.05f;
        const unsigned int BUG_COUNT = (unsigned int)((GRID_WIDTH * GRID_HEIGHT) * bug_density);

        std::vector<bool> bug_at(GRID_WIDTH * GRID_HEIGHT, false);
        flamegpu::AgentVector bug_pop(bug, BUG_COUNT);
        std::uniform_real_distribution<float> bug_sugar_dist(10.0f, 30.0f);
        std::uniform_real_distribution<float> bug_metabolism_dist(1.0f, 2.5f);

        for (unsigned int i = 0; i < BUG_COUNT; ++i) {
            int idx = indices[i];
            bug_at[idx] = true;
            auto instance = bug_pop[i];
            instance.setVariable<int>("x", idx / GRID_HEIGHT);
            instance.setVariable<int>("y", idx % GRID_HEIGHT);
            instance.setVariable<float>("sugar", bug_sugar_dist(rng));
            instance.setVariable<float>("metabolism", bug_metabolism_dist(rng));
        }

        flamegpu::AgentVector cell_pop(sugar_cell, GRID_WIDTH * GRID_HEIGHT);
        for (unsigned int x = 0; x < GRID_WIDTH; ++x) {
            for (unsigned int y = 0; y < GRID_HEIGHT; ++y) {
                unsigned int idx = x * GRID_HEIGHT + y;
                auto instance = cell_pop[idx];
                instance.setVariable<int>("x", (int)x);
                instance.setVariable<int>("y", (int)y);
                instance.setVariable<int>("is_occupied", bug_at[idx] ? 1 : 0);

                float max_val = 0;
                for (auto &hs : sugar_hotspots) {
                    float dx = (float)hs[0] - (float)x;
                    float dy = (float)hs[1] - (float)y;
                    float dist = sqrtf(dx*dx + dy*dy);
                    if (dist < (float)hs[2]) {
                        float v = (float)hs[3] * (1.0f - (dist / (float)hs[2]));
                        if (v > max_val) max_val = v;
                    }
                }
                instance.setVariable<float>("max_sugar", max_val);
                instance.setVariable<float>("sugar", max_val);
            }
        }

        cudaSimulation.setPopulationData(bug_pop);
        cudaSimulation.setPopulationData(cell_pop);
    }

    return 0;
}
