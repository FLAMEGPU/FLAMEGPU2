#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <filesystem>

#include "flamegpu/flamegpu.h"
#include "flamegpu/stock/subModels/SingleAgentDiscreteMovement.h"

// Grid Size
#define GRID_WIDTH 256
#define GRID_HEIGHT 256

// Growback variables
#define SUGAR_GROWBACK_RATE 1.0f
#define SUGAR_MAX_CAPACITY 7.0f

flamegpu::CUDASimulation *global_sim = nullptr;

/**
 * Agent Functions
 */

// 1. Metabolise: Bug eats the sugar at its new location and consumes energy
// This runs AFTER movement, so current_cell_score is already updated by the submodel.
FLAMEGPU_AGENT_FUNCTION(metabolise, flamegpu::MessageNone, flamegpu::MessageNone) {
    float sugar_level = FLAMEGPU->getVariable<float>("sugar_level");
    float metabolism = FLAMEGPU->getVariable<float>("metabolism");
    float harvested = FLAMEGPU->getVariable<float>("current_cell_score");

    // Add what we found and subtract what we used
    if (harvested > 0) {
        sugar_level += harvested;
    }
    sugar_level -= metabolism;

    // Death check
    if (sugar_level <= 0.0f) {
        return flamegpu::DEAD;
    }

    FLAMEGPU->setVariable<float>("sugar_level", sugar_level);
#ifdef FLAMEGPU_VISUALISATION
    FLAMEGPU->setVariable<float>("vis_x", static_cast<float>(FLAMEGPU->getVariable<int>("x")));
    FLAMEGPU->setVariable<float>("vis_y", static_cast<float>(FLAMEGPU->getVariable<int>("y")));
#endif
    return flamegpu::ALIVE;
}

// 2. Growback: SugarCell grows sugar or is emptied if a bug is currently standing on it
FLAMEGPU_AGENT_FUNCTION(growback, flamegpu::MessageNone, flamegpu::MessageNone) {
    float env_sugar_level = FLAMEGPU->getVariable<float>("env_sugar_level");
    float env_max_sugar_level = FLAMEGPU->getVariable<float>("env_max_sugar_level");
    int is_occupied = FLAMEGPU->getVariable<int>("is_occupied");

    if (is_occupied) {
        // A bug is here, so it has eaten the sugar. Mark as -1 to mirror original.
        env_sugar_level = -1.0f;
    } else {
        // Grow back
        env_sugar_level += SUGAR_GROWBACK_RATE;
        if (env_sugar_level > env_max_sugar_level) {
            env_sugar_level = env_max_sugar_level;
        }
        // Ensure it's not negative if it was just vacated
        if (env_sugar_level < 0) env_sugar_level = 0;
    }

    FLAMEGPU->setVariable<float>("env_sugar_level", env_sugar_level);
    return flamegpu::ALIVE;
}

/**
 * Step function to log simulation state
 */
FLAMEGPU_STEP_FUNCTION(step_logger) {
    unsigned int step = FLAMEGPU->getStepCounter();
    unsigned int bug_count = FLAMEGPU->agent("bug").count();
    printf("Step %u: bugs=%u\n", step, bug_count);

    if (global_sim) {
        global_sim->exportData("output/step_" + std::to_string(step) + ".xml");
    }
}

/**
 * Main
 */
int main(int argc, const char ** argv) {
    std::filesystem::create_directories("output/");

    flamegpu::ModelDescription model("Sugarscape");

    /**
     * Agents
     */
    // Bug Agent (The moving agent)
    flamegpu::AgentDescription bug = model.newAgent("bug");
    bug.newVariable<float>("sugar_level");
    bug.newVariable<float>("metabolism");
    bug.newVariable<int>("x");
    bug.newVariable<int>("y");
    bug.newVariable<float>("current_cell_score", 0.0f);
#ifdef FLAMEGPU_VISUALISATION
    bug.newVariable<float>("vis_x");
    bug.newVariable<float>("vis_y");
    bug.newVariable<float>("vis_z");
#endif

    // SugarCell Agent (The environment agent)
    flamegpu::AgentDescription sugar_cell = model.newAgent("sugar_cell");
    sugar_cell.newVariable<int>("x");
    sugar_cell.newVariable<int>("y");
    sugar_cell.newVariable<float>("env_sugar_level");
    sugar_cell.newVariable<float>("env_max_sugar_level");
    sugar_cell.newVariable<int>("is_occupied", 0);
#ifdef FLAMEGPU_VISUALISATION
    sugar_cell.newVariable<float>("vis_x");
    sugar_cell.newVariable<float>("vis_y");
    sugar_cell.newVariable<float>("vis_z");
#endif

    /**
     * Submodel Configuration
     */
    flamegpu::stock::submodels::SingleAgentDiscreteMovement move_sub_logic(model, GRID_WIDTH, GRID_HEIGHT);

    // Bind Bug to the submodel's moving agent
    move_sub_logic.setMovingAgent("bug",
        {
            {"x", "x"},
            {"y", "y"},
            {"current_cell_score", "current_cell_score"}
        },
        {});

    // Bind SugarCell to the submodel's environment agent
    move_sub_logic.setEnvironmentAgent("sugar_cell",
        {
            {"x", "x"},
            {"y", "y"},
            {"is_occupied", "is_occupied"},
            {"cell_score", "env_sugar_level"}
        },
        {});

    /**
     * Functions and Layers
     */
    bug.newFunction("metabolise", metabolise).setAllowAgentDeath(true);
    sugar_cell.newFunction("growback", growback);

    //  Layer 1: Movement
    model.newLayer().addSubModel(move_sub_logic.getSubModelDescription());

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
    global_sim = &cudaSimulation;

    cudaSimulation.SimulationConfig().steps = 100;
    cudaSimulation.SimulationConfig().truncate_log_files = true;

    flamegpu::StepLoggingConfig step_log(model);
    step_log.agent("bug").logCount();
    step_log.agent("bug").logMean<float>("sugar_level");
    step_log.agent("sugar_cell").logMean<float>("env_sugar_level");
    cudaSimulation.setStepLog(step_log);

#ifdef FLAMEGPU_VISUALISATION
    flamegpu::visualiser::ModelVis visualisation = cudaSimulation.getVisualisation();
    {
        visualisation.setSimulationSpeed(2);
        visualisation.setInitialCameraLocation(GRID_WIDTH / 2.0f, GRID_HEIGHT / 2.0f, 225.0f);
        visualisation.setInitialCameraTarget(GRID_WIDTH / 2.0f, GRID_HEIGHT / 2.0f, 0.0f);
        visualisation.setCameraSpeed(0.001f * GRID_WIDTH);
        visualisation.setOrthographic(true);
        visualisation.setOrthographicZoomModifier(0.365f);
        visualisation.setViewClips(0.1f, 5000);

        auto bug_agt = visualisation.addAgent("bug");
        bug_agt.setModel(flamegpu::visualiser::Stock::Models::CUBE);
        bug_agt.setModelScale(0.5f);
        bug_agt.setXVariable("vis_x");
        bug_agt.setYVariable("vis_y");
        bug_agt.setZVariable("vis_z");
        bug_agt.setColor(flamegpu::visualiser::Stock::Colors::RED);

        auto cell_agt = visualisation.addAgent("sugar_cell");
        cell_agt.setModel(flamegpu::visualiser::Stock::Models::CUBE);
        cell_agt.setModelScale(1.0f);
        cell_agt.setXVariable("vis_x");
        cell_agt.setYVariable("vis_y");
        cell_agt.setZVariable("vis_z");
        cell_agt.setColor(flamegpu::visualiser::ViridisInterpolation("env_sugar_level", 0.0f, SUGAR_MAX_CAPACITY));
    }
    visualisation.activate();
#endif

    cudaSimulation.initialise(argc, argv);

    // If no input file, generate a random starting state
    if (cudaSimulation.getSimulationConfig().input_file.empty()) {
        std::mt19937_64 rng(42);
        // Define sugar hotspots (spatial distribution of env_max_sugar_level)
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
            instance.setVariable<float>("sugar_level", bug_sugar_dist(rng));
            instance.setVariable<float>("metabolism", bug_metabolism_dist(rng));
#ifdef FLAMEGPU_VISUALISATION
            instance.setVariable<float>("vis_x", static_cast<float>(idx / GRID_HEIGHT));
            instance.setVariable<float>("vis_y", static_cast<float>(idx % GRID_HEIGHT));
            instance.setVariable<float>("vis_z", 0.1f);
#endif
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
                instance.setVariable<float>("env_max_sugar_level", max_val);
                instance.setVariable<float>("env_sugar_level", max_val);
#ifdef FLAMEGPU_VISUALISATION
                instance.setVariable<float>("vis_x", static_cast<float>(x));
                instance.setVariable<float>("vis_y", static_cast<float>(y));
                instance.setVariable<float>("vis_z", 0.0f);
#endif
            }
        }

        cudaSimulation.setPopulationData(bug_pop);
        cudaSimulation.setPopulationData(cell_pop);
    }

    cudaSimulation.exportData("output/initial_state.xml");

    cudaSimulation.simulate();

    cudaSimulation.exportLog("output/simulation_log.json", true, true, false, false);

#ifdef FLAMEGPU_VISUALISATION
    visualisation.join();
#endif

    return 0;
}
