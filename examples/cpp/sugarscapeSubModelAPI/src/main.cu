#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>
#include <vector>

#include "flamegpu/flamegpu.h"
#include "flamegpu/stockAgent/subModels/SingleAgentDiscreteMovement.h"


// Grid Size (the product of these is the agent count)
#define GRID_WIDTH 256
#define GRID_HEIGHT 256

// Agent state variables
#define AGENT_STATUS_UNOCCUPIED 0
#define AGENT_STATUS_OCCUPIED 1
#define AGENT_STATUS_MOVEMENT_REQUESTED 2
#define AGENT_STATUS_MOVEMENT_UNRESOLVED 3

// Growback variables
#define SUGAR_GROWBACK_RATE 1
#define SUGAR_MAX_CAPACITY 7

// Visualisation mode (0=occupied/move status, 1=occupied/sugar/level)
#define VIS_MODE 1


FLAMEGPU_AGENT_FUNCTION(metabolise_and_growback, flamegpu::MessageNone, flamegpu::MessageNone) {
    int sugar_level = FLAMEGPU->getVariable<int>("sugar_level");
    int env_sugar_level = FLAMEGPU->getVariable<int>("env_sugar_level");
    int env_max_sugar_level = FLAMEGPU->getVariable<int>("env_max_sugar_level");
    int status = FLAMEGPU->getVariable<int>("status");
    // metabolise if occupied
    if (status == AGENT_STATUS_OCCUPIED || status == AGENT_STATUS_MOVEMENT_UNRESOLVED) {
        // store any sugar present in the cell
        if (env_sugar_level > 0) {
            sugar_level += env_sugar_level;
            // Occupied cells are marked as -1 sugar.
            env_sugar_level = -1;
        }

        // metabolise
        sugar_level -= FLAMEGPU->getVariable<int>("metabolism");

        // check if agent dies
        if (sugar_level == 0) {
            status = AGENT_STATUS_UNOCCUPIED;
            FLAMEGPU->setVariable<int>("agent_id", -1);
            env_sugar_level = 0;
            FLAMEGPU->setVariable<int>("metabolism", 0);
        }
    }

    // growback if unoccupied
    if (status == AGENT_STATUS_UNOCCUPIED) {
        env_sugar_level += SUGAR_GROWBACK_RATE;
        if (env_sugar_level > env_max_sugar_level) {
            env_sugar_level = env_max_sugar_level;
        }
    }

    // set all active agents to unresolved as they may now want to move
    if (status == AGENT_STATUS_OCCUPIED) {
        status = AGENT_STATUS_MOVEMENT_UNRESOLVED;
    }
    FLAMEGPU->setVariable<int>("sugar_level", sugar_level);
    FLAMEGPU->setVariable<int>("env_sugar_level", env_sugar_level);
    FLAMEGPU->setVariable<int>("status", status);

    return flamegpu::ALIVE;
}

/**
 * Construct the common components of agent shared between both parent and submodel
 */
flamegpu::AgentDescription makeCoreAgent(flamegpu::ModelDescription &model) {
    flamegpu::AgentDescription  agent = model.newAgent("agent");
    agent.newVariable<unsigned int, 2>("pos");
    agent.newVariable<int>("agent_id");
    agent.newVariable<int>("status");
    // agent specific variables
    agent.newVariable<int>("sugar_level");
    agent.newVariable<int>("metabolism");
    // environment specific var
    agent.newVariable<int>("env_sugar_level");
    agent.newVariable<int>("env_max_sugar_level");
#ifdef FLAMEGPU_VISUALISATION
    // Redundant seperate floating point position vars for vis
    agent.newVariable<float>("x");
    agent.newVariable<float>("y");
#endif
    return agent;
}
int main(int argc, const char ** argv) {
    flamegpu::util::nvtx::Range range{"main"};
    flamegpu::util::nvtx::push("ModelDescription");
    flamegpu::ModelDescription model("Sugarscape");


    flamegpu::stockAgent::submodels::SingleAgentDiscreteMovement move_sub_logic;

    // Initialize the submodel
    auto move_sub_desc = move_sub_logic.addSingleAgentDiscreteMovementSubmodel(model, GRID_WIDTH, GRID_HEIGHT);

    // Bind parent agent to submodel
    move_sub_logic.setMovingAgent("agent",
        {
            {"x", "x"},
            {"y", "y"},
            {"cell_score", "env_sugar_level"},
            {"is_occupied", "status"}  // Use sugar level as the movement priority
        },
        {
            // We MUST map the submodel's "active" state to our parent's state (default in this case)
            {"active", flamegpu::ModelData::DEFAULT_STATE}
        });



    /**
     * Agents
     */
    {   // Per cell agent
        flamegpu::AgentDescription  agent = makeCoreAgent(model);
        // Functions
        agent.newFunction("metabolise_and_growback", metabolise_and_growback);
    }



    /**
     * Globals
     */
    {
        // flamegpu::EnvironmentDescription  env = model.Environment();
    }

    /**
     * Control flow
     */
    {   // Layer #1
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(metabolise_and_growback);
    }
    {   // Layer #2
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addSubModel(move_sub_desc);
    }
    flamegpu::util::nvtx::pop();

    /**
     * Create Model Runner
     */
    flamegpu::util::nvtx::push("CUDASimulation creation");
    flamegpu::CUDASimulation  cudaSimulation(model);
    flamegpu::util::nvtx::pop();

    /**
     * Create visualisation
     * @note FLAMEGPU2 doesn't currently have proper support for discrete/2d visualisations
     */
#ifdef FLAMEGPU_VISUALISATION
    flamegpu::visualiser::ModelVis visualisation = cudaSimulation.getVisualisation();
    {
        visualisation.setSimulationSpeed(2);
        visualisation.setInitialCameraLocation(GRID_WIDTH / 2.0f, GRID_HEIGHT / 2.0f, 225.0f);
        visualisation.setInitialCameraTarget(GRID_WIDTH / 2.0f, GRID_HEIGHT /2.0f, 0.0f);
        visualisation.setCameraSpeed(0.001f * GRID_WIDTH);
        visualisation.setOrthographic(true);
        visualisation.setOrthographicZoomModifier(0.365f);
        visualisation.setViewClips(0.1f, 5000);

        auto agt = visualisation.addAgent("agent");
        // Position vars are named x, y, z; so they are used by default
        agt.setModel(flamegpu::visualiser::Stock::Models::CUBE);  // 5 unwanted faces!
        agt.setModelScale(1.0f);
#if VIS_MODE == 0
            flamegpu::visualiser::DiscreteColor<int> cell_colors = flamegpu::visualiser::DiscreteColor<int>("status", flamegpu::visualiser::Color{"#666"});
            cell_colors[AGENT_STATUS_UNOCCUPIED] = flamegpu::visualiser::Stock::Colors::RED;
            cell_colors[AGENT_STATUS_OCCUPIED] = flamegpu::visualiser::Stock::Colors::GREEN;
            cell_colors[AGENT_STATUS_MOVEMENT_REQUESTED] = flamegpu::visualiser::Stock::Colors::BLUE;  // Not possible, only occurs inside the submodel
            cell_colors[AGENT_STATUS_MOVEMENT_UNRESOLVED] = flamegpu::visualiser::Stock::Colors::WHITE;
            agt.setColor(cell_colors);
#else
            flamegpu::visualiser::DiscreteColor<int> cell_colors = flamegpu::visualiser::DiscreteColor<int>("env_sugar_level", flamegpu::visualiser::Stock::Palettes::Viridis(SUGAR_MAX_CAPACITY + 1), flamegpu::visualiser::Color{"#f00"});
            agt.setColor(cell_colors);
#endif
    }
    visualisation.activate();
#endif

    /**
     * Initialisation
     */
    flamegpu::util::nvtx::push("CUDASimulation initialisation");
    cudaSimulation.initialise(argc, argv);
    if (cudaSimulation.getSimulationConfig().input_file.empty()) {
        std::mt19937_64 rng;
        // Pre init, decide the sugar hotspots
        std::vector<std::array<unsigned int, 4>> sugar_hotspots;
        {
            std::uniform_int_distribution<unsigned int> width_dist(0, GRID_WIDTH-1);
            std::uniform_int_distribution<unsigned int> height_dist(0, GRID_HEIGHT-1);
            // Each sugar hotspot has a radius of 3-15 blocks
            std::uniform_int_distribution<unsigned int> radius_dist(5, 30);
            // Hostpot area should cover around 50% of the map
            float hotspot_area = 0;
            while (hotspot_area < GRID_WIDTH * GRID_HEIGHT) {
                unsigned int rad = radius_dist(rng);
                std::array<unsigned int, 4> hs = {width_dist(rng), height_dist(rng), rad, SUGAR_MAX_CAPACITY };
                sugar_hotspots.push_back(hs);
                hotspot_area += 3.141f * rad * rad;
            }
        }


        // Currently population has not been init, so generate an agent population on the fly
        const unsigned int CELL_COUNT = GRID_WIDTH * GRID_HEIGHT;
        std::uniform_real_distribution<float> normal(0, 1);
        std::uniform_int_distribution<int> agent_sugar_dist(0, SUGAR_MAX_CAPACITY * 2);
        std::uniform_int_distribution<int> poor_env_sugar_dist(0, SUGAR_MAX_CAPACITY/2);
        unsigned int i = 0;
        unsigned int agent_id = 0;
        flamegpu::AgentVector init_pop(model.Agent("agent"), CELL_COUNT);
        for (unsigned int x = 0; x < GRID_WIDTH; ++x) {
            for (unsigned int y = 0; y < GRID_HEIGHT; ++y) {
                flamegpu::AgentVector::Agent instance = init_pop[i++];
                instance.setVariable<unsigned int, 2>("pos", { x, y });
                // TODO: How should these values be init?
                // agent specific variables
                // 10% chance of cell holding an agent
                if (normal(rng) < 0.1) {
                    instance.setVariable<int>("agent_id", agent_id++);
                    instance.setVariable<int>("status", AGENT_STATUS_OCCUPIED);
                    instance.setVariable<int>("sugar_level", agent_sugar_dist(rng)/2);  // Agent sugar dist 0-3, less chance of 0
                    instance.setVariable<int>("metabolism", 6);
                } else {
                    instance.setVariable<int>("agent_id", -1);
                    instance.setVariable<int>("status", AGENT_STATUS_UNOCCUPIED);
                    instance.setVariable<int>("sugar_level", 0);
                    instance.setVariable<int>("metabolism", 0);
                }
                // environment specific var
                unsigned int env_sugar_lvl = 0;
                const int hotspot_core_size = 5;
                for (auto &hs : sugar_hotspots) {
                    // Workout the highest sugar lvl from a nearby hotspot
                    int hs_x = static_cast<int>(std::get<0>(hs));
                    int hs_y = static_cast<int>(std::get<1>(hs));
                    unsigned int hs_rad = std::get<2>(hs);
                    unsigned int hs_level = std::get<3>(hs);
                    float hs_dist = static_cast<float>(sqrt(pow(hs_x-static_cast<int>(x), 2.0) + pow(hs_y-static_cast<int>(y), 2.0)));
                    if (hs_dist <= hotspot_core_size) {
                        unsigned int t = hs_level;
                        env_sugar_lvl = t > env_sugar_lvl ? t : env_sugar_lvl;
                    } else if (hs_dist <= hs_rad) {
                        int non_core_len = hs_rad - hotspot_core_size;
                        float dist_from_core = hs_dist - hotspot_core_size;
                        unsigned int t = static_cast<unsigned int>(hs_level * (non_core_len - dist_from_core) / non_core_len);
                        env_sugar_lvl = t > env_sugar_lvl ? t : env_sugar_lvl;
                    }
                }
                env_sugar_lvl = env_sugar_lvl < SUGAR_MAX_CAPACITY / 2 ? poor_env_sugar_dist(rng) : env_sugar_lvl;
                instance.setVariable<int>("env_max_sugar_level", env_sugar_lvl);  // All cells begin at their local max sugar
                instance.setVariable<int>("env_sugar_level", env_sugar_lvl);
#ifdef FLAMEGPU_VISUALISATION
                // Redundant separate floating point position vars for vis
                instance.setVariable<float>("x", static_cast<float>(x));
                instance.setVariable<float>("y", static_cast<float>(y));
#endif
            }
        }
        cudaSimulation.setPopulationData(init_pop);
    }
    flamegpu::util::nvtx::pop();

    /**
     * Execution
     */
    cudaSimulation.simulate();

    /**
     * Export Pop
     */
    // cudaSimulation.exportData("end.xml");

#ifdef FLAMEGPU_VISUALISATION
    visualisation.join();
#endif

    // Ensure profiling / memcheck work correctly
    flamegpu::util::cleanup();

    return 0;
}
