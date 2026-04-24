#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include "flamegpu/flamegpu.h"
#include "flamegpu/stockAgent/subModels/AddOneAgent.h"

#define ENV_DIM 100
#define SIMULATION_STEPS 100

using namespace flamegpu;

// Global log file
std::ofstream agents_log;

FLAMEGPU_AGENT_FUNCTION(calculate_priority, MessageNone, MessageNone) {
    int wait = FLAMEGPU->getVariable<int>("wait");
    
    // Priority = wait time + random tie breaker [0, 1)
    float priority = (float)wait + FLAMEGPU->random.uniform<float>();
    FLAMEGPU->setVariable<float>("priority", priority);
    
    return ALIVE;
}

/**
 * If the movingAgent moved, reset its wait counter. Otherwise, increment it.
 */
FLAMEGPU_AGENT_FUNCTION(update_wait_status, MessageNone, MessageNone) {
    int moved = FLAMEGPU->getVariable<int>("moved_this_step");
    int wait = FLAMEGPU->getVariable<int>("wait");
    
    if (moved == 1) {
        wait = 0;
    } else {
        wait += 1;
    }
    
    FLAMEGPU->setVariable<int>("wait", wait);
    return ALIVE;
}

FLAMEGPU_INIT_FUNCTION(createAgent) {
    const int GRID_DIM = 100;
    const int NUM_AGENTS = 2000;
    auto agent_api = FLAMEGPU->agent("movingAgent", "active");
    
    std::vector<int> available_indices(GRID_DIM * GRID_DIM);
    std::iota(available_indices.begin(), available_indices.end(), 0);
    
    std::mt19937 g(std::random_device{}());
    std::shuffle(available_indices.begin(), available_indices.end(), g);

    for (int i = 0; i < NUM_AGENTS; ++i) {
        int index = available_indices[i];
        int x = index / GRID_DIM;
        int y = index % GRID_DIM;
        
        auto agent = agent_api.newAgent();
        agent.setVariable<int>("x", x);
        agent.setVariable<int>("y", y);
        agent.setVariable<int>("wait", 0);
        agent.setVariable<float>("priority", 0.0f);
        agent.setVariable<int>("moved_this_step", 0);
    }
}

FLAMEGPU_INIT_FUNCTION(initLog) {
    agents_log.open("agents_log.csv");
    agents_log << "step,id,x,y,wait" << std::endl;
}

FLAMEGPU_STEP_FUNCTION(stepLogger) {
    auto movingAgents = FLAMEGPU->agent("movingAgent", "active");
    auto& agent_pop = movingAgents.getPopulationData();
    unsigned int step = FLAMEGPU->getStepCounter();
    
    std::vector<int> occupancy(ENV_DIM * ENV_DIM, 0);
    int collisions = 0;

    for (const auto& agent : agent_pop) {
        int x = agent.getVariable<int>("x");
        int y = agent.getVariable<int>("y");

        int idx = x * ENV_DIM + y;
        occupancy[idx]++;
        if (occupancy[idx] > 1) {
            collisions++;
        }

        agents_log << step << ","
                 << agent.getID() << ","
                 << x << ","
                 << y << ","
                 << agent.getVariable<int>("wait") << "\n";
    }

    if (collisions > 0) {
        std::cerr << "!!! STEP " << step << ": DETECTED " << collisions << " COLLISIONS !!!" << std::endl;
    }
    
    std::cout << "Step: " << step 
              << " | Agent count: " << movingAgents.count() 
              << " | Collisions: " << collisions << std::endl;
}

FLAMEGPU_EXIT_FUNCTION(exitLog) {
    if (agents_log.is_open()) {
        agents_log.close();
    }
}

void define_model(ModelDescription &model) {
    AgentDescription movingAgent = model.newAgent("movingAgent");
    movingAgent.newState("active");
    movingAgent.newVariable<int>("x");
    movingAgent.newVariable<int>("y");
    movingAgent.newVariable<int>("wait", 0);
    movingAgent.newVariable<float>("priority", 0.0f);
    movingAgent.newVariable<int>("moved_this_step", 0);

    // Declared on the stack - uses empty constructor
    flamegpu::stockAgent::submodels::AddOneAgent move_sub_logic;
    
    // Initialize the submodel
    auto move_sub_desc = move_sub_logic.addOneAgentSubmodel(model, ENV_DIM, ENV_DIM);

    // Bind parent agent to submodel
    move_sub_logic.setAgent("movingAgent", 
        {
            {"x", "x"},
            {"y", "y"},
            {"priority", "priority"},
            {"moved_this_step", "moved_this_step"}
        },
        {
            {"active", "active"}
        }
    );

    AgentFunctionDescription calc_priority = movingAgent.newFunction("calculate_priority", calculate_priority);
    calc_priority.setInitialState("active");
    calc_priority.setEndState("active");

    AgentFunctionDescription upd_wait = movingAgent.newFunction("update_wait_status", update_wait_status);
    upd_wait.setInitialState("active");
    upd_wait.setEndState("active");

    LayerDescription l0 = model.newLayer();
    l0.addAgentFunction(calc_priority);

    LayerDescription l1 = model.newLayer();
    l1.addSubModel(move_sub_desc); 

    LayerDescription l2 = model.newLayer();
    l2.addAgentFunction(upd_wait);

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
