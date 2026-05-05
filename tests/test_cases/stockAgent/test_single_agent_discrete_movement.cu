#include "flamegpu/flamegpu.h"
#include "flamegpu/stockAgent/subModels/SingleAgentDiscreteMovement.h"
#include "gtest/gtest.h"

namespace flamegpu {
namespace stockAgent {
namespace submodels {

/**
 * Test 1: Initialization & Validation
 * Verifies that the submodel correctly validates its agent and variable bindings.
 */
TEST(SingleAgentDiscreteMovementTest, Initialization) {
    ModelDescription model("parent_model");
    SingleAgentDiscreteMovement move_submodel;
    
    // Should throw if we try to bind before calling addSingleAgentDiscreteMovementSubmodel
    EXPECT_THROW(move_submodel.setMovingAgent("agent"), exception::InvalidSubModel);
    
    // Initialize the submodel
    move_submodel.addSingleAgentDiscreteMovementSubmodel(model, 10, 10);
    
    // Setup a valid parent agent for moving
    auto agent = model.newAgent("agent");
    agent.newVariable<int>("x");
    agent.newVariable<int>("y");
    agent.newVariable<int>("last_x");
    agent.newVariable<int>("last_y");
    agent.newVariable<int>("last_resources_x");
    agent.newVariable<int>("last_resources_y");
    agent.newVariable<float>("current_cell_score");
    agent.newState("default");

    // Setup a valid parent agent for the environment grid
    auto cell = model.newAgent("cell");
    cell.newVariable<int>("x");
    cell.newVariable<int>("y");
    cell.newVariable<int>("is_occupied");
    cell.newVariable<float>("cell_score");
    cell.newState("default");

    // Bind agents: auto_map=true handles variables with matching names, 
    // but we must explicitly map internal "active" state to parent "default" state.
    move_submodel.setMovingAgent("agent", {}, {{"active", "default"}}, true);
    move_submodel.setEnvironmentAgent("cell", {}, {{"active", "default"}}, true);

    // Validation should pass now
    EXPECT_NO_THROW(move_submodel.validate());
}

/**
 * Test 2: Basic Greedy Movement
 * Verifies that an agent at (0,0) moves to (1,1) if that cell has a higher score.
 */
TEST(SingleAgentDiscreteMovementTest, SimpleMove) {
    ModelDescription model("parent_model");
    int WIDTH = 3;
    int HEIGHT = 3;

    SingleAgentDiscreteMovement move_submodel;
    auto smd = move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT);
    
    // Submodels must be added to a layer to be executed during the simulation step
    model.newLayer().addSubModel(smd);

    // Define parent agents with necessary variables
    auto agent = model.newAgent("agent");
    agent.newVariable<int>("x");
    agent.newVariable<int>("y");
    agent.newVariable<int>("last_x", -1);
    agent.newVariable<int>("last_y", -1);
    agent.newVariable<int>("last_resources_x", -1);
    agent.newVariable<int>("last_resources_y", -1);
    agent.newVariable<float>("current_cell_score", 0.0f);
    agent.newVariable<float>("priority", 1.0f);
    agent.newState("default");

    auto cell = model.newAgent("cell");
    cell.newVariable<int>("x");
    cell.newVariable<int>("y");
    cell.newVariable<int>("is_occupied", 0);
    cell.newVariable<float>("cell_score", 0.0f);
    cell.newState("default");

    // Bind to submodel
    move_submodel.setMovingAgent("agent", {}, {{"active", "default"}}, true);
    move_submodel.setEnvironmentAgent("cell", {}, {{"active", "default"}}, true);

    CUDASimulation sim(model);
    sim.SimulationConfig().steps = 1;

    // Initialize the grid: 3x3 cells
    auto cell_pop = AgentVector(cell, WIDTH * HEIGHT);
    for (int x = 0; x < WIDTH; ++x) {
        for (int y = 0; y < HEIGHT; ++y) {
            auto c = cell_pop[x * HEIGHT + y];
            c.setVariable<int>("x", x);
            c.setVariable<int>("y", y);
            c.setVariable<int>("is_occupied", 0);
            c.setVariable<float>("cell_score", 0.0f);
        }
    }
    // Set a high "reward" at (1, 1)
    cell_pop[1 * HEIGHT + 1].setVariable<float>("cell_score", 10.0f);
    sim.setPopulationData(cell_pop);

    // Initialize 1 agent at (0, 0)
    auto agent_pop = AgentVector(agent, 1);
    agent_pop[0].setVariable<int>("x", 0);
    agent_pop[0].setVariable<int>("y", 0);
    agent_pop[0].setVariable<float>("priority", 1.0f);
    sim.setPopulationData(agent_pop);

    // Execute one simulation step
    sim.step();

    // Verify the agent moved to the high-score cell (1, 1)
    sim.getPopulationData(agent_pop);
    EXPECT_EQ(agent_pop[0].getVariable<int>("x"), 1);
    EXPECT_EQ(agent_pop[0].getVariable<int>("y"), 1);
    
    // Verify occupancy status was updated
    sim.getPopulationData(cell_pop);
    EXPECT_EQ(cell_pop[0 * HEIGHT + 0].getVariable<int>("is_occupied"), 0); // Left (0,0)
    EXPECT_EQ(cell_pop[1 * HEIGHT + 1].getVariable<int>("is_occupied"), 1); // Entered (1,1)
}

}  // namespace submodels
}  // namespace stockAgent
}  // namespace flamegpu
