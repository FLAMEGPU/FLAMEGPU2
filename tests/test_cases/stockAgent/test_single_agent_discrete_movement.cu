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
    EXPECT_EQ(cell_pop[0 * HEIGHT + 0].getVariable<int>("is_occupied"), 0);  // Left (0,0)
    EXPECT_EQ(cell_pop[1 * HEIGHT + 1].getVariable<int>("is_occupied"), 1);  // Entered (1,1)
}

/**
 * Test 3: Collision Avoidance
 * Verifies that when two agents try to move to the same cell, only one succeeds.
 * Note: Now relies on the submodel's InitFunction to auto-sync occupancy.
 */
TEST(SingleAgentDiscreteMovementTest, CollisionAvoidance) {
    ModelDescription model("parent_model");
    int WIDTH = 3;
    int HEIGHT = 3;

    SingleAgentDiscreteMovement move_submodel;
    auto smd = move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT);
    model.newLayer().addSubModel(smd);

    auto agent = model.newAgent("agent");
    agent.newVariable<int>("x");
    agent.newVariable<int>("y");
    agent.newVariable<int>("last_x", -1);
    agent.newVariable<int>("last_y", -1);
    agent.newVariable<int>("last_resources_x", -1);
    agent.newVariable<int>("last_resources_y", -1);
    agent.newVariable<float>("current_cell_score", 0.0f);
    agent.newVariable<float>("priority", 0.0f);
    agent.newState("default");

    auto cell = model.newAgent("cell");
    cell.newVariable<int>("x");
    cell.newVariable<int>("y");
    cell.newVariable<int>("is_occupied", 0);
    cell.newVariable<float>("cell_score", 0.0f);
    cell.newState("default");

    move_submodel.setMovingAgent("agent", {}, {{"active", "default"}}, true);
    move_submodel.setEnvironmentAgent("cell", {}, {{"active", "default"}}, true);

    CUDASimulation sim(model);
    sim.SimulationConfig().steps = 1;

    // Grid initialization:
    // is_occupied is 0 by default.
    // The submodel's internal InitFunction will automatically detect the agents
    // and set is_occupied to 1 at (0,1) and (2,1) before the first move starts.
    auto cell_pop = AgentVector(cell, WIDTH * HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        cell_pop[i].setVariable<int>("x", i / HEIGHT);
        cell_pop[i].setVariable<int>("y", i % HEIGHT);
        cell_pop[i].setVariable<float>("cell_score", 0.0f);
    }
    cell_pop[1 * HEIGHT + 1].setVariable<float>("cell_score", 10.0f);  // Target
    cell_pop[0 * HEIGHT + 1].setVariable<float>("cell_score", 1.0f);  // Agent 0 start score
    cell_pop[2 * HEIGHT + 1].setVariable<float>("cell_score", 1.0f);  // Agent 1 start score
    sim.setPopulationData(cell_pop);

    // Two agents: A at (0,1) with Priority 10, B at (2,1) with Priority 5.
    auto agent_pop = AgentVector(agent, 2);
    agent_pop[0].setVariable<int>("x", 0);
    agent_pop[0].setVariable<int>("y", 1);
    agent_pop[0].setVariable<float>("priority", 10.0f);
    agent_pop[1].setVariable<int>("x", 2);
    agent_pop[1].setVariable<int>("y", 1);
    agent_pop[1].setVariable<float>("priority", 5.0f);
    sim.setPopulationData(agent_pop);

    sim.step();

    sim.getPopulationData(agent_pop);
    // Agent 0 (higher priority) should be at (1,1)
    EXPECT_EQ(agent_pop[0].getVariable<int>("x"), 1);
    EXPECT_EQ(agent_pop[0].getVariable<int>("y"), 1);
    // Agent 1 (lower priority) should have failed and stayed at (2,1)
    EXPECT_EQ(agent_pop[1].getVariable<int>("x"), 2);
    EXPECT_EQ(agent_pop[1].getVariable<int>("y"), 1);

    sim.getPopulationData(cell_pop);
    EXPECT_EQ(cell_pop[1 * HEIGHT + 1].getVariable<int>("is_occupied"), 1);
}

/**
 * Test 4: Resource Memory
 * Verifies that last_resources_x/y are updated when moving to a cell with score > 0.
 */
TEST(SingleAgentDiscreteMovementTest, ResourceMemory) {
    ModelDescription model("parent_model");
    int WIDTH = 3;
    int HEIGHT = 3;

    SingleAgentDiscreteMovement move_submodel;
    auto smd = move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT);
    model.newLayer().addSubModel(smd);

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

    move_submodel.setMovingAgent("agent", {}, {{"active", "default"}}, true);
    move_submodel.setEnvironmentAgent("cell", {}, {{"active", "default"}}, true);

    CUDASimulation sim(model);
    sim.SimulationConfig().steps = 1;

    auto cell_pop = AgentVector(cell, WIDTH * HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        cell_pop[i].setVariable<int>("x", i / HEIGHT);
        cell_pop[i].setVariable<int>("y", i % HEIGHT);
        cell_pop[i].setVariable<float>("cell_score", 0.0f);
    }
    // High score at (1,1)
    cell_pop[1 * HEIGHT + 1].setVariable<float>("cell_score", 10.0f);
    sim.setPopulationData(cell_pop);

    auto agent_pop = AgentVector(agent, 1);
    agent_pop[0].setVariable<int>("x", 0);
    agent_pop[0].setVariable<int>("y", 0);
    agent_pop[0].setVariable<int>("last_resources_x", -1);
    agent_pop[0].setVariable<int>("last_resources_y", -1);
    sim.setPopulationData(agent_pop);

    sim.step();

    sim.getPopulationData(agent_pop);
    // Agent moved to (1,1)
    EXPECT_EQ(agent_pop[0].getVariable<int>("x"), 1);
    EXPECT_EQ(agent_pop[0].getVariable<int>("y"), 1);
    // Resource memory should be updated
    EXPECT_EQ(agent_pop[0].getVariable<int>("last_resources_x"), 1);
    EXPECT_EQ(agent_pop[0].getVariable<int>("last_resources_y"), 1);
}

/**
 * Test 5: Grid Boundaries
 * Verifies that an agent at the corner moves correctly and doesn't crash or go out of bounds.
 */
TEST(SingleAgentDiscreteMovementTest, GridBoundaries) {
    ModelDescription model("parent_model");
    int WIDTH = 2;
    int HEIGHT = 2;

    SingleAgentDiscreteMovement move_submodel;
    auto smd = move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT);
    model.newLayer().addSubModel(smd);

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

    move_submodel.setMovingAgent("agent", {}, {{"active", "default"}}, true);
    move_submodel.setEnvironmentAgent("cell", {}, {{"active", "default"}}, true);

    CUDASimulation sim(model);
    sim.SimulationConfig().steps = 1;

    auto cell_pop = AgentVector(cell, WIDTH * HEIGHT);
    for (int i = 0; i < WIDTH * HEIGHT; ++i) {
        cell_pop[i].setVariable<int>("x", i / HEIGHT);
        cell_pop[i].setVariable<int>("y", i % HEIGHT);
        cell_pop[i].setVariable<float>("cell_score", 0.0f);
    }
    // High score at (1,1)
    cell_pop[1 * HEIGHT + 1].setVariable<float>("cell_score", 10.0f);
    sim.setPopulationData(cell_pop);

    // Agent at (0,1) - on the edge.
    auto agent_pop = AgentVector(agent, 1);
    agent_pop[0].setVariable<int>("x", 0);
    agent_pop[0].setVariable<int>("y", 1);
    sim.setPopulationData(agent_pop);

    sim.step();

    sim.getPopulationData(agent_pop);
    // Should move to (1,1)
    EXPECT_EQ(agent_pop[0].getVariable<int>("x"), 1);
    EXPECT_EQ(agent_pop[0].getVariable<int>("y"), 1);
}

}  // namespace submodels
}  // namespace stockAgent
}  // namespace flamegpu
