import pytest
from unittest import TestCase
from pyflamegpu import *

class SingleAgentDiscreteMovementTest(TestCase):
    """
    Test 1: Initialization & Validation
    Verifies that the submodel correctly validates its agent and variable bindings.
    """
    def test_initialization(self):
        model = pyflamegpu.ModelDescription("parent_model")
        move_submodel = pyflamegpu.SingleAgentDiscreteMovement()

        # Should throw if we try to bind before calling addSingleAgentDiscreteMovementSubmodel
        with pytest.raises(pyflamegpu.FLAMEGPURuntimeException) as e:
            move_submodel.setMovingAgent("agent")
        assert e.value.type() == "InvalidSubModel"

        # Initialize the submodel
        move_submodel.addSingleAgentDiscreteMovementSubmodel(model, 10, 10)

        # Setup a valid parent agent for moving
        agent = model.newAgent("agent")
        agent.newVariableInt("x")
        agent.newVariableInt("y")
        agent.newVariableInt("last_x")
        agent.newVariableInt("last_y")
        agent.newVariableInt("last_resources_x")
        agent.newVariableInt("last_resources_y")
        agent.newVariableFloat("current_cell_score")
        agent.newState("default")

        # Setup a valid parent agent for the environment grid
        cell = model.newAgent("cell")
        cell.newVariableInt("x")
        cell.newVariableInt("y")
        cell.newVariableInt("is_occupied")
        cell.newVariableFloat("cell_score")
        cell.newState("default")

        # Bind agents: auto_map=true handles variables with matching names,
        # but we must explicitly map internal "active" state to parent "default" state.
        bug_vars = pyflamegpu.map_string_string()
        bug_states = pyflamegpu.map_string_string()
        bug_states["active"] = "default"
        move_submodel.setMovingAgent("agent", bug_vars, bug_states, True)
        
        env_vars = pyflamegpu.map_string_string()
        env_states = pyflamegpu.map_string_string()
        env_states["active"] = "default"
        move_submodel.setEnvironmentAgent("cell", env_vars, env_states, True)

        # Validation should pass now
        try:
            move_submodel.validate()
        except pyflamegpu.FLAMEGPURuntimeException as e:
            pytest.fail(f"validate() threw {e.type()} unexpectedly: {e.what()}")

        # Check getName and getSubModelDescription
        assert move_submodel.getName() == "SingleAgentDiscreteMovement"
        assert move_submodel.getSubModelDescription() is not None

    """
    Test 2: Basic Greedy Movement
    Verifies that an agent at (0,0) moves to (1,1) if that cell has a higher score.
    """
    def test_simple_move(self):
        model = pyflamegpu.ModelDescription("parent_model")
        WIDTH = 3
        HEIGHT = 3

        move_submodel = pyflamegpu.SingleAgentDiscreteMovement()
        move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT)

        # Submodels must be added to a layer to be executed during the simulation step
        model.newLayer().addSubModel(move_submodel.getSubModelDescription())

        # Define parent agents with necessary variables
        agent = model.newAgent("agent")
        agent.newVariableInt("x")
        agent.newVariableInt("y")
        agent.newVariableInt("last_x", -1)
        agent.newVariableInt("last_y", -1)
        agent.newVariableInt("last_resources_x", -1)
        agent.newVariableInt("last_resources_y", -1)
        agent.newVariableFloat("current_cell_score", 0.0)
        agent.newVariableFloat("priority", 1.0)
        agent.newState("default")

        cell = model.newAgent("cell")
        cell.newVariableInt("x")
        cell.newVariableInt("y")
        cell.newVariableInt("is_occupied", 0)
        cell.newVariableFloat("cell_score", 0.0)
        cell.newState("default")

        # Bind to submodel
        bug_vars = pyflamegpu.map_string_string()
        bug_states = pyflamegpu.map_string_string()
        bug_states["active"] = "default"
        move_submodel.setMovingAgent("agent", bug_vars, bug_states, True)
        
        env_vars = pyflamegpu.map_string_string()
        env_states = pyflamegpu.map_string_string()
        env_states["active"] = "default"
        move_submodel.setEnvironmentAgent("cell", env_vars, env_states, True)

        sim = pyflamegpu.CUDASimulation(model)
        sim.SimulationConfig().steps = 1

        # Initialize the grid: 3x3 cells
        cell_pop = pyflamegpu.AgentVector(cell, WIDTH * HEIGHT)
        for x in range(WIDTH):
            for y in range(HEIGHT):
                c = cell_pop[x * HEIGHT + y]
                c.setVariableInt("x", x)
                c.setVariableInt("y", y)
                c.setVariableFloat("cell_score", 0.0)
        
        # Set a high "reward" at (1, 1)
        cell_pop[1 * HEIGHT + 1].setVariableFloat("cell_score", 10.0)
        sim.setPopulationData(cell_pop)

        # Initialize 1 agent at (0, 0)
        agent_pop = pyflamegpu.AgentVector(agent, 1)
        agent_pop[0].setVariableInt("x", 0)
        agent_pop[0].setVariableInt("y", 0)
        agent_pop[0].setVariableFloat("priority", 1.0)
        sim.setPopulationData(agent_pop)

        # Execute one simulation step
        sim.step()

        # Verify the agent moved to the high-score cell (1, 1)
        sim.getPopulationData(agent_pop)
        assert agent_pop[0].getVariableInt("x") == 1
        assert agent_pop[0].getVariableInt("y") == 1

        # Verify occupancy status was updated
        sim.getPopulationData(cell_pop)
        assert cell_pop[0 * HEIGHT + 0].getVariableInt("is_occupied") == 0  # Left (0,0)
        assert cell_pop[1 * HEIGHT + 1].getVariableInt("is_occupied") == 1  # Entered (1,1)

    """
    Test 3: Collision Avoidance
    Verifies that when two agents try to move to the same cell, only one succeeds.
    """
    def test_collision_avoidance(self):
        model = pyflamegpu.ModelDescription("parent_model")
        WIDTH = 3
        HEIGHT = 3

        move_submodel = pyflamegpu.SingleAgentDiscreteMovement()
        move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT)
        model.newLayer().addSubModel(move_submodel.getSubModelDescription())

        agent = model.newAgent("agent")
        agent.newVariableInt("x")
        agent.newVariableInt("y")
        agent.newVariableInt("last_x", -1)
        agent.newVariableInt("last_y", -1)
        agent.newVariableInt("last_resources_x", -1)
        agent.newVariableInt("last_resources_y", -1)
        agent.newVariableFloat("current_cell_score", 0.0)
        agent.newVariableFloat("priority", 0.0)
        agent.newState("default")

        cell = model.newAgent("cell")
        cell.newVariableInt("x")
        cell.newVariableInt("y")
        cell.newVariableInt("is_occupied", 0)
        cell.newVariableFloat("cell_score", 0.0)
        cell.newState("default")

        bug_vars = pyflamegpu.map_string_string()
        bug_states = pyflamegpu.map_string_string()
        bug_states["active"] = "default"
        move_submodel.setMovingAgent("agent", bug_vars, bug_states, True)
        
        env_vars = pyflamegpu.map_string_string()
        env_states = pyflamegpu.map_string_string()
        env_states["active"] = "default"
        move_submodel.setEnvironmentAgent("cell", env_vars, env_states, True)

        sim = pyflamegpu.CUDASimulation(model)
        sim.SimulationConfig().steps = 1

        # Grid initialization
        cell_pop = pyflamegpu.AgentVector(cell, WIDTH * HEIGHT)
        for i in range(WIDTH * HEIGHT):
            cell_pop[i].setVariableInt("x", i // HEIGHT)
            cell_pop[i].setVariableInt("y", i % HEIGHT)
            cell_pop[i].setVariableFloat("cell_score", 0.0)
        
        cell_pop[1 * HEIGHT + 1].setVariableFloat("cell_score", 10.0)  # Target
        cell_pop[0 * HEIGHT + 1].setVariableFloat("cell_score", 1.0)  # Agent 0 start score
        cell_pop[2 * HEIGHT + 1].setVariableFloat("cell_score", 1.0)  # Agent 1 start score
        sim.setPopulationData(cell_pop)

        # Two agents: A at (0,1) with Priority 10, B at (2,1) with Priority 5.
        agent_pop = pyflamegpu.AgentVector(agent, 2)
        agent_pop[0].setVariableInt("x", 0)
        agent_pop[0].setVariableInt("y", 1)
        agent_pop[0].setVariableFloat("priority", 10.0)
        agent_pop[1].setVariableInt("x", 2)
        agent_pop[1].setVariableInt("y", 1)
        agent_pop[1].setVariableFloat("priority", 5.0)
        sim.setPopulationData(agent_pop)

        sim.step()

        sim.getPopulationData(agent_pop)
        # Agent 0 (higher priority) should be at (1,1)
        assert agent_pop[0].getVariableInt("x") == 1
        assert agent_pop[0].getVariableInt("y") == 1
        # Agent 1 (lower priority) should have failed and stayed at (2,1)
        assert agent_pop[1].getVariableInt("x") == 2
        assert agent_pop[1].getVariableInt("y") == 1

        sim.getPopulationData(cell_pop)
        assert cell_pop[1 * HEIGHT + 1].getVariableInt("is_occupied") == 1

    """
    Test 4: Resource Memory
    Verifies that last_resources_x/y are updated when moving to a cell with score > 0.
    """
    def test_resource_memory(self):
        model = pyflamegpu.ModelDescription("parent_model")
        WIDTH = 3
        HEIGHT = 3

        move_submodel = pyflamegpu.SingleAgentDiscreteMovement()
        move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT)
        model.newLayer().addSubModel(move_submodel.getSubModelDescription())

        agent = model.newAgent("agent")
        agent.newVariableInt("x")
        agent.newVariableInt("y")
        agent.newVariableInt("last_x", -1)
        agent.newVariableInt("last_y", -1)
        agent.newVariableInt("last_resources_x", -1)
        agent.newVariableInt("last_resources_y", -1)
        agent.newVariableFloat("current_cell_score", 0.0)
        agent.newVariableFloat("priority", 1.0)
        agent.newState("default")

        cell = model.newAgent("cell")
        cell.newVariableInt("x")
        cell.newVariableInt("y")
        cell.newVariableInt("is_occupied", 0)
        cell.newVariableFloat("cell_score", 0.0)
        cell.newState("default")

        bug_vars = pyflamegpu.map_string_string()
        bug_states = pyflamegpu.map_string_string()
        bug_states["active"] = "default"
        move_submodel.setMovingAgent("agent", bug_vars, bug_states, True)
        
        env_vars = pyflamegpu.map_string_string()
        env_states = pyflamegpu.map_string_string()
        env_states["active"] = "default"
        move_submodel.setEnvironmentAgent("cell", env_vars, env_states, True)

        sim = pyflamegpu.CUDASimulation(model)
        sim.SimulationConfig().steps = 1

        cell_pop = pyflamegpu.AgentVector(cell, WIDTH * HEIGHT)
        for i in range(WIDTH * HEIGHT):
            cell_pop[i].setVariableInt("x", i // HEIGHT)
            cell_pop[i].setVariableInt("y", i % HEIGHT)
            cell_pop[i].setVariableFloat("cell_score", 0.0)
        
        # High score at (1,1)
        cell_pop[1 * HEIGHT + 1].setVariableFloat("cell_score", 10.0)
        sim.setPopulationData(cell_pop)

        agent_pop = pyflamegpu.AgentVector(agent, 1)
        agent_pop[0].setVariableInt("x", 0)
        agent_pop[0].setVariableInt("y", 0)
        agent_pop[0].setVariableInt("last_resources_x", -1)
        agent_pop[0].setVariableInt("last_resources_y", -1)
        sim.setPopulationData(agent_pop)

        sim.step()

        sim.getPopulationData(agent_pop)
        # Agent moved to (1,1)
        assert agent_pop[0].getVariableInt("x") == 1
        assert agent_pop[0].getVariableInt("y") == 1
        # Resource memory should be updated
        assert agent_pop[0].getVariableInt("last_resources_x") == 1
        assert agent_pop[0].getVariableInt("last_resources_y") == 1

    """
    Test 5: Grid Boundaries
    Verifies that an agent at the corner moves correctly and doesn't crash or go out of bounds.
    """
    def test_grid_boundaries(self):
        model = pyflamegpu.ModelDescription("parent_model")
        WIDTH = 2
        HEIGHT = 2

        move_submodel = pyflamegpu.SingleAgentDiscreteMovement()
        move_submodel.addSingleAgentDiscreteMovementSubmodel(model, WIDTH, HEIGHT)
        model.newLayer().addSubModel(move_submodel.getSubModelDescription())

        agent = model.newAgent("agent")
        agent.newVariableInt("x")
        agent.newVariableInt("y")
        agent.newVariableInt("last_x", -1)
        agent.newVariableInt("last_y", -1)
        agent.newVariableInt("last_resources_x", -1)
        agent.newVariableInt("last_resources_y", -1)
        agent.newVariableFloat("current_cell_score", 0.0)
        agent.newVariableFloat("priority", 1.0)
        agent.newState("default")

        cell = model.newAgent("cell")
        cell.newVariableInt("x")
        cell.newVariableInt("y")
        cell.newVariableInt("is_occupied", 0)
        cell.newVariableFloat("cell_score", 0.0)
        cell.newState("default")

        bug_vars = pyflamegpu.map_string_string()
        bug_states = pyflamegpu.map_string_string()
        bug_states["active"] = "default"
        move_submodel.setMovingAgent("agent", bug_vars, bug_states, True)
        
        env_vars = pyflamegpu.map_string_string()
        env_states = pyflamegpu.map_string_string()
        env_states["active"] = "default"
        move_submodel.setEnvironmentAgent("cell", env_vars, env_states, True)

        sim = pyflamegpu.CUDASimulation(model)
        sim.SimulationConfig().steps = 1

        cell_pop = pyflamegpu.AgentVector(cell, WIDTH * HEIGHT)
        for i in range(WIDTH * HEIGHT):
            cell_pop[i].setVariableInt("x", i // HEIGHT)
            cell_pop[i].setVariableInt("y", i % HEIGHT)
            cell_pop[i].setVariableFloat("cell_score", 0.0)
        
        # High score at (1,1)
        cell_pop[1 * HEIGHT + 1].setVariableFloat("cell_score", 10.0)
        sim.setPopulationData(cell_pop)

        # Agent at (0,1) - on the edge.
        agent_pop = pyflamegpu.AgentVector(agent, 1)
        agent_pop[0].setVariableInt("x", 0)
        agent_pop[0].setVariableInt("y", 1)
        sim.setPopulationData(agent_pop)

        sim.step()

        sim.getPopulationData(agent_pop)
        # Should move to (1,1)
        assert agent_pop[0].getVariableInt("x") == 1
        assert agent_pop[0].getVariableInt("y") == 1

if __name__ == "__main__":
    import unittest
    unittest.main()

if __name__ == "__main__":
    import unittest
    unittest.main()
