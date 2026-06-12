import os
import sys
import pyflamegpu
import pyflamegpu.codegen
import random, math
import json
import shutil

# Ensure CUDA_PATH is set for RTC (Jitify) to find headers
if "CUDA_PATH" not in os.environ:
    os.environ["CUDA_PATH"] = "/usr/local/cuda"

# Grid Size
GRID_WIDTH: pyflamegpu.constant = 256
GRID_HEIGHT: pyflamegpu.constant = 256

# Growback variables
SUGAR_GROWBACK_RATE: pyflamegpu.constant = 1.0
SUGAR_MAX_CAPACITY: pyflamegpu.constant = 7.0

# Output directory
OUTPUT_DIR = "output"

@pyflamegpu.agent_function
def metabolise(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    sugar_level = pyflamegpu.getVariableFloat("sugar_level")
    metabolism = pyflamegpu.getVariableFloat("metabolism")
    harvested = pyflamegpu.getVariableFloat("current_cell_score")

    # Add what we found and subtract what we used
    if harvested > 0.0:
        sugar_level += harvested
    
    sugar_level -= metabolism

    # Death check
    if sugar_level <= 0.0:
        return pyflamegpu.DEAD

    pyflamegpu.setVariableFloat("sugar_level", sugar_level)
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def growback(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    env_sugar_level = pyflamegpu.getVariableFloat("env_sugar_level")
    env_max_sugar_level = pyflamegpu.getVariableFloat("env_max_sugar_level")
    is_occupied = pyflamegpu.getVariableInt("is_occupied")

    if is_occupied:
        # A bug is here, so it has eaten the sugar. Mark as -1 to mirror original.
        env_sugar_level = -1.0
    else:
        # Grow back
        env_sugar_level += SUGAR_GROWBACK_RATE
        if env_sugar_level > env_max_sugar_level:
            env_sugar_level = env_max_sugar_level
        
        # Ensure it's not negative if it was just vacated
        if env_sugar_level < 0.0:
            env_sugar_level = 0.0

    pyflamegpu.setVariableFloat("env_sugar_level", env_sugar_level)
    return pyflamegpu.ALIVE


class step_logger(pyflamegpu.HostFunction):
    def __init__(self, output_dir):
        super().__init__()
        self.output_dir = output_dir
        self.sim = None

    def set_simulation(self, sim):
        self.sim = sim

    def run(self, FLAMEGPU):
        step = FLAMEGPU.getStepCounter()
        bug_count = FLAMEGPU.agent("bug").count()
        print(f"Step {step}: bugs={bug_count}")
        
        if self.sim:
            try:
                self.sim.exportData(os.path.join(self.output_dir, f"step_{step}.xml"))
            except pyflamegpu.FLAMEGPURuntimeException as e:
                if "FileAlreadyExists" in str(e):
                    # We might get this if we already exported step_0.xml manually
                    pass
                else:
                    raise e


def generate_initial_population(cudaSimulation, bug_desc, cell_desc):
    # 1. Generate Sugar Hotspots
    sugar_hotspots = []
    hotspot_area = 0
    target_area = (GRID_WIDTH * GRID_HEIGHT) * 0.6

    while hotspot_area < target_area:
        rad = random.randint(15, 45)
        hs = [random.randint(0, GRID_WIDTH - 1),
                random.randint(0, GRID_HEIGHT - 1),
                rad,
                SUGAR_MAX_CAPACITY]
        sugar_hotspots.append(hs)
        hotspot_area += math.pi * rad * rad

    # 2. Place Bugs randomly
    bug_density = 0.05
    bug_count = int((GRID_WIDTH * GRID_HEIGHT) * bug_density)

    indices = list(range(GRID_WIDTH * GRID_HEIGHT))
    random.shuffle(indices)

    bug_at = [False] * (GRID_WIDTH * GRID_HEIGHT)
    
    # Create Bug Population
    bug_pop = pyflamegpu.AgentVector(bug_desc, bug_count)
    for i in range(bug_count):
        idx = indices[i]
        bug_at[idx] = True
        b = bug_pop[i]
        b.setVariableInt("x", idx // GRID_HEIGHT)
        b.setVariableInt("y", idx % GRID_HEIGHT)
        b.setVariableFloat("sugar_level", random.uniform(10.0, 30.0))
        b.setVariableFloat("metabolism", random.uniform(1.0, 2.5))
    cudaSimulation.setPopulationData(bug_pop)

    # 3. Create Sugar Cells
    cell_pop = pyflamegpu.AgentVector(cell_desc, GRID_WIDTH * GRID_HEIGHT)
    for x in range(GRID_WIDTH):
        for y in range(GRID_HEIGHT):
            idx = x * GRID_HEIGHT + y
            cell = cell_pop[idx]
            cell.setVariableInt("x", x)
            cell.setVariableInt("y", y)
            cell.setVariableInt("is_occupied", 1 if bug_at[idx] else 0)

            max_val = 0.0
            for hs in sugar_hotspots:
                dx = hs[0] - x
                dy = hs[1] - y
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < hs[2]:
                    v = hs[3] * (1.0 - (dist / hs[2]))
                    if v > max_val:
                        max_val = v

            cell.setVariableFloat("env_max_sugar_level", max_val)
            cell.setVariableFloat("env_sugar_level", max_val)
    cudaSimulation.setPopulationData(cell_pop)

if __name__ == "__main__":

    # create the model
    model = pyflamegpu.ModelDescription("Sugarscape")

    # Bug Agent
    bug = model.newAgent("bug")
    bug.newVariableInt("x")
    bug.newVariableInt("y")
    bug.newVariableFloat("sugar_level")
    bug.newVariableFloat("metabolism")
    bug.newVariableFloat("current_cell_score", 0.0)

    # Sugar Cell Agent
    sugar_cell = model.newAgent("sugar_cell")
    sugar_cell.newVariableInt("x")
    sugar_cell.newVariableInt("y")
    sugar_cell.newVariableFloat("env_sugar_level")
    sugar_cell.newVariableFloat("env_max_sugar_level")
    sugar_cell.newVariableInt("is_occupied", 0)

    # Submodel Configuration
    submodel = pyflamegpu.SingleAgentDiscreteMovement(model, GRID_WIDTH, GRID_HEIGHT)

    bug_vars = pyflamegpu.map_string_string()
    bug_vars["x"] = "x"
    bug_vars["y"] = "y"
    bug_vars["current_cell_score"] = "current_cell_score"

    submodel.setMovingAgent("bug",
        bug_vars,
        pyflamegpu.map_string_string(), # Empty state map
    )

    env_vars = pyflamegpu.map_string_string()
    env_vars["x"] = "x"
    env_vars["y"] = "y"
    env_vars["is_occupied"] = "is_occupied"
    env_vars["cell_score"] = "env_sugar_level"

    submodel.setEnvironmentAgent("sugar_cell",
        env_vars,
        pyflamegpu.map_string_string(), # Empty state map
    )

    # RTC Functions
    metabolise_fn = bug.newRTCFunction("metabolise", pyflamegpu.codegen.translate(metabolise))
    metabolise_fn.setAllowAgentDeath(True)
    growback_fn = sugar_cell.newRTCFunction("growback", pyflamegpu.codegen.translate(growback))

    # Layers
    model.newLayer().addSubModel(submodel.getSubModelDescription())
    
    layer2 = model.newLayer()
    layer2.addAgentFunction(metabolise_fn)
    layer2.addAgentFunction(growback_fn)

    # Host Functions
    my_step = step_logger(OUTPUT_DIR)
    model.addStepFunction(my_step)

    # Set up and run the simulation
    cudaSimulation = pyflamegpu.CUDASimulation(model)
    my_step.set_simulation(cudaSimulation)
    
    cudaSimulation.SimulationConfig().steps = 100
    
    # Configure logging
    step_log = pyflamegpu.StepLoggingConfig(model)
    step_log.agent("bug").logCount()
    step_log.agent("bug").logMeanFloat("sugar_level")
    step_log.agent("sugar_cell").logMeanFloat("env_sugar_level")
    cudaSimulation.setStepLog(step_log)
    
    cudaSimulation.initialise(sys.argv)
    
    # Generate initial population if no input file provided
    if not cudaSimulation.SimulationConfig().input_file:
        generate_initial_population(cudaSimulation, bug, sugar_cell)

    # Clean up output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Export initial state (Step 0)
    cudaSimulation.exportData(os.path.join(OUTPUT_DIR, "step_0.xml"))
    
    cudaSimulation.simulate()
    
    # Export log
    cudaSimulation.exportLog(os.path.join(OUTPUT_DIR, "simulation_log.json"), True, True, False, False)

    print(f"Finished pyflamegpu example: Sugarscape model. Data saved to {OUTPUT_DIR}/")
