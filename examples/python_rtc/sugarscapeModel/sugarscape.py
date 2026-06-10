import os
print("DEBUG: Running sugarscape.py")
import pyflamegpu
import pyflamegpu.codegen
import csv
import random, math

# Ensure CUDA_PATH is set for RTC (Jitify) to find headers
if "CUDA_PATH" not in os.environ:
    os.environ["CUDA_PATH"] = "/usr/local/cuda"

GRID_WIDTH: pyflamegpu.constant = 256
GRID_HEIGHT: pyflamegpu.constant = 256

# Growback variables
SUGAR_GROWBACK_RATE: pyflamegpu.constant = 1.0
SUGAR_MAX_CAPACITY: pyflamegpu.constant = 7.0

@pyflamegpu.agent_function
def metabolise(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    sugar = pyflamegpu.getVariableFloat("sugar")
    metabolism = pyflamegpu.getVariableFloat("metabolism")
    harvested = pyflamegpu.getVariableFloat("current_cell_score")

    # Add what we found and subtract what we used
    sugar += harvested;
    sugar -= metabolism;

    # Death check
    if (sugar <= 0.0 ):
        return pyflamegpu.DEAD

    pyflamegpu.setVariableFloat("sugar", sugar)
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def growback(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    sugar = pyflamegpu.getVariableFloat("sugar")
    max_sugar = pyflamegpu.getVariableFloat("max_sugar")
    is_occupied = pyflamegpu.getVariableInt("is_occupied")


    if (is_occupied):
        # A bug is here, so it has eaten the sugar
        sugar = 0.0
    else:
        # Grow back
        sugar += SUGAR_GROWBACK_RATE
        if (sugar > max_sugar):
            sugar = max_sugar

    pyflamegpu.setVariableFloat("sugar", sugar)
    return pyflamegpu.ALIVE


class step_logger(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        self.bug_log_file = None
        self.cell_log_file = None
        self.bug_writer = None
        self.cell_writer = None

    def run(self, FLAMEGPU):
        step = FLAMEGPU.getStepCounter()
        bug_count = FLAMEGPU.agent("bug").count()
        print(f"Step {step}: bugs={bug_count}")

        # 1. Log bugs every step
        if step == 0:
            self.bug_log_file = open("bugs_log.csv", "w", newline='')
            self.bug_writer = csv.writer(self.bug_log_file)
            self.bug_writer.writerow(["step", "x", "y", "sugar", "metabolism"])

        bug_pop = FLAMEGPU.agent("bug").getPopulationData()
        for bug in bug_pop:
            self.bug_writer.writerow([
                step,
                bug.getVariableInt("x"),
                bug.getVariableInt("y"),
                bug.getVariableFloat("sugar"),
                bug.getVariableFloat("metabolism")
            ])
        self.bug_log_file.flush()

        # 2. Log cells every 10 steps
        if step == 0:
            self.cell_log_file = open("cells_log.csv", "w", newline='')
            self.cell_writer = csv.writer(self.cell_log_file)
            self.cell_writer.writerow(["step", "x", "y", "sugar", "max_sugar"])

        if step % 10 == 0:
            cell_pop = FLAMEGPU.agent("sugar_cell").getPopulationData()
            for cell in cell_pop:
                self.cell_writer.writerow([
                    step,
                    cell.getVariableInt("x"),
                    cell.getVariableInt("y"),
                    cell.getVariableFloat("sugar"),
                    cell.getVariableFloat("max_sugar")
                ])
            self.cell_log_file.flush()


class random_initialisation(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()

    def run(self, FLAMEGPU):
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
        for i in range(bug_count):
            idx = indices[i]
            bug_at[idx] = True

            b = FLAMEGPU.agent("bug").newAgent()
            b.setVariableInt("x", idx // GRID_HEIGHT)
            b.setVariableInt("y", idx % GRID_HEIGHT)
            b.setVariableFloat("sugar", random.uniform(10.0, 30.0))
            b.setVariableFloat("metabolism", random.uniform(1.0, 2.5))

        # 3. Create Sugar Cells
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                idx = x * GRID_HEIGHT + y
                cell = FLAMEGPU.agent("sugar_cell").newAgent()
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

                cell.setVariableFloat("max_sugar", max_val)
                cell.setVariableFloat("sugar", max_val)

if __name__ == "__main__":

    # create the model and define the environmet, agents and dependancies
    model = pyflamegpu.ModelDescription("sugarscape")

    bug = model.newAgent("bug")

    bug.newVariableInt("x")
    bug.newVariableInt("y")
    bug.newVariableFloat("sugar")
    bug.newVariableFloat("metabolism")
    bug.newVariableFloat("current_cell_score")

    sugar_cell = model.newAgent("sugar_cell")

    sugar_cell.newVariableInt("x")
    sugar_cell.newVariableInt("y")
    sugar_cell.newVariableFloat("sugar")
    sugar_cell.newVariableFloat("max_sugar")
    sugar_cell.newVariableInt("is_occupied")

    submodel = pyflamegpu.SingleAgentDiscreteMovement(model, GRID_HEIGHT, GRID_WIDTH)

    bug_vars = pyflamegpu.map_string_string()
    bug_vars["x"] = "x"
    bug_vars["y"] = "y"
    bug_vars["current_cell_score"] = "current_cell_score"

    submodel.setMovingAgent("bug",
        bug_vars,
        pyflamegpu.map_string_string(), # Empty state map
        True # auto_map=True will map "default" to "default"
    )

    env_vars = pyflamegpu.map_string_string()
    env_vars["x"] = "x"
    env_vars["y"] = "y"
    env_vars["is_occupied"] = "is_occupied"
    env_vars["cell_score"] = "sugar"

    submodel.setEnvironmentAgent("sugar_cell",
        env_vars,
        pyflamegpu.map_string_string(), # Empty state map
        True # auto_map=True
    )

    metabolise_fn = bug.newRTCFunction("metabolise", pyflamegpu.codegen.translate(metabolise))
    metabolise_fn.setAllowAgentDeath(True)
    growback_fn = sugar_cell.newRTCFunction("growback", pyflamegpu.codegen.translate(growback))

    layer1 = model.newLayer()
    layer1.addAgentFunction(growback_fn)
    layer2 = model.newLayer()
    layer2.addAgentFunction(metabolise_fn)
    layer3 = model.newLayer()
    layer3.addSubModel(submodel.getSubModelDescription())

    my_step = step_logger()
    model.addStepFunction(my_step)

    # Register the random initialisation
    model.addInitFunction(random_initialisation())

    # Set up and run the simulation
    cudaSimulation = pyflamegpu.CUDASimulation(model)
    cudaSimulation.SimulationConfig().steps = 100
    cudaSimulation.simulate()

    print("Starting pyflamegpu example: Sugarscape model")