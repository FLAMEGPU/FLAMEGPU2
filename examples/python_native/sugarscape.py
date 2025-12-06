# Import pyflamegpu and some other libraries we will use in the tutorial
import pyflamegpu
import pyflamegpu.codegen
import sys, random, math
import matplotlib.pyplot as plt
import importlib
import argparse
import time
import keyboard 

import common
import sugarscape_movement
import sugarscape_common


# Environment size
D = 256

# Model parameters
S_MAX = 4              # S_{max}, env sugar maximum level
S_ALPHA = 1            # /alpha, sugar growback rate

# Parameters for initialisation of the model (see `Creation of Initial Population Data`)
H_MIN = 20             # H_{min}, hotspot minimum distance
W_MIN = 5              # W_{min}, agent wealth minimum
W_MAX = 25             # W_{max}, agent wealth maximum
M_MIN = 1              # M_{min}, agent metabolism minimum
M_MAX = 4              # M_{max}, agent metabolism maximum
P_OCCUPATION = 0.17    # /rho, probability of sugarscape agent occupation

DEFAULT_SIMULATION_STEPS = 100 # number of simulation steps

def create_model():
    model = pyflamegpu.ModelDescription("sugarscape")
    return model

def define_environment(model):
    env = model.Environment()
    env.newPropertyInt("sugar_growback_rate", S_ALPHA)
    env.newPropertyInt("sugar_max_capacity", S_MAX)
    
    # agent status
    env.newPropertyInt("agent_status_unoccupied", sugarscape_common.AGENT_STATUS_UNOCCUPIED)
    env.newPropertyInt("agent_status_occupied", sugarscape_common.AGENT_STATUS_OCCUPIED)
    env.newPropertyInt("agent_status_movement_requested", sugarscape_common.AGENT_STATUS_MOVEMENT_REQUESTED)
    env.newPropertyInt("agent_status_movement_unresolved", sugarscape_common.AGENT_STATUS_MOVEMENT_UNRESOLVED)

def define_agents(model):
    agent = model.newAgent("agent")
    
    # Generic variables
    agent.newVariableInt("x")
    agent.newVariableInt("y")
    agent.newVariableInt("agent_id")
    agent.newVariableInt("status")
    # Sugarscape agent specific variables
    agent.newVariableInt("sugar_level")
    agent.newVariableInt("metabolism")
    # Environment cell specific variables
    agent.newVariableInt("env_sugar_level")
    agent.newVariableInt("env_max_sugar_level")
    if pyflamegpu.VISUALISATION:
        agent.newVariableFloat("x_vis")   
        agent.newVariableFloat("y_vis")  
        agent.newVariableFloat("z_vis")   

    # Metabolise, growback and move functions
    metabolise_fn = agent.newRTCFunction("metabolise", pyflamegpu.codegen.translate(metabolise))
    growback_fn = agent.newRTCFunction("growback", pyflamegpu.codegen.translate(growback))
    move_fn = agent.newRTCFunction("move", pyflamegpu.codegen.translate(move))

      

def define_dependencies(model, movement_submodel, plotting):
    """
    Control Flow
    """
    agent = model.Agent("agent")
    move_fn = agent.Function("move")
    growback_fn = agent.Function("growback")
    metabolise_fn = agent.Function("metabolise")

    # specify the dependencies
    model.addExecutionRoot(movement_submodel)
    move_fn.dependsOn(growback_fn)
    growback_fn.dependsOn(metabolise_fn)
    metabolise_fn.dependsOn(movement_submodel)
    model.generateLayers()

    # add plotting host function
    if plotting:
        model.addStepFunction(plot_pop())
        model.addStepFunction(plot_sugar())
    
@pyflamegpu.agent_function
def metabolise(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    agent_sugar_level = pyflamegpu.getVariableInt("sugar_level")
    env_sugar_level = pyflamegpu.getVariableInt("env_sugar_level")
    status = pyflamegpu.getVariableInt("status")
    metabolism = pyflamegpu.getVariableInt("metabolism")

    # metabolise if occupied
    if (status == pyflamegpu.environment.getPropertyInt("agent_status_occupied")):
        
        # store any sugar present in the cell
        if (env_sugar_level > 0) :
            agent_sugar_level += env_sugar_level
            # Occupied cells are marked as -1 sugar.
            env_sugar_level = 0;

        # metabolise
        agent_sugar_level -= metabolism

        # check if agent dies
        if (agent_sugar_level <= 0) :
            status = pyflamegpu.environment.getPropertyInt("agent_status_unoccupied");
            # env_sugar_level = 0;
            pyflamegpu.setVariableInt("agent_id", -1);
            pyflamegpu.setVariableInt("metabolism", 0);
            pyflamegpu.setVariableInt("status", status);

    pyflamegpu.setVariableInt("sugar_level", agent_sugar_level);
    pyflamegpu.setVariableInt("env_sugar_level", env_sugar_level);

    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def growback(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    env_sugar_level = pyflamegpu.getVariableInt("env_sugar_level");
    status = pyflamegpu.getVariableInt("status");
    env_max_sugar_level = pyflamegpu.getVariableInt("env_max_sugar_level");
    
    # growback if unoccupied
    if (status == pyflamegpu.environment.getPropertyInt("agent_status_unoccupied")) :
        env_sugar_level += pyflamegpu.environment.getPropertyInt("sugar_growback_rate")
        if (env_sugar_level > env_max_sugar_level) :
            env_sugar_level = env_max_sugar_level
            

    pyflamegpu.setVariableInt("env_sugar_level", env_sugar_level);

    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def move(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageNone):
    status = pyflamegpu.getVariableInt("status");
        
    # set all active agents to unresolved as they may now want to move
    if (status == pyflamegpu.environment.getPropertyInt("agent_status_occupied")) :
        status = pyflamegpu.environment.getPropertyInt("agent_status_movement_unresolved")

    pyflamegpu.setVariableInt("status", status);

    return pyflamegpu.ALIVE

def initialise_population(model):
    # Create some randomised sugar areas for the map
    sugar_hotspots = []
    # Calculate the number of hotspots (average denisty based on that of the original model)
    num_hotspots = int((2 * D * D) / (49*49))
    for i in range(num_hotspots):
        attempts = 0
        while True:
            hs = [random.randint(0, D - 1), random.randint(0, D - 1)]
            # Check if the new hotspot is at least H_MIN units away from existing ones
            if all(math.sqrt((hx - hs[0]) ** 2 + (hy - hs[1]) ** 2) >= H_MIN for hx, hy in sugar_hotspots):
                sugar_hotspots.append(hs)
                break  # Valid position found, move to the next hotspot
            # limit attemtps to create a position
            attempts += 1
            if attempts == 100:
                print(f"Warning: Maximum attempts reached creating unique location for sugar hotspot {i}.")
                break

    # Initialize agent population
    init_pop = pyflamegpu.AgentVector(model.Agent("agent"), D * D)
    # Create some random distributions
    normal = random.uniform 
    agent_sugar_dist = lambda: random.randint(W_MIN, W_MAX)
    agent_metabolism_dist = lambda: random.randint(M_MIN, M_MAX)
    # loop to create agents / cells
    i = 0
    for x in range(D):
        for y in range(D):
            instance = init_pop[i]
            i += 1
            instance.setVariableInt("x", x)
            instance.setVariableInt("y", y)
            if pyflamegpu.VISUALISATION:
                instance.setVariableFloat("x_vis", float(x))
                instance.setVariableFloat("y_vis", float(y))
                instance.setVariableFloat("z_vis", 0)
            # chance of cell holding an agent
            if normal(0, 1) < P_OCCUPATION:
                instance.setVariableInt("agent_id", i)
                instance.setVariableInt("status", sugarscape_common.AGENT_STATUS_OCCUPIED)
                instance.setVariableInt("sugar_level", agent_sugar_dist())
                instance.setVariableInt("metabolism", agent_metabolism_dist())
            else:
                instance.setVariableInt("agent_id", -1)
                instance.setVariableInt("status", sugarscape_common.AGENT_STATUS_UNOCCUPIED)
                instance.setVariableInt("sugar_level", 0)
                instance.setVariableInt("metabolism", 0)
    
            # Environment-specific variable
            env_sugar_lvl = 0
            hotspot_core_size = 5
    
            for hs in sugar_hotspots:
                hs_x, hs_y = hs  # Unpack hotspot coordinates
                # Compute Euclidean distance to hotspot
                hs_dist = math.sqrt((hs_x - x) ** 2 + (hs_y - y) ** 2)
                # Compute environmental sugar level
                env_sugar_lvl += max(0, 4 - min(4, math.floor(hs_dist / hotspot_core_size)))
            env_sugar_lvl = min(env_sugar_lvl, S_MAX)
    
            # Set environmental sugar variables
            instance.setVariableInt("env_sugar_level", env_sugar_lvl)
            instance.setVariableInt("env_max_sugar_level", env_sugar_lvl) # All cells begin at their local max sugar
    
    return init_pop

def initialise_simulation(plotting):
    # create the model and define the environmet, agents and dependancies
    model = create_model()
    define_environment(model)
    define_agents(model)
    submodel = sugarscape_movement.add_movement_submodel(model, D)  # Create a movement submodel to handle agent communication, negotiation and movement
    define_dependencies(model, submodel, plotting)
    
    # create an initial population of agents
    init_pop = initialise_population(model)

    # create a CUDA simulation and set the initial data
    cuda_simulation = pyflamegpu.CUDASimulation(model)
    cuda_simulation.setPopulationData(init_pop);
    
    return cuda_simulation


class plot_pop(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        self.plotter = common.LinePlotter("Population count", "Iteration", "Count")

    def run(self, FLAMEGPU):
        # Retrieve the host agent tools for agent sheep in the default state
        agents = FLAMEGPU.agent("agent");
        count = agents.countInt("status", sugarscape_common.AGENT_STATUS_MOVEMENT_UNRESOLVED)   # all occupied will be set to unresolved for start of next iteration
        self.plotter.add_data(count)

class plot_sugar(pyflamegpu.HostFunction):
    def __init__(self):
        super().__init__()
        self.plotter = common.LinePlotter("Sugar Amount (Total)", "Iteration", "Sugar")

    def run(self, FLAMEGPU):
        # Retrieve the host agent tools for agent sheep in the default state
        agents = FLAMEGPU.agent("agent");
        sugar = agents.sumInt("sugar_level")
        self.plotter.add_data(sugar)


def execute_simulation(plotting, args):
    cuda_simulation = initialise_simulation(plotting) 

    # Create Visualisation
    if pyflamegpu.VISUALISATION:
        visualisation = cuda_simulation.getVisualisation()
        #visualisation config
        visualisation.setClearColor(255, 255, 255)
        visualisation.setInitialCameraLocation(D / 2.0, D / 2.0, 225.0)
        visualisation.setInitialCameraTarget(D / 2.0, D / 2.0, 0.0)
        visualisation.setCameraSpeed(0.001 * D)
        visualisation.setOrthographic(True)
        visualisation.setOrthographicZoomModifier(0.365)
        visualisation.setViewClips(0.1, 5000)
        visualisation.setSimulationSpeed(0 if args.step_delay==0 else int(1.0/args.step_delay))
        # visualisation of agent
        agent = visualisation.addAgent("agent")
        agent.setXVariable("x_vis")
        agent.setYVariable("y_vis")
        agent.setModelScale(1.0)
        agent.setModel(pyflamegpu.CUBE)
        cell_colours = pyflamegpu.iDiscreteColor("env_sugar_level", pyflamegpu.Viridis(S_MAX+1), pyflamegpu.Color("#f00"))
        agent.setColor(cell_colours)
        # Visualisation UI
        ui = visualisation.newUIPanel("Settings")
        ui.newSection("Model Parameters")
        ui.newEnvironmentPropertySliderInt("sugar_growback_rate", 0, 5)
        ui.newEnvironmentPropertySliderInt("sugar_max_capacity", 0, S_MAX)
        visualisation.activate()
        # Execute simulation and visualise
        cuda_simulation.SimulationConfig().steps = args.steps
        cuda_simulation.simulate()
        if pyflamegpu.VISUALISATION:
            visualisation.join()
    else:
        for i in range(args.steps):
            cuda_simulation.step()
            if keyboard.is_pressed("q"):
                return
            time.sleep(args.step_delay)
        print("Simulation complete (press q to exit)") 

        while True: 
            if keyboard.is_pressed('q'): 
                return  


if __name__ == '__main__':  
    parser = argparse.ArgumentParser(description="Simulation config")
    parser.add_argument("--no-plotting", action="store_true", help="Disable plotting (default False)")
    parser.add_argument("--no-vis", action="store_true", help="Disable visualisation if it is enabled (default False)")
    parser.add_argument("--step-delay", type=float, default=0.2, help="Step delay in seconds (default is 0.5)")
    parser.add_argument("-s", "--steps", type=int, default=DEFAULT_SIMULATION_STEPS, help="Number of simulation steps (visualisation will loop i.e. '-s 0'")
    args, _ = parser.parse_known_args()
    if args.no_vis:
        pyflamegpu.VISUALISATION = False
    if pyflamegpu.VISUALISATION:
        args.steps = 0

    execute_simulation(plotting=not(args.no_plotting), args=args) 

    pyflamegpu.cleanup()

