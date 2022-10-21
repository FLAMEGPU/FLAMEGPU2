from textwrap import indent
from pyflamegpu import *
import pyflamegpu.codegen
import sys, random, math

"""
  FLAME GPU 2 implementation of Boids in python 3, using python to define agent functions rather than CUDA C++ RTC.
  This is based on the FLAME GPU 1 implementation, but with dynamic generation of agents.
  Agents are also wrapped within the environment bounds and use key framing to show the animation of flapping wings.
"""

def vec3Mult(x, y, z, multiplier):
    """
      Multiply a vector by a scalar value in-place
      @param x x component of the vector
      @param y y component of the vector
      @param z z component of the vector
      @param multiplier scalar value to multiply by
    """
    x *= multiplier
    y *= multiplier
    z *= multiplier
    
def vec3Div(x, y, z, divisor):
    """
      Divide a vector by a scalar value in-place
      @param x x component of the vector
      @param y y component of the vector
      @param z z component of the vector
      @param divisor scalar value to divide by
    """
    x /= divisor
    y /= divisor
    z /= divisor

def vec3Normalize(x, y, z):
    """
      Normalize a 3 component vector in-place
      @param x x component of the vector
      @param y y component of the vector
      @param z z component of the vector
    """
    # Get the length (dont call the device vec3Length func)
    length = math.sqrt(x * x + y * y + z * z)
    vec3Div(x, y, z, length)

# pyflamegpu agent and device functions

@pyflamegpu.device_function
def vec3Length(x: float, y: float, z: float) -> float :
    """
    Utility function to calculate length of a vector
    @param x x component of the vector
    @param y y component of the vector
    @param z z component of the vector
    """
    return math.sqrtf(x * x + y * y + z * z)

@pyflamegpu.device_function
def clamp(v : float, min: float, max: float) -> float:
    """
    Utility function to clamp a value between a min and max
    @param v value to clamp
    @param min the minimum value of range
    @param max the maximum value of range
    """
    v = min if v < min else v
    v = max if v > max else v
    return v


@pyflamegpu.agent_function
def outputdata(message_in: pyflamegpu.MessageNone, message_out: pyflamegpu.MessageSpatial3D):
    """
    Pure python version of output agent function for Boid agents, which outputs a spatial message
    """
    # Output each agents publicly visible properties.
    message_out.setVariableInt("id", pyflamegpu.getID())
    message_out.setVariableFloat("x", pyflamegpu.getVariableFloat("x"))
    message_out.setVariableFloat("y", pyflamegpu.getVariableFloat("y"))
    message_out.setVariableFloat("z", pyflamegpu.getVariableFloat("z"))
    message_out.setVariableFloat("fx", pyflamegpu.getVariableFloat("fx"))
    message_out.setVariableFloat("fy", pyflamegpu.getVariableFloat("fy"))
    message_out.setVariableFloat("fz", pyflamegpu.getVariableFloat("fz"))
    return pyflamegpu.ALIVE

@pyflamegpu.agent_function
def inputdata(message_in: pyflamegpu.MessageSpatial3D, message_out: pyflamegpu.MessageNone):
    """
    Pure python version of inputdata agent function for Boid agents, which reads data from neighboring Boid agents, to perform the boid flocking model.
    """
    # Agent properties in local register
    id = pyflamegpu.getID()
    # Agent position
    agent_x = pyflamegpu.getVariableFloat("x")
    agent_y = pyflamegpu.getVariableFloat("y")
    agent_z = pyflamegpu.getVariableFloat("z")
    #/ Agent velocity
    agent_fx = pyflamegpu.getVariableFloat("fx")
    agent_fy = pyflamegpu.getVariableFloat("fy")
    agent_fz = pyflamegpu.getVariableFloat("fz")

    # Boids percieved centerexit()
    perceived_centre_x = 0.0
    perceived_centre_y = 0.0
    perceived_centre_z = 0.0
    perceived_count = 0

    # Boids global velocity matching
    global_velocity_x = 0.0
    global_velocity_y = 0.0
    global_velocity_z = 0.0

    # Total change in velocity
    velocity_change_x = 0.0
    velocity_change_y = 0.0
    velocity_change_z = 0.0

    INTERACTION_RADIUS = pyflamegpu.environment.getPropertyFloat("INTERACTION_RADIUS")
    SEPARATION_RADIUS = pyflamegpu.environment.getPropertyFloat("SEPARATION_RADIUS")
    # Iterate location messages, accumulating relevant data and counts.
    for message in message_in.wrap(agent_x, agent_y, agent_z) :
        # Ignore self messages.
        if message.getVariableInt("id") != id :
            # Get the message location and velocity.
            message_x = message.getVirtualX()
            message_y = message.getVirtualY()
            message_z = message.getVirtualZ()

            # Check interaction radius
            separation = vec3Length(agent_x - message_x, agent_y - message_y, agent_z - message_z)

            if separation < INTERACTION_RADIUS :
                # Update the perceived centre
                perceived_centre_x += message_x
                perceived_centre_y += message_y
                perceived_centre_z += message_z
                perceived_count += 1

                # Update perceived velocity matching
                message_fx = message.getVariableFloat("fx")
                message_fy = message.getVariableFloat("fy")
                message_fz = message.getVariableFloat("fz")
                global_velocity_x += message_fx
                global_velocity_y += message_fy
                global_velocity_z += message_fz

                # Update collision centre
                if separation < SEPARATION_RADIUS :  # dependant on model size
                    # Rule 3) Avoid other nearby boids (Separation)
                    normalizedSeparation = (separation / SEPARATION_RADIUS)
                    invNormSep = (float(1.0) - normalizedSeparation)
                    invSqSep = invNormSep * invNormSep

                    collisionScale = pyflamegpu.environment.getPropertyFloat("COLLISION_SCALE")
                    velocity_change_x += collisionScale * (agent_x - message_x) * invSqSep
                    velocity_change_y += collisionScale * (agent_y - message_y) * invSqSep
                    velocity_change_z += collisionScale * (agent_z - message_z) * invSqSep


    if (perceived_count) :
        # Divide positions/velocities by relevant counts.
        perceived_centre_x /= perceived_count
        perceived_centre_y /= perceived_count
        perceived_centre_z /= perceived_count
        global_velocity_x /= perceived_count
        global_velocity_y /= perceived_count
        global_velocity_z /= perceived_count

        # Rule 1) Steer towards perceived centre of flock (Cohesion)
        steer_velocity_x = 0.0
        steer_velocity_y = 0.0
        steer_velocity_z = 0.0

        STEER_SCALE = pyflamegpu.environment.getPropertyFloat("STEER_SCALE")
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE
        steer_velocity_z = (perceived_centre_z - agent_z) * STEER_SCALE

        velocity_change_x += steer_velocity_x
        velocity_change_y += steer_velocity_y
        velocity_change_z += steer_velocity_z

        # Rule 2) Match neighbours speeds (Alignment)
        match_velocity_x = 0.0
        match_velocity_y = 0.0
        match_velocity_z = 0.0

        MATCH_SCALE = pyflamegpu.environment.getPropertyFloat("MATCH_SCALE")
        match_velocity_x = global_velocity_x * MATCH_SCALE
        match_velocity_y = global_velocity_y * MATCH_SCALE
        match_velocity_z = global_velocity_z * MATCH_SCALE

        velocity_change_x += match_velocity_x - agent_fx
        velocity_change_y += match_velocity_y - agent_fy
        velocity_change_z += match_velocity_z - agent_fz


    # Global scale of velocity change
    GLOBAL_SCALE = pyflamegpu.environment.getPropertyFloat("GLOBAL_SCALE")
    velocity_change_x *= GLOBAL_SCALE
    velocity_change_y *= GLOBAL_SCALE
    velocity_change_z *= GLOBAL_SCALE

    # Update agent velocity
    agent_fx += velocity_change_x
    agent_fy += velocity_change_y
    agent_fz += velocity_change_z

    # Bound velocity
    agent_fscale = vec3Length(agent_fx, agent_fy, agent_fz)
    if agent_fscale > 1 : 
        agent_fx /=  agent_fscale
        agent_fy /=  agent_fscale
        agent_fz /=  agent_fscale
    

    minSpeed = float(0.5)
    if agent_fscale < minSpeed :
        # Normalise
        agent_fx /= agent_fscale
        agent_fy /= agent_fscale
        agent_fz /= agent_fscale

        # Scale to min
        agent_fx *= minSpeed
        agent_fy *= minSpeed
        agent_fz *= minSpeed


    # Apply the velocity
    TIME_SCALE = pyflamegpu.environment.getPropertyFloat("TIME_SCALE")
    agent_x += agent_fx * TIME_SCALE
    agent_y += agent_fy * TIME_SCALE
    agent_z += agent_fz * TIME_SCALE


    # Wrap positions
    minPosition = pyflamegpu.environment.getPropertyFloat("MIN_POSITION")
    maxPosition = pyflamegpu.environment.getPropertyFloat("MAX_POSITION")
    width = maxPosition-minPosition
    if (agent_x < minPosition) :
        agent_x += width
    if (agent_y < minPosition) :
        agent_y += width
    if (agent_z < minPosition) :
        agent_z += width

    if (agent_x > maxPosition) :
        agent_x -= width
    if (agent_y > maxPosition) :
        agent_y -= width
    if (agent_z > maxPosition) :
        agent_z -= width

    # Update wing speed and animation position
    agent_fscale = vec3Length(agent_fx, agent_fy, agent_fz)
    wing_position = pyflamegpu.getVariableFloat("wing_position") + agent_fscale*GLOBAL_SCALE

    # Update global agent memory.
    pyflamegpu.setVariableFloat("x", agent_x)
    pyflamegpu.setVariableFloat("y", agent_y)
    pyflamegpu.setVariableFloat("z", agent_z)

    pyflamegpu.setVariableFloat("fx", agent_fx)
    pyflamegpu.setVariableFloat("fy", agent_fy)
    pyflamegpu.setVariableFloat("fz", agent_fz)

    pyflamegpu.setVariableFloat("wing_position", wing_position)
    pyflamegpu.setVariableFloat("wing_animation", math.sinf(wing_position))

    return pyflamegpu.ALIVE


model = pyflamegpu.ModelDescription("Boids_BruteForce")


# GLOBALS

env = model.Environment()
# Population size to generate, if no agents are loaded from disk
env.newPropertyUInt("POPULATION_TO_GENERATE", 100000)

# Environment Bounds
env.newPropertyFloat("MIN_POSITION", -0.5)
env.newPropertyFloat("MAX_POSITION", +0.5)

# Initialisation parameter(s)
env.newPropertyFloat("MAX_INITIAL_SPEED", 1.0)
env.newPropertyFloat("MIN_INITIAL_SPEED", 0.1)

# Interaction radius
env.newPropertyFloat("INTERACTION_RADIUS", 0.05)
env.newPropertyFloat("SEPARATION_RADIUS", 0.01)

# Global Scalers
env.newPropertyFloat("TIME_SCALE", 0.001)
env.newPropertyFloat("GLOBAL_SCALE", 0.25)

# Rule scalers
env.newPropertyFloat("STEER_SCALE", 0.055)
env.newPropertyFloat("COLLISION_SCALE", 10.0)
env.newPropertyFloat("MATCH_SCALE", 0.015)


# Location message
message = model.newMessageSpatial3D("location")
# Set the range and bounds.
message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"))
message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"))
message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"))
# A message to hold the location of an agent.
message.newVariableID("id")
# X Y Z are implicit.
# message.newVariable<float>("x")
# message.newVariable<float>("y")
# message.newVariable<float>("z")
message.newVariableFloat("fx")
message.newVariableFloat("fy")
message.newVariableFloat("fz")
    

# Boid agent
agent = model.newAgent("Boid")
agent.newVariableFloat("x")
agent.newVariableFloat("y")
agent.newVariableFloat("z")
agent.newVariableFloat("fx")
agent.newVariableFloat("fy")
agent.newVariableFloat("fz")
agent.newVariableFloat("wing_position")
agent.newVariableFloat("wing_animation")
outputdata_translated = pyflamegpu.codegen.translate(outputdata)
inputdata_translated = pyflamegpu.codegen.translate(inputdata)
agent.newRTCFunction("outputdata", outputdata_translated).setMessageOutput("location")
agent.newRTCFunction("inputdata", inputdata_translated).setMessageInput("location")


# Control flow
    
# Layer #1
model.newLayer().addAgentFunction("Boid", "outputdata")
# Layer #2
model.newLayer().addAgentFunction("Boid", "inputdata")

# Create Model Runner
cudaSimulation = pyflamegpu.CUDASimulation(model)

# Create Visualisation

if pyflamegpu.VISUALISATION:
    visualisation = cudaSimulation.getVisualisation()
    # Configure vis
    visualisation.setClearColor(255, 255, 255)
    envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION")
    INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 3.0
    visualisation.setInitialCameraLocation(0, 0, INIT_CAM)
    visualisation.setCameraSpeed(0.001 * envWidth)
    visualisation.setViewClips(0.00001, 50)
    boid_agt = visualisation.addAgent("Boid")
    # Position vars are named x, y, z so they are used by default
    boid_agt.setForwardXVariable("fx")
    boid_agt.setForwardYVariable("fy")
    boid_agt.setForwardZVariable("fz")
    #boid_agt.setModel(pyflamegpu.ARROWHEAD) # Alternative simple 3D model for very large pop sizes
    boid_agt.setKeyFrameModel("./model/bird_a.obj", "./model/bird_b.obj", "wing_animation")
    boid_agt.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS") /2.0)
    boid_agt.setColor(pyflamegpu.RED);
    # Visualisation UI
    ui = visualisation.newUIPanel("Settings")
    ui.newSection("Model Parameters")
    ui.newEnvironmentPropertySliderFloat("TIME_SCALE", 0.00001, 0.01)
    ui.newEnvironmentPropertySliderFloat("GLOBAL_SCALE", 0.05, 0.5)
    ui.newEnvironmentPropertySliderFloat("SEPARATION_RADIUS", 0.01, env.getPropertyFloat("INTERACTION_RADIUS"))
    ui.newEnvironmentPropertySliderFloat("STEER_SCALE", 0.00, 1.0)
    ui.newEnvironmentPropertySliderFloat("COLLISION_SCALE", 0.00, 100.0)
    ui.newEnvironmentPropertySliderFloat("MATCH_SCALE", 0.00, 0.10)
    
    visualisation.activate()

# Initialise Model

cudaSimulation.initialise(sys.argv)

# If no xml model file was is provided, generate a population.
if not cudaSimulation.SimulationConfig().input_file:
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
    random.seed(cudaSimulation.SimulationConfig().random_seed)
    min_pos = env.getPropertyFloat("MIN_POSITION")
    max_pos = env.getPropertyFloat("MAX_POSITION")
    min_speed = env.getPropertyFloat("MIN_INITIAL_SPEED")
    max_speed = env.getPropertyFloat("MAX_INITIAL_SPEED")
    populationSize = env.getPropertyUInt("POPULATION_TO_GENERATE")
    population = pyflamegpu.AgentVector(model.Agent("Boid"), populationSize)
    for i in range(populationSize):
        instance = population[i]

        # Agent position in space
        instance.setVariableFloat("x", random.uniform(min_pos, max_pos))
        instance.setVariableFloat("y", random.uniform(min_pos, max_pos))
        instance.setVariableFloat("z", random.uniform(min_pos, max_pos))

        # Generate a random velocity direction
        fx = random.uniform(-1, 1)
        fy = random.uniform(-1, 1)
        fz = random.uniform(-1, 1)
        # Generate a random speed between 0 and the maximum initial speed
        fmagnitude = random.uniform(min_speed, max_speed)
        # Use the random speed for the velocity.
        vec3Normalize(fx, fy, fz)
        vec3Mult(fx, fy, fz, fmagnitude)

        # Set these for the agent.
        instance.setVariableFloat("fx", fx)
        instance.setVariableFloat("fy", fy)
        instance.setVariableFloat("fz", fz)

        # initialise wing speed
        instance.setVariableFloat("wing_position", random.uniform(0, 3.14))
        instance.setVariableFloat("wing_animation", 0)

    cudaSimulation.setPopulationData(population)

# Execution
cudaSimulation.simulate()

# Export Pop (optional)
# cudaSimulation.exportData("end.xml")

if pyflamegpu.VISUALISATION:
    visualisation.join()
