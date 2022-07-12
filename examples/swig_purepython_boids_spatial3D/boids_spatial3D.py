from pyflamegpu import *
import pyflamegpu.codegen
import sys, random, math

"""
  FLAME GPU 2 implementation of the Boids model, using spatial3D messaging and pure Python agent functions.
  This is based on the FLAME GPU 1 implementation, but with dynamic generation of agents and is equivalent to the non pure Python version.
"""


"""
  Get the length of a vector
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @return the length of the vector
"""
def vec3Length(x, y, z):
    return math.sqrt(x * x + y * y + z * z);

"""
  Add a scalar to a vector in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param value scalar value to add
"""
def vec3Add(x, y, z, value):
    x += value;
    y += value;
    z += value;

"""
  Subtract a scalar from a vector in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param value scalar value to subtract
"""
def vec3Sub(x, y, z, value):
    x -= value;
    y -= value;
    z -= value;

"""
  Multiply a vector by a scalar value in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param multiplier scalar value to multiply by
"""
def vec3Mult(x, y, z, multiplier):
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;

"""
  Divide a vector by a scalar value in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param divisor scalar value to divide by
"""
def vec3Div(x, y, z, divisor):
    x /= divisor;
    y /= divisor;
    z /= divisor;


"""
  Normalize a 3 component vector in-place
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
"""
def vec3Normalize(x, y, z):
    # Get the length
    length = vec3Length(x, y, z);
    vec3Div(x, y, z, length);

"""
  Clamp each component of a 3-part position to lie within a minimum and maximum value.
  Performs the operation in place
  Unlike the FLAME GPU 1 example, this is a clamping operation, rather than wrapping.
  @param x x component of the vector
  @param y y component of the vector
  @param z z component of the vector
  @param MIN_POSITION the minimum value for each component
  @param MAX_POSITION the maximum value for each component
"""
def clampPosition(x, y, z, MIN_POSITION, MAX_POSITION):
    x = MIN_POSITION if (x < MIN_POSITION) else x;
    x = MAX_POSITION if (x > MAX_POSITION) else x;

    y = MIN_POSITION if (y < MIN_POSITION) else y;
    y = MAX_POSITION if (y > MAX_POSITION) else y;

    z = MIN_POSITION if (z < MIN_POSITION) else z;
    z = MAX_POSITION if (z > MAX_POSITION) else z;

# Change to false if pyflamegpu has not been built with visualisation support
VISUALISATION = True;

"""
  Pure python version of outputdata agent function for Boid agents, which outputs publicly visible properties to a message list. 
"""
outputdata_py = r"""
@flamegpu_agent_function
def outputdata(message_in: MessageNone, message_out: MessageSpatial3D):
    # Output each agents publicly visible properties.
    message_out.setVariableInt("id", FLAMEGPU.getID())
    message_out.setVariableFloat("x", FLAMEGPU.getVariableFloat("x"))
    message_out.setVariableFloat("y", FLAMEGPU.getVariableFloat("y"))
    message_out.setVariableFloat("z", FLAMEGPU.getVariableFloat("z"))
    message_out.setVariableFloat("fx", FLAMEGPU.getVariableFloat("fx"))
    message_out.setVariableFloat("fy", FLAMEGPU.getVariableFloat("fy"))
    message_out.setVariableFloat("fz", FLAMEGPU.getVariableFloat("fz"))
    return FLAMEGPU.ALIVE;
"""

"""
  Pure python version of inputdata agent function for Boid agents, which reads data from neighboring Boid agents, to perform the boid flocking model.
  Helper functions which use pass by reference basic types are not possible without the use of python containers.
"""
inputdata_py = r"""
# Vector utility functions, see top of file for versions with commentary
@flamegpu_device_function
def vec3Length(x: float, y: float, z: float) -> float :
    return sqrtf(x * x + y * y + z * z)

@flamegpu_device_function
def clamp(v : float, min: float, max: float) -> float:
    v = min if v < min else v
    v = max if v > max else v
    return v


@flamegpu_agent_function
def inputdata(message_in: MessageSpatial3D, message_out: MessageNone):
    # Agent properties in local register
    id = FLAMEGPU.getID()
    # Agent position
    agent_x = FLAMEGPU.getVariableFloat("x")
    agent_y = FLAMEGPU.getVariableFloat("y")
    agent_z = FLAMEGPU.getVariableFloat("z")
    #/ Agent velocity
    agent_fx = FLAMEGPU.getVariableFloat("fx")
    agent_fy = FLAMEGPU.getVariableFloat("fy")
    agent_fz = FLAMEGPU.getVariableFloat("fz")

    # Boids percieved center
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

    INTERACTION_RADIUS = FLAMEGPU.environment.getPropertyFloat("INTERACTION_RADIUS")
    SEPARATION_RADIUS = FLAMEGPU.environment.getPropertyFloat("SEPARATION_RADIUS")
    # Iterate location messages, accumulating relevant data and counts.
    for message in message_in(agent_x, agent_y, agent_z) :
        # Ignore self messages.
        if message.getVariableInt("id") != id :
            # Get the message location and velocity.
            message_x = message.getVariableFloat("x")
            message_y = message.getVariableFloat("y")
            message_z = message.getVariableFloat("z")

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

                    collisionScale = FLAMEGPU.environment.getPropertyFloat("COLLISION_SCALE")
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

        STEER_SCALE = FLAMEGPU.environment.getPropertyFloat("STEER_SCALE")
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

        MATCH_SCALE = FLAMEGPU.environment.getPropertyFloat("MATCH_SCALE")
        match_velocity_x = global_velocity_x * MATCH_SCALE
        match_velocity_y = global_velocity_y * MATCH_SCALE
        match_velocity_z = global_velocity_z * MATCH_SCALE

        velocity_change_x += match_velocity_x - agent_fx
        velocity_change_y += match_velocity_y - agent_fy
        velocity_change_z += match_velocity_z - agent_fz


    # Global scale of velocity change
    GLOBAL_SCALE = FLAMEGPU.environment.getPropertyFloat("GLOBAL_SCALE")
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

    # Steer away from walls - Computed post normalization to ensure good avoidance. Prevents constant term getting swamped
    wallInteractionDistance = float(0.10)
    wallSteerStrength = float(0.05)
    minPosition = FLAMEGPU.environment.getPropertyFloat("MIN_POSITION")
    maxPosition = FLAMEGPU.environment.getPropertyFloat("MAX_POSITION")

    if (agent_x - minPosition) < wallInteractionDistance :
        agent_fx += wallSteerStrength
    if (agent_y - minPosition) < wallInteractionDistance :
        agent_fy += wallSteerStrength
    if (agent_z - minPosition) < wallInteractionDistance :
        agent_fz += wallSteerStrength

    if (maxPosition - agent_x) < wallInteractionDistance :
        agent_fx -= wallSteerStrength
    if (maxPosition - agent_y) < wallInteractionDistance :
        agent_fy -= wallSteerStrength
    if (maxPosition - agent_z) < wallInteractionDistance :
        agent_fz -= wallSteerStrength


    # Apply the velocity
    TIME_SCALE = FLAMEGPU.environment.getPropertyFloat("TIME_SCALE")
    agent_x += agent_fx * TIME_SCALE
    agent_y += agent_fy * TIME_SCALE
    agent_z += agent_fz * TIME_SCALE

    # Bound position
    MIN_POSITION = FLAMEGPU.environment.getPropertyFloat("MIN_POSITION")
    MAX_POSITION = FLAMEGPU.environment.getPropertyFloat("MAX_POSITION")
    clamp(agent_x, MIN_POSITION, MAX_POSITION)
    clamp(agent_y, MIN_POSITION, MAX_POSITION)
    clamp(agent_z, MIN_POSITION, MAX_POSITION)

    # Update global agent memory.
    FLAMEGPU.setVariableFloat("x", agent_x)
    FLAMEGPU.setVariableFloat("y", agent_y)
    FLAMEGPU.setVariableFloat("z", agent_z)

    FLAMEGPU.setVariableFloat("fx", agent_fx)
    FLAMEGPU.setVariableFloat("fy", agent_fy)
    FLAMEGPU.setVariableFloat("fz", agent_fz)

    return FLAMEGPU.ALIVE

"""

model = pyflamegpu.ModelDescription("Boids_BruteForce");

"""
  GLOBALS
"""
env = model.Environment();
# Population size to generate, if no agents are loaded from disk
env.newPropertyUInt("POPULATION_TO_GENERATE", 4000);

# Environment Bounds
env.newPropertyFloat("MIN_POSITION", -0.5);
env.newPropertyFloat("MAX_POSITION", +0.5);

# Initialisation parameter(s)
env.newPropertyFloat("MAX_INITIAL_SPEED", 1.0);
env.newPropertyFloat("MIN_INITIAL_SPEED", 0.1);

# Interaction radius
env.newPropertyFloat("INTERACTION_RADIUS", 0.05);
env.newPropertyFloat("SEPARATION_RADIUS", 0.01);

# Global Scalers
env.newPropertyFloat("TIME_SCALE", 0.0005);
env.newPropertyFloat("GLOBAL_SCALE", 0.15);

# Rule scalers
env.newPropertyFloat("STEER_SCALE", 0.055);
env.newPropertyFloat("COLLISION_SCALE", 10.0);
env.newPropertyFloat("MATCH_SCALE", 0.015);

"""
  Location message
"""
message = model.newMessageSpatial3D("location");
# Set the range and bounds.
message.setRadius(env.getPropertyFloat("INTERACTION_RADIUS"));
message.setMin(env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"), env.getPropertyFloat("MIN_POSITION"));
message.setMax(env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"), env.getPropertyFloat("MAX_POSITION"));
# A message to hold the location of an agent.
message.newVariableID("id");
# X Y Z are implicit.
# message.newVariable<float>("x");
# message.newVariable<float>("y");
# message.newVariable<float>("z");
message.newVariableFloat("fx");
message.newVariableFloat("fy");
message.newVariableFloat("fz");
    
"""
  Boid agent
"""
agent = model.newAgent("Boid");
agent.newVariableFloat("x");
agent.newVariableFloat("y");
agent.newVariableFloat("z");
agent.newVariableFloat("fx");
agent.newVariableFloat("fy");
agent.newVariableFloat("fz");
outputdata_translated = pyflamegpu.codegen.translate(outputdata_py)
inputdata_translated = pyflamegpu.codegen.translate(inputdata_py)
print(inputdata_translated)
agent.newRTCFunction("outputdata", outputdata_translated).setMessageOutput("location");
agent.newRTCFunction("inputdata", inputdata_translated).setMessageInput("location");


"""
  Control flow
"""    
# Layer #1
model.newLayer().addAgentFunction("Boid", "outputdata");
# Layer #2
model.newLayer().addAgentFunction("Boid", "inputdata");

"""
  Create Model Runner
"""   
cudaSimulation = pyflamegpu.CUDASimulation(model);

"""
  Create Visualisation
"""
if pyflamegpu.VISUALISATION:
    visualisation = cudaSimulation.getVisualisation();
    # Configure vis
    envWidth = env.getPropertyFloat("MAX_POSITION") - env.getPropertyFloat("MIN_POSITION");
    INIT_CAM = env.getPropertyFloat("MAX_POSITION") * 1.25;
    visualisation.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM);
    visualisation.setCameraSpeed(0.001 * envWidth);
    visualisation.setViewClips(0.00001, 50);
    circ_agt = visualisation.addAgent("Boid");
    # Position vars are named x, y, z; so they are used by default
    circ_agt.setForwardXVariable("fx");
    circ_agt.setForwardYVariable("fy");
    circ_agt.setForwardZVariable("fz");
    circ_agt.setModel(pyflamegpu.STUNTPLANE);
    circ_agt.setModelScale(env.getPropertyFloat("SEPARATION_RADIUS") /3.0);
    visualisation.activate();

"""
  Initialise Model
"""
cudaSimulation.initialise(sys.argv);

# If no xml model file was is provided, generate a population.
if not cudaSimulation.SimulationConfig().input_file:
    # Uniformly distribute agents within space, with uniformly distributed initial velocity.
    random.seed(cudaSimulation.SimulationConfig().random_seed);
    min_pos = env.getPropertyFloat("MIN_POSITION");
    max_pos = env.getPropertyFloat("MAX_POSITION");
    min_speed = env.getPropertyFloat("MIN_INITIAL_SPEED");
    max_speed = env.getPropertyFloat("MAX_INITIAL_SPEED");
    populationSize = env.getPropertyUInt("POPULATION_TO_GENERATE");
    population = pyflamegpu.AgentVector(model.Agent("Boid"), populationSize);
    for i in range(populationSize):
        instance = population[i];

        # Agent position in space
        instance.setVariableFloat("x", random.uniform(min_pos, max_pos));
        instance.setVariableFloat("y", random.uniform(min_pos, max_pos));
        instance.setVariableFloat("z", random.uniform(min_pos, max_pos));

        # Generate a random velocity direction
        fx = random.uniform(-1, 1);
        fy = random.uniform(-1, 1);
        fz = random.uniform(-1, 1);
        # Generate a random speed between 0 and the maximum initial speed
        fmagnitude = random.uniform(min_speed, max_speed);
        # Use the random speed for the velocity.
        vec3Normalize(fx, fy, fz);
        vec3Mult(fx, fy, fz, fmagnitude);

        # Set these for the agent.
        instance.setVariableFloat("fx", fx);
        instance.setVariableFloat("fy", fy);
        instance.setVariableFloat("fz", fz);

    cudaSimulation.setPopulationData(population);

"""
  Execution
"""
cudaSimulation.simulate();

"""
  Export Pop
"""
# cudaSimulation.exportData("end.xml");

if pyflamegpu.VISUALISATION:
    visualisation.join();
