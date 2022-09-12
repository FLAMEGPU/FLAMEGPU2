from pyflamegpu import *
import sys, random, math

"""
  FLAME GPU 2 implementation of the Boids model, using spatial3D messaging.
  This is based on the FLAME GPU 1 implementation, but with dynamic generation of agents. 
  Agents are also clamped to be within the environment bounds, rather than wrapped as in FLAME GPU 1.

  @todo - Should the agent's velocity change when it is clamped to the environment?
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
  outputdata agent function for Boid agents, which outputs publicly visible properties to a message list
"""
outputdata = r"""
FLAMEGPU_AGENT_FUNCTION(outputdata, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    // Output each agents publicly visible properties.
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->message_out.setVariable<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->message_out.setVariable<float>("z", FLAMEGPU->getVariable<float>("z"));
    FLAMEGPU->message_out.setVariable<float>("fx", FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->message_out.setVariable<float>("fy", FLAMEGPU->getVariable<float>("fy"));
    FLAMEGPU->message_out.setVariable<float>("fz", FLAMEGPU->getVariable<float>("fz"));
    return flamegpu::ALIVE;
}
"""
"""
  inputdata agent function for Boid agents, which reads data from neighbouring Boid agents, to perform the boid flocking model.
"""
inputdata = r"""
// Vector utility functions, see top of file for versions with commentary
FLAMEGPU_HOST_DEVICE_FUNCTION float vec3Length(const float x, const float y, const float z) {
    return sqrtf(x * x + y * y + z * z);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Add(float &x, float &y, float &z, const float value) {
    x += value;
    y += value;
    z += value;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Sub(float &x, float &y, float &z, const float value) {
    x -= value;
    y -= value;
    z -= value;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Mult(float &x, float &y, float &z, const float multiplier) {
    x *= multiplier;
    y *= multiplier;
    z *= multiplier;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Div(float &x, float &y, float &z, const float divisor) {
    x /= divisor;
    y /= divisor;
    z /= divisor;
}
FLAMEGPU_HOST_DEVICE_FUNCTION void vec3Normalize(float &x, float &y, float &z) {
    // Get the length
    float length = vec3Length(x, y, z);
    vec3Div(x, y, z, length);
}
FLAMEGPU_HOST_DEVICE_FUNCTION void clampPosition(float &x, float &y, float &z, const float MIN_POSITION, const float MAX_POSITION) {
    x = (x < MIN_POSITION)? MIN_POSITION: x;
    x = (x > MAX_POSITION)? MAX_POSITION: x;

    y = (y < MIN_POSITION)? MIN_POSITION: y;
    y = (y > MAX_POSITION)? MAX_POSITION: y;

    z = (z < MIN_POSITION)? MIN_POSITION: z;
    z = (z > MAX_POSITION)? MAX_POSITION: z;
}
// Agent function
FLAMEGPU_AGENT_FUNCTION(inputdata, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    // Agent properties in local register
    const flamegpu::id_t id = FLAMEGPU->getID();
    // Agent position
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    float agent_z = FLAMEGPU->getVariable<float>("z");
    // Agent velocity
    float agent_fx = FLAMEGPU->getVariable<float>("fx");
    float agent_fy = FLAMEGPU->getVariable<float>("fy");
    float agent_fz = FLAMEGPU->getVariable<float>("fz");

    // Boids percieved center
    float perceived_centre_x = 0.0f;
    float perceived_centre_y = 0.0f;
    float perceived_centre_z = 0.0f;
    int perceived_count = 0;

    // Boids global velocity matching
    float global_velocity_x = 0.0f;
    float global_velocity_y = 0.0f;
    float global_velocity_z = 0.0f;

    // Total change in velocity
    float velocity_change_x = 0.f;
    float velocity_change_y = 0.f;
    float velocity_change_z = 0.f;

    const float INTERACTION_RADIUS = FLAMEGPU->environment.getProperty<float>("INTERACTION_RADIUS");
    const float SEPARATION_RADIUS = FLAMEGPU->environment.getProperty<float>("SEPARATION_RADIUS");
    // Iterate location messages, accumulating relevant data and counts.
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        // Ignore self messages.
        if (message.getVariable<flamegpu::id_t>("id") != id) {
            // Get the message location and velocity.
            const float message_x = message.getVariable<float>("x");
            const float message_y = message.getVariable<float>("y");
            const float message_z = message.getVariable<float>("z");

            // Check interaction radius
            float separation = vec3Length(agent_x - message_x, agent_y - message_y, agent_z - message_z);

            if (separation < INTERACTION_RADIUS) {
                // Update the percieved centre
                perceived_centre_x += message_x;
                perceived_centre_y += message_y;
                perceived_centre_z += message_z;
                perceived_count++;

                // Update percieved velocity matching
                const float message_fx = message.getVariable<float>("fx");
                const float message_fy = message.getVariable<float>("fy");
                const float message_fz = message.getVariable<float>("fz");
                global_velocity_x += message_fx;
                global_velocity_y += message_fy;
                global_velocity_z += message_fz;

                // Update collision centre
                if (separation < (SEPARATION_RADIUS)) {  // dependant on model size
                    // Rule 3) Avoid other nearby boids (Separation)
                    float normalizedSeparation = (separation / SEPARATION_RADIUS);
                    float invNormSep = (1.0f - normalizedSeparation);
                    float invSqSep = invNormSep * invNormSep;

                    const float collisionScale = FLAMEGPU->environment.getProperty<float>("COLLISION_SCALE");
                    velocity_change_x += collisionScale * (agent_x - message_x) * invSqSep;
                    velocity_change_y += collisionScale * (agent_y - message_y) * invSqSep;
                    velocity_change_z += collisionScale * (agent_z - message_z) * invSqSep;
                }
            }
        }
    }

    if (perceived_count) {
        // Divide positions/velocities by relevant counts.
        vec3Div(perceived_centre_x, perceived_centre_y, perceived_centre_z, perceived_count);
        vec3Div(global_velocity_x, global_velocity_y, global_velocity_z, perceived_count);

        // Rule 1) Steer towards perceived centre of flock (Cohesion)
        float steer_velocity_x = 0.f;
        float steer_velocity_y = 0.f;
        float steer_velocity_z = 0.f;

        const float STEER_SCALE = FLAMEGPU->environment.getProperty<float>("STEER_SCALE");
        steer_velocity_x = (perceived_centre_x - agent_x) * STEER_SCALE;
        steer_velocity_y = (perceived_centre_y - agent_y) * STEER_SCALE;
        steer_velocity_z = (perceived_centre_z - agent_z) * STEER_SCALE;

        velocity_change_x += steer_velocity_x;
        velocity_change_y += steer_velocity_y;
        velocity_change_z += steer_velocity_z;

        // Rule 2) Match neighbours speeds (Alignment)
        float match_velocity_x = 0.f;
        float match_velocity_y = 0.f;
        float match_velocity_z = 0.f;

        const float MATCH_SCALE = FLAMEGPU->environment.getProperty<float>("MATCH_SCALE");
        match_velocity_x = global_velocity_x * MATCH_SCALE;
        match_velocity_y = global_velocity_y * MATCH_SCALE;
        match_velocity_z = global_velocity_z * MATCH_SCALE;

        velocity_change_x += match_velocity_x - agent_fx;
        velocity_change_y += match_velocity_y - agent_fy;
        velocity_change_z += match_velocity_z - agent_fz;
    }

    // Global scale of velocity change
    vec3Mult(velocity_change_x, velocity_change_y, velocity_change_z, FLAMEGPU->environment.getProperty<float>("GLOBAL_SCALE"));

    // Update agent velocity
    agent_fx += velocity_change_x;
    agent_fy += velocity_change_y;
    agent_fz += velocity_change_z;

    // Bound velocity
    float agent_fscale = vec3Length(agent_fx, agent_fy, agent_fz);
    if (agent_fscale > 1) {
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale);
    }

    float minSpeed = 0.5f;
    if (agent_fscale < minSpeed) {
        // Normalise
        vec3Div(agent_fx, agent_fy, agent_fz, agent_fscale);

        // Scale to min
        vec3Mult(agent_fx, agent_fy, agent_fz, minSpeed);
    }

    // Steer away from walls - Computed post normalization to ensure good avoidance. Prevents constant term getting swamped
    const float wallInteractionDistance = 0.10f;
    const float wallSteerStrength = 0.05f;
    const float minPosition = FLAMEGPU->environment.getProperty<float>("MIN_POSITION");
    const float maxPosition = FLAMEGPU->environment.getProperty<float>("MAX_POSITION");

    if (agent_x - minPosition < wallInteractionDistance) {
        agent_fx += wallSteerStrength;
    }
    if (agent_y - minPosition < wallInteractionDistance) {
        agent_fy += wallSteerStrength;
    }
    if (agent_z - minPosition < wallInteractionDistance) {
        agent_fz += wallSteerStrength;
    }

    if (maxPosition - agent_x < wallInteractionDistance) {
        agent_fx -= wallSteerStrength;
    }
    if (maxPosition - agent_y < wallInteractionDistance) {
        agent_fy -= wallSteerStrength;
    }
    if (maxPosition - agent_z < wallInteractionDistance) {
        agent_fz -= wallSteerStrength;
    }

    // Apply the velocity
    const float TIME_SCALE = FLAMEGPU->environment.getProperty<float>("TIME_SCALE");
    agent_x += agent_fx * TIME_SCALE;
    agent_y += agent_fy * TIME_SCALE;
    agent_z += agent_fz * TIME_SCALE;

    // Bound position
    clampPosition(agent_x, agent_y, agent_z, FLAMEGPU->environment.getProperty<float>("MIN_POSITION"), FLAMEGPU->environment.getProperty<float>("MAX_POSITION"));

    // Update global agent memory.
    FLAMEGPU->setVariable<float>("x", agent_x);
    FLAMEGPU->setVariable<float>("y", agent_y);
    FLAMEGPU->setVariable<float>("z", agent_z);

    FLAMEGPU->setVariable<float>("fx", agent_fx);
    FLAMEGPU->setVariable<float>("fy", agent_fy);
    FLAMEGPU->setVariable<float>("fz", agent_fz);

    return flamegpu::ALIVE;
}
"""

model = pyflamegpu.ModelDescription("Boids Spatial3D (Python)");

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
agent.newRTCFunction("outputdata", outputdata).setMessageOutput("location");
agent.newRTCFunction("inputdata", inputdata).setMessageInput("location");


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
    # Add a settings UI
    ui = visualisation.newUIPanel("Environment");
    ui.newStaticLabel("Interaction");
    ui.newEnvironmentPropertyDragFloat("INTERACTION_RADIUS", 0.0, 0.05, 0.001);
    ui.newEnvironmentPropertyDragFloat("SEPARATION_RADIUS", 0.0, 0.05, 0.001);
    ui.newStaticLabel("Environment Scalars");
    ui.newEnvironmentPropertyDragFloat("TIME_SCALE", 0.0, 1.0, 0.0001);
    ui.newEnvironmentPropertyDragFloat("GLOBAL_SCALE", 0.0, 0.5, 0.001);
    ui.newStaticLabel("Force Scalars");
    ui.newEnvironmentPropertyDragFloat("STEER_SCALE", 0.0, 10.0, 0.001);
    ui.newEnvironmentPropertyDragFloat("COLLISION_SCALE", 0.0, 10.0, 0.001);
    ui.newEnvironmentPropertyDragFloat("MATCH_SCALE", 0.0, 10.0, 0.001);
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
