#include "flamegpu/flamegpu.h"
#include <iostream>

#define SCALE_FACTOR 0.03125

#define I_SCALER (SCALE_FACTOR*0.35f)
#define MIN_DISTANCE 0.0001f

#define ENV_MAX 1.0f
#define ENV_MIN -ENV_MAX
#define ENV_WIDTH (2*ENV_MAX)

FLAMEGPU_DEVICE_FUNCTION int getNewExitLocation(flamegpu::DeviceAPI<flamegpu::MessageArray2D, flamegpu::MessageNone> *FLAMEGPU) {
    // Load Env Properties once
    const int EXIT1_PROBABILITY = FLAMEGPU->environment.getProperty<int>("EXIT1_PROBABILITY");
    const int EXIT2_PROBABILITY = FLAMEGPU->environment.getProperty<int>("EXIT2_PROBABILITY");
    const int EXIT3_PROBABILITY = FLAMEGPU->environment.getProperty<int>("EXIT3_PROBABILITY");
    const int EXIT4_PROBABILITY = FLAMEGPU->environment.getProperty<int>("EXIT4_PROBABILITY");
    const int EXIT5_PROBABILITY = FLAMEGPU->environment.getProperty<int>("EXIT5_PROBABILITY");
    const int EXIT6_PROBABILITY = FLAMEGPU->environment.getProperty<int>("EXIT6_PROBABILITY");
    const int EXIT7_PROBABILITY = FLAMEGPU->environment.getProperty<int>("EXIT7_PROBABILITY");
    const int exit1_compare = EXIT1_PROBABILITY;
    const int exit2_compare = EXIT2_PROBABILITY + exit1_compare;
    const int exit3_compare = EXIT3_PROBABILITY + exit2_compare;
    const int exit4_compare = EXIT4_PROBABILITY + exit3_compare;
    const int exit5_compare = EXIT5_PROBABILITY + exit4_compare;
    const int exit6_compare = EXIT6_PROBABILITY + exit5_compare;

    const float range = static_cast<float>(EXIT1_PROBABILITY +
                   EXIT2_PROBABILITY +
                   EXIT3_PROBABILITY +
                   EXIT4_PROBABILITY +
                   EXIT5_PROBABILITY +
                   EXIT6_PROBABILITY +
                   EXIT7_PROBABILITY);

    const float rand = FLAMEGPU->random.uniform<float>()*range;

     if (rand < exit1_compare)
         return 1;
     else if (rand < exit2_compare)
         return 2;
     else if (rand < exit3_compare)
         return 3;
     else if (rand < exit4_compare)
         return 4;
     else if (rand < exit5_compare)
         return 5;
     else if (rand < exit6_compare)
         return 6;
     else
         return 7;
}
FLAMEGPU_AGENT_FUNCTION(output_pedestrian_location, flamegpu::MessageNone, flamegpu::MessageSpatial2D) {
    FLAMEGPU->message_out.setLocation(FLAMEGPU->getVariable<float>("x"), FLAMEGPU->getVariable<float>("y"));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(output_navmap_cells, flamegpu::MessageNone, flamegpu::MessageArray2D) {
#ifdef FLAMEGPU_VISUALISATION
    // It would be far more efficient to set these once within the input file, as they do not change
    if (FLAMEGPU->getStepCounter() == 0) {
        FLAMEGPU->setVariable<float>("x_vis", (ENV_WIDTH / 256.0f) * FLAMEGPU->getVariable<int>("x") - (ENV_WIDTH / 2.0f) + (ENV_WIDTH / 512.0f));
        FLAMEGPU->setVariable<float>("y_vis", 0.999f);
        FLAMEGPU->setVariable<float>("z_vis", (ENV_WIDTH / 256.0f) * FLAMEGPU->getVariable<int>("y") - (ENV_WIDTH / 2.0f) + (ENV_WIDTH / 512.0f));
        FLAMEGPU->setVariable<int>("is_active", FLAMEGPU->getVariable<float>("collision_x") + FLAMEGPU->getVariable<float>("collision_y") == 0 ? 0 : 1);
    }
#endif

    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<int>("x"), FLAMEGPU->getVariable<int>("y"));
    FLAMEGPU->message_out.setVariable<int>("exit_no", FLAMEGPU->getVariable<int>("exit_no"));
    FLAMEGPU->message_out.setVariable<float>("height", FLAMEGPU->getVariable<float>("height"));
    FLAMEGPU->message_out.setVariable<float>("collision_x", FLAMEGPU->getVariable<float>("collision_x"));
    FLAMEGPU->message_out.setVariable<float>("collision_y", FLAMEGPU->getVariable<float>("collision_y"));
    FLAMEGPU->message_out.setVariable<float>("exit0_x", FLAMEGPU->getVariable<float>("exit0_x"));
    FLAMEGPU->message_out.setVariable<float>("exit0_y", FLAMEGPU->getVariable<float>("exit0_y"));
    FLAMEGPU->message_out.setVariable<float>("exit1_x", FLAMEGPU->getVariable<float>("exit1_x"));
    FLAMEGPU->message_out.setVariable<float>("exit1_y", FLAMEGPU->getVariable<float>("exit1_y"));
    FLAMEGPU->message_out.setVariable<float>("exit2_x", FLAMEGPU->getVariable<float>("exit2_x"));
    FLAMEGPU->message_out.setVariable<float>("exit2_y", FLAMEGPU->getVariable<float>("exit2_y"));
    FLAMEGPU->message_out.setVariable<float>("exit3_x", FLAMEGPU->getVariable<float>("exit3_x"));
    FLAMEGPU->message_out.setVariable<float>("exit3_y", FLAMEGPU->getVariable<float>("exit3_y"));
    FLAMEGPU->message_out.setVariable<float>("exit4_x", FLAMEGPU->getVariable<float>("exit4_x"));
    FLAMEGPU->message_out.setVariable<float>("exit4_y", FLAMEGPU->getVariable<float>("exit4_y"));
    FLAMEGPU->message_out.setVariable<float>("exit5_x", FLAMEGPU->getVariable<float>("exit5_x"));
    FLAMEGPU->message_out.setVariable<float>("exit5_y", FLAMEGPU->getVariable<float>("exit5_y"));
    FLAMEGPU->message_out.setVariable<float>("exit6_x", FLAMEGPU->getVariable<float>("exit6_x"));
    FLAMEGPU->message_out.setVariable<float>("exit6_y", FLAMEGPU->getVariable<float>("exit6_y"));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(avoid_pedestrians, flamegpu::MessageSpatial2D, flamegpu::MessageNone) {
    const float STEER_WEIGHT = FLAMEGPU->environment.getProperty<float>("STEER_WEIGHT");
    const float AVOID_WEIGHT = FLAMEGPU->environment.getProperty<float>("AVOID_WEIGHT");

    glm::vec2 agent_pos = glm::vec2(FLAMEGPU->getVariable<float>("x"), FLAMEGPU->getVariable<float>("y"));
    glm::vec2 agent_vel = glm::vec2(FLAMEGPU->getVariable<float>("velx"), FLAMEGPU->getVariable<float>("vely"));

    glm::vec2 navigate_velocity = glm::vec2(0.0f, 0.0f);
    glm::vec2 avoid_velocity = glm::vec2(0.0f, 0.0f);

    for (auto &current_message : FLAMEGPU->message_in(agent_pos.x, agent_pos.y)) {
        glm::vec2 message_pos = glm::vec2(current_message.getVariable<float>("x"), current_message.getVariable<float>("y"));
        float separation = length(agent_pos - message_pos);
        if ((separation < FLAMEGPU->message_in.radius()) && (separation > MIN_DISTANCE)) {
            glm::vec2 to_agent = normalize(agent_pos - message_pos);
            float ang = acosf(dot(agent_vel, to_agent));
            float perception = 45.0f;

            // STEER
            if ((ang < glm::radians(perception)) || (ang > 3.14159265f - glm::radians(perception))) {
                glm::vec2 s_velocity = to_agent;
                s_velocity *= powf(I_SCALER/separation, 1.25f)*STEER_WEIGHT;
                navigate_velocity += s_velocity;
            }

            // AVOID
            glm::vec2 a_velocity = to_agent;
            a_velocity *= powf(I_SCALER/separation, 2.00f)*AVOID_WEIGHT;
            avoid_velocity += a_velocity;
        }
    }

    // maximum velocity rule
    glm::vec2 steer_velocity = navigate_velocity + avoid_velocity;

    FLAMEGPU->setVariable<float>("steer_x", steer_velocity.x);
    FLAMEGPU->setVariable<float>("steer_y", steer_velocity.y);

    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(force_flow, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    // map agent position into 2d grid
    const int x = floor(((FLAMEGPU->getVariable<float>("x") + ENV_MAX) / ENV_WIDTH) * FLAMEGPU->message_in.getDimX());
    const int y = floor(((FLAMEGPU->getVariable<float>("y") + ENV_MAX) / ENV_WIDTH) * FLAMEGPU->message_in.getDimY());

    // lookup single message
    const auto current_message = FLAMEGPU->message_in.at(x, y);

    glm::vec2 collision_force = glm::vec2(current_message.getVariable<float>("collision_x"), current_message.getVariable<float>("collision_y"));
    collision_force *= FLAMEGPU->environment.getProperty<float>("COLLISION_WEIGHT");

    // exit location of cell
    const int exit_location = current_message.getVariable<int>("exit_no");

    // agent death flag
    flamegpu::AGENT_STATUS kill_agent = flamegpu::ALIVE;

    // goal force
    glm::vec2 goal_force;
    int agent_exit_no = FLAMEGPU->getVariable<int>("exit_no");
    if (agent_exit_no == 1) {
        goal_force = glm::vec2(current_message.getVariable<float>("exit0_x"), current_message.getVariable<float>("exit0_y"));
        if (exit_location == 1) {
            if (FLAMEGPU->environment.getProperty<int>("EXIT1_STATE")) {
                kill_agent = flamegpu::DEAD;
            } else {
                agent_exit_no = getNewExitLocation(FLAMEGPU);
            }
        }
     } else if (agent_exit_no == 2) {
        goal_force = glm::vec2(current_message.getVariable<float>("exit1_x"), current_message.getVariable<float>("exit1_y"));
        if (exit_location == 2) {
            if (FLAMEGPU->environment.getProperty<int>("EXIT2_STATE")) {
                kill_agent = flamegpu::DEAD;
            } else {
                agent_exit_no = getNewExitLocation(FLAMEGPU);
            }
        }
    } else if (agent_exit_no == 3) {
        goal_force = glm::vec2(current_message.getVariable<float>("exit2_x"), current_message.getVariable<float>("exit2_y"));
        if (exit_location == 3) {
            if (FLAMEGPU->environment.getProperty<int>("EXIT3_STATE")) {
                kill_agent = flamegpu::DEAD;
            } else {
                agent_exit_no = getNewExitLocation(FLAMEGPU);
            }
        }
    } else if (agent_exit_no == 4) {
        goal_force = glm::vec2(current_message.getVariable<float>("exit3_x"), current_message.getVariable<float>("exit3_y"));
        if (exit_location == 4) {
            if (FLAMEGPU->environment.getProperty<int>("EXIT4_STATE")) {
                kill_agent = flamegpu::DEAD;
            } else {
                agent_exit_no = getNewExitLocation(FLAMEGPU);
            }
        }
    } else if (agent_exit_no == 5) {
        goal_force = glm::vec2(current_message.getVariable<float>("exit4_x"), current_message.getVariable<float>("exit4_y"));
        if (exit_location == 5) {
            if (FLAMEGPU->environment.getProperty<int>("EXIT5_STATE")) {
                kill_agent = flamegpu::DEAD;
            } else {
                agent_exit_no = getNewExitLocation(FLAMEGPU);
            }
        }
    } else if (agent_exit_no == 6) {
        goal_force = glm::vec2(current_message.getVariable<float>("exit5_x"), current_message.getVariable<float>("exit5_y"));
        if (exit_location == 6) {
            if (FLAMEGPU->environment.getProperty<int>("EXIT6_STATE")) {
                kill_agent = flamegpu::DEAD;
            } else {
                agent_exit_no = getNewExitLocation(FLAMEGPU);
            }
        }
    } else if (agent_exit_no == 7) {
        goal_force = glm::vec2(current_message.getVariable<float>("exit6_x"), current_message.getVariable<float>("exit6_y"));
        if (exit_location == 7) {
            if (FLAMEGPU->environment.getProperty<int>("EXIT7_STATE")) {
                kill_agent = flamegpu::DEAD;
            } else {
                agent_exit_no = getNewExitLocation(FLAMEGPU);
            }
        }
    }
    FLAMEGPU->setVariable<int>("exit_no", agent_exit_no);

    // scale goal force
    goal_force *= FLAMEGPU->environment.getProperty<float>("GOAL_WEIGHT");

    FLAMEGPU->setVariable<float>("steer_x", FLAMEGPU->getVariable<float>("steer_x") + collision_force.x + goal_force.x);
    FLAMEGPU->setVariable<float>("steer_y", FLAMEGPU->getVariable<float>("steer_y") + collision_force.y + goal_force.y);

    // update height
    FLAMEGPU->setVariable<float>("height", current_message.getVariable<float>("height"));

    return kill_agent;
}
FLAMEGPU_AGENT_FUNCTION(move, flamegpu::MessageNone, flamegpu::MessageNone) {
    glm::vec2 agent_pos = glm::vec2(FLAMEGPU->getVariable<float>("x"), FLAMEGPU->getVariable<float>("y"));
    glm::vec2 agent_vel = glm::vec2(FLAMEGPU->getVariable<float>("velx"), FLAMEGPU->getVariable<float>("vely"));
    glm::vec2 agent_steer = glm::vec2(FLAMEGPU->getVariable<float>("steer_x"), FLAMEGPU->getVariable<float>("steer_y"));

    float current_speed = length(agent_vel)+0.025f;  // (powf(length(agent_vel), 1.75f)*0.01f)+0.025f;

    // apply more steer if speed is greater
    agent_vel += current_speed * agent_steer;
    float speed = length(agent_vel);
    // limit speed
    const float agent_speed = FLAMEGPU->getVariable<float>("speed");
    if (speed >= agent_speed) {
        agent_vel = normalize(agent_vel)* agent_speed;
        speed = agent_speed;
    }

     // update position
    const float TIME_SCALER = FLAMEGPU->environment.getProperty<float>("TIME_SCALER");
    agent_pos += agent_vel * TIME_SCALER;

    // animation
    const float agent_animate = FLAMEGPU->getVariable<float>("animate") + (FLAMEGPU->getVariable<int>("animate_dir") * powf(speed, 2.0f) * TIME_SCALER * 100.0f);
    if (agent_animate >= 1)
        FLAMEGPU->setVariable<int>("animate_dir", -1);
    else if (agent_animate <= 0)
        FLAMEGPU->setVariable<int>("animate_dir", 1);
    FLAMEGPU->setVariable<float>("animate", agent_animate);

    // lod
    FLAMEGPU->setVariable<int>("lod", 1);

    // bound by wrapping
    if (agent_pos.x < -1.0f)
        agent_pos.x += 2.0f;
    if (agent_pos.x > 1.0f)
        agent_pos.x -= 2.0f;
    if (agent_pos.y < -1.0f)
        agent_pos.y += 2.0f;
    if (agent_pos.y > 1.0f)
        agent_pos.y -= 2.0f;

    // update
    FLAMEGPU->setVariable<float>("x", agent_pos.x);
    FLAMEGPU->setVariable<float>("y", agent_pos.y);
    FLAMEGPU->setVariable<float>("velx", agent_vel.x);
    FLAMEGPU->setVariable<float>("vely", agent_vel.y);

    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(generate_pedestrians, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const int agent_exit_no = FLAMEGPU->getVariable<int>("exit_no");
    if (agent_exit_no > 0) {
        float random = FLAMEGPU->random.uniform<float>();
        bool emit_agent = false;
        const float TIME_SCALER = FLAMEGPU->environment.getProperty<float>("TIME_SCALER");
        if ((agent_exit_no == 1)&&((random < FLAMEGPU->environment.getProperty<float>("EMMISION_RATE_EXIT1") * TIME_SCALER)))
            emit_agent = true;
        if ((agent_exit_no == 2)&&((random < FLAMEGPU->environment.getProperty<float>("EMMISION_RATE_EXIT2") * TIME_SCALER)))
            emit_agent = true;
        if ((agent_exit_no == 3)&&((random < FLAMEGPU->environment.getProperty<float>("EMMISION_RATE_EXIT3") * TIME_SCALER)))
            emit_agent = true;
        if ((agent_exit_no == 4)&&((random < FLAMEGPU->environment.getProperty<float>("EMMISION_RATE_EXIT4") * TIME_SCALER)))
            emit_agent = true;
        if ((agent_exit_no == 5)&&((random < FLAMEGPU->environment.getProperty<float>("EMMISION_RATE_EXIT5") * TIME_SCALER)))
            emit_agent = true;
        if ((agent_exit_no == 6)&&((random < FLAMEGPU->environment.getProperty<float>("EMMISION_RATE_EXIT6") * TIME_SCALER)))
            emit_agent = true;
        if ((agent_exit_no == 7)&&((random < FLAMEGPU->environment.getProperty<float>("EMMISION_RATE_EXIT7") * TIME_SCALER)))
            emit_agent = true;

        if (emit_agent) {
            FLAMEGPU->agent_out.setVariable<float>("x", ((FLAMEGPU->getVariable<int>("x") + 0.5f) / (FLAMEGPU->message_in.getDimX() / ENV_WIDTH)) - ENV_MAX);
            FLAMEGPU->agent_out.setVariable<float>("y", ((FLAMEGPU->getVariable<int>("y") + 0.5f) / (FLAMEGPU->message_in.getDimY() / ENV_WIDTH)) - ENV_MAX);
            FLAMEGPU->agent_out.setVariable<int>("exit_no", getNewExitLocation(FLAMEGPU));
            FLAMEGPU->agent_out.setVariable<float>("height", FLAMEGPU->getVariable<float>("height"));
            FLAMEGPU->agent_out.setVariable<float>("animate", FLAMEGPU->random.uniform<float>());
            FLAMEGPU->agent_out.setVariable<float>("speed", FLAMEGPU->random.uniform<float>() * 0.5f + 1.0f);
        }
    }

    return flamegpu::ALIVE;
}

int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("Pedestrian Navigation");

    {   // Location message
        flamegpu::MessageSpatial2D::Description message = model.newMessage<flamegpu::MessageSpatial2D>("pedestrian_location");
        message.setRadius(0.025f);
        message.setMin(-1.0f, -1.0f);
        message.setMax(1.0f, 1.0f);
    }
    {   // Navmap cell message
        flamegpu::MessageArray2D::Description message = model.newMessage<flamegpu::MessageArray2D>("navmap_cell");
        message.setDimensions(256, 256);
        message.newVariable<int>("x");
        message.newVariable<int>("y");
        message.newVariable<int>("exit_no");
        message.newVariable<float>("height");
        message.newVariable<float>("collision_x");
        message.newVariable<float>("collision_y");
        message.newVariable<float>("exit0_x");
        message.newVariable<float>("exit0_y");
        message.newVariable<float>("exit1_x");
        message.newVariable<float>("exit1_y");
        message.newVariable<float>("exit2_x");
        message.newVariable<float>("exit2_y");
        message.newVariable<float>("exit3_x");
        message.newVariable<float>("exit3_y");
        message.newVariable<float>("exit4_x");
        message.newVariable<float>("exit4_y");
        message.newVariable<float>("exit5_x");
        message.newVariable<float>("exit5_y");
        message.newVariable<float>("exit6_x");
        message.newVariable<float>("exit6_y");
    }
    {   // Pedestrian agent
        flamegpu::AgentDescription agent = model.newAgent("agent");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("velx", 0.0f);
        agent.newVariable<float>("vely", 0.0f);
        agent.newVariable<float>("steer_x", 0.0f);
        agent.newVariable<float>("steer_y", 0.0f);
        agent.newVariable<float>("height");
        agent.newVariable<int>("exit_no");
        agent.newVariable<float>("speed");
        // rendering and animation variables
        agent.newVariable<int>("lod", 1);
        agent.newVariable<float>("animate");
        agent.newVariable<int>("animate_dir", 1);

        agent.newFunction("output_pedestrian_location", output_pedestrian_location).setMessageOutput("pedestrian_location");
        agent.newFunction("avoid_pedestrians", avoid_pedestrians).setMessageInput("pedestrian_location");
        flamegpu::AgentFunctionDescription ff_fn = agent.newFunction("force_flow", force_flow);
        ff_fn.setMessageInput("navmap_cell");
        ff_fn.setAllowAgentDeath(true);
        agent.newFunction("move", move);
    }
    {   // Navmap agent
        flamegpu::AgentDescription navmap = model.newAgent("navmap");
        navmap.newVariable<int>("x");
        navmap.newVariable<int>("y");
        navmap.newVariable<int>("exit_no");
        navmap.newVariable<float>("height");
        navmap.newVariable<float>("collision_x");
        navmap.newVariable<float>("collision_y");
        navmap.newVariable<float>("exit0_x");
        navmap.newVariable<float>("exit0_y");
        navmap.newVariable<float>("exit1_x");
        navmap.newVariable<float>("exit1_y");
        navmap.newVariable<float>("exit2_x");
        navmap.newVariable<float>("exit2_y");
        navmap.newVariable<float>("exit3_x");
        navmap.newVariable<float>("exit3_y");
        navmap.newVariable<float>("exit4_x");
        navmap.newVariable<float>("exit4_y");
        navmap.newVariable<float>("exit5_x");
        navmap.newVariable<float>("exit5_y");
        navmap.newVariable<float>("exit6_x");
        navmap.newVariable<float>("exit6_y");
#ifdef FLAMEGPU_VISUALISATION
        // Extra vars, not present in FLAMEGPU 1 version are required to visualise with stock visualiser
        navmap.newVariable<float>("x_vis");
        navmap.newVariable<float>("y_vis");
        navmap.newVariable<float>("z_vis");
        navmap.newVariable<int>("is_active");
#endif

        navmap.newFunction("output_navmap_cells", output_navmap_cells).setMessageOutput("navmap_cell");
        flamegpu::AgentFunctionDescription gp_fn = navmap.newFunction("generate_pedestrians", generate_pedestrians);
        gp_fn.setAgentOutput("agent");
        gp_fn.setMessageInput("navmap_cell");  // Don't actually read these messages, just makes some things simpler
    }

    /**
     * GLOBALS
     */
    {
        flamegpu::EnvironmentDescription env = model.Environment();
        env.newProperty<float>("EMMISION_RATE_EXIT1", 10.0f);
        env.newProperty<float>("EMMISION_RATE_EXIT2", 10.0f);
        env.newProperty<float>("EMMISION_RATE_EXIT3", 10.0f);
        env.newProperty<float>("EMMISION_RATE_EXIT4", 10.0f);
        env.newProperty<float>("EMMISION_RATE_EXIT5", 10.0f);
        env.newProperty<float>("EMMISION_RATE_EXIT6", 10.0f);
        env.newProperty<float>("EMMISION_RATE_EXIT7", 10.0f);
        env.newProperty<int>("EXIT1_PROBABILITY", 1);
        env.newProperty<int>("EXIT2_PROBABILITY", 1);
        env.newProperty<int>("EXIT3_PROBABILITY", 1);
        env.newProperty<int>("EXIT4_PROBABILITY", 1);
        env.newProperty<int>("EXIT5_PROBABILITY", 3);
        env.newProperty<int>("EXIT6_PROBABILITY", 2);
        env.newProperty<int>("EXIT7_PROBABILITY", 1);
        env.newProperty<int>("EXIT1_STATE", 1);
        env.newProperty<int>("EXIT2_STATE", 1);
        env.newProperty<int>("EXIT3_STATE", 1);
        env.newProperty<int>("EXIT4_STATE", 1);
        env.newProperty<int>("EXIT5_STATE", 1);
        env.newProperty<int>("EXIT6_STATE", 1);
        env.newProperty<int>("EXIT7_STATE", 1);
        env.newProperty<int>("EXIT1_CELL_COUNT", 16);
        env.newProperty<int>("EXIT2_CELL_COUNT", 26);
        env.newProperty<int>("EXIT3_CELL_COUNT", 31);
        env.newProperty<int>("EXIT4_CELL_COUNT", 24);
        env.newProperty<int>("EXIT5_CELL_COUNT", 20);
        env.newProperty<int>("EXIT6_CELL_COUNT", 66);
        env.newProperty<int>("EXIT7_CELL_COUNT", 120);
        env.newProperty<float>("TIME_SCALER", 0.0003f);
        env.newProperty<float>("STEER_WEIGHT", 0.1f);
        env.newProperty<float>("AVOID_WEIGHT", 0.02f);
        env.newProperty<float>("COLLISION_WEIGHT", 0.5f);
        env.newProperty<float>("GOAL_WEIGHT", 0.2f);
    }

    /**
     * Control flow
     */
    {   // Layer #1
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(generate_pedestrians);
    }
    {   // Layer #2
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(output_pedestrian_location);
        layer.addAgentFunction(output_navmap_cells);
    }
    {   // Layer #3
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(avoid_pedestrians);
    }
    {   // Layer #4
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(force_flow);
    }
    {   // Layer #5
        flamegpu::LayerDescription layer = model.newLayer();
        layer.addAgentFunction(move);
    }

    /**
     * Create Model Runner
     */
    flamegpu::CUDASimulation  cudaSimulation(model, argc, argv);

    /**
     * Create visualisation
     */
#ifdef FLAMEGPU_VISUALISATION
    flamegpu::visualiser::ModelVis  m_vis = cudaSimulation.getVisualisation();
    {
        m_vis.setInitialCameraLocation(0.873f, 1.740f, 0.800f);
        m_vis.setInitialCameraTarget(0.873f - 0.489f, 1.740f - 0.741f, 0.800f - 0.459f);
        m_vis.setSimulationSpeed(50);
        m_vis.setCameraSpeed(0.0005f);
        m_vis.setViewClips(0.00001f, 5.0f);  // Model environment is in the range [-1.0f, 1.0f]
        {
            auto pedestrian = m_vis.addAgent("agent");
            // Position vars are named x, y; so they are used by default
            pedestrian.setYVariable("height");
            pedestrian.setZVariable("y");
            pedestrian.setForwardXVariable("velx");
            pedestrian.setForwardZVariable("vely");
            pedestrian.setKeyFrameModel(flamegpu::visualiser::Stock::Models::PEDESTRIAN, "animate");
            pedestrian.setModelScale(2.0f * 2.0f / 256.0f);  // TBC
            // Colour agents based on their targeted exit
            pedestrian.setColor(flamegpu::visualiser::DiscreteColor<int>("exit_no", flamegpu::visualiser::Stock::Palettes::DARK2, 1));
        }
        {
            auto navmap = m_vis.addAgent("navmap");
            navmap.clearYVariable();
            navmap.setXVariable("x_vis");  // Can't vis int vars
            navmap.setYVariable("y_vis");  // Want to offset height
            navmap.setZVariable("z_vis");  // Can't vis int vars
            navmap.setForwardXVariable("collision_x");
            navmap.setForwardZVariable("collision_y");
            navmap.setModel(flamegpu::visualiser::Stock::Models::ARROWHEAD);
            navmap.setModelScale(0.8f * 2.0f / 256.0f);  // TBC
            flamegpu::visualiser::DiscreteColor<int> nav_color("is_active", flamegpu::visualiser::Stock::Colors::BLACK);
            nav_color[1] = flamegpu::visualiser::Stock::Colors::RED;
            navmap.setColor(nav_color);
        }
        // Render the Subdivision of the navigation grid
        {
            const float RADIUS = (ENV_MAX - ENV_MIN) / 256.0f;
            auto pen = m_vis.newLineSketch(1.0f, 1.0f, 1.0f, 0.2f);  // white
            // auto pen = m_vis.newLineSketch(0.0f, 0.0f, 0.0f, 0.2f);  // black
            // Grid Lines
            for (float i = ENV_MIN; i <= ENV_MAX; i += RADIUS) {
                pen.addVertex(i, 1.0f, ENV_MIN);
                pen.addVertex(i, 1.0f, ENV_MAX);
                pen.addVertex(ENV_MIN, 1.0f, i);
                pen.addVertex(ENV_MAX, 1.0f, i);
            }
        }
        // Specify a UI for changing settings
        {
            auto ui1 = m_vis.newUIPanel("Exit Settings");
            ui1.newSection("Exit Emission Rate");
            // Global emission rate (Nope)
            ui1.newEnvironmentPropertySlider<float>("EMMISION_RATE_EXIT1", 0.0f, 100.0f);
            ui1.newEnvironmentPropertySlider<float>("EMMISION_RATE_EXIT2", 0.0f, 100.0f);
            ui1.newEnvironmentPropertySlider<float>("EMMISION_RATE_EXIT3", 0.0f, 100.0f);
            ui1.newEnvironmentPropertySlider<float>("EMMISION_RATE_EXIT4", 0.0f, 100.0f);
            ui1.newEnvironmentPropertySlider<float>("EMMISION_RATE_EXIT5", 0.0f, 100.0f);
            ui1.newEnvironmentPropertySlider<float>("EMMISION_RATE_EXIT6", 0.0f, 100.0f);
            ui1.newEnvironmentPropertySlider<float>("EMMISION_RATE_EXIT7", 0.0f, 100.0f);
            ui1.newEndSection();
            ui1.newSection("Target Exit Chance");
            ui1.newEnvironmentPropertyInput<int>("EXIT1_PROBABILITY", 1, 2);
            ui1.newEnvironmentPropertyInput<int>("EXIT2_PROBABILITY", 1, 2);
            ui1.newEnvironmentPropertyInput<int>("EXIT3_PROBABILITY", 1, 2);
            ui1.newEnvironmentPropertyInput<int>("EXIT4_PROBABILITY", 1, 2);
            ui1.newEnvironmentPropertyInput<int>("EXIT5_PROBABILITY", 1, 2);
            ui1.newEnvironmentPropertyInput<int>("EXIT6_PROBABILITY", 1, 2);
            ui1.newEnvironmentPropertyInput<int>("EXIT7_PROBABILITY", 1, 2);
            ui1.newEndSection();
            ui1.newSection("Exit Open");
            ui1.newEnvironmentPropertyToggle<int>("EXIT1_STATE");
            ui1.newEnvironmentPropertyToggle<int>("EXIT2_STATE");
            ui1.newEnvironmentPropertyToggle<int>("EXIT3_STATE");
            ui1.newEnvironmentPropertyToggle<int>("EXIT4_STATE");
            ui1.newEnvironmentPropertyToggle<int>("EXIT5_STATE");
            ui1.newEnvironmentPropertyToggle<int>("EXIT6_STATE");
            ui1.newEnvironmentPropertyToggle<int>("EXIT7_STATE");
            ui1.newEndSection();
            auto ui2 = m_vis.newUIPanel("Movement Settings");
            ui2.newEnvironmentPropertySlider<float>("TIME_SCALER", 0.0001f, 0.001f);
            ui2.newEnvironmentPropertySlider<float>("STEER_WEIGHT", 0.0f, 1.f);
            ui2.newEnvironmentPropertySlider<float>("AVOID_WEIGHT", 0.0f, 1.f);
            ui2.newEnvironmentPropertySlider<float>("COLLISION_WEIGHT", 0.0f, 1.f);
            ui2.newEnvironmentPropertySlider<float>("GOAL_WEIGHT", 0.0f, 1.f);
        }
    }
    m_vis.activate();
#endif
    /**
     * Initialisation
     */
    // Model must initialise from file to load navigation mesh

    /**
     * Execution
     */
    cudaSimulation.simulate();

#ifdef FLAMEGPU_VISUALISATION
    m_vis.join();
#endif
    return 0;
}
