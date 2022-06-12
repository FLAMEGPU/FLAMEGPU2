#include "flamegpu/flamegpu.h"

FLAMEGPU_AGENT_FUNCTION(output_message, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    return flamegpu::ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(move, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const flamegpu::id_t ID = FLAMEGPU->getID();
    const float REPULSE_FACTOR = FLAMEGPU->environment.getProperty<float>("repulse");
    const float RADIUS = FLAMEGPU->message_in.radius();
    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    int count = 0;
    for (const auto &message : FLAMEGPU->message_in(x1, y1, z1)) {
        if (message.getVariable<flamegpu::id_t>("id") != ID) {
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            const float z2 = message.getVariable<float>("z");
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            float z21 = z2 - z1;
            const float separation = sqrtf(x21*x21 + y21*y21 + z21*z21);
            if (separation < RADIUS && separation > 0.0f) {
                float k = sinf((separation / RADIUS)*3.141f*-2)*REPULSE_FACTOR;
                // Normalise without recalculating separation
                x21 /= separation;
                y21 /= separation;
                z21 /= separation;
                fx += k * x21;
                fy += k * y21;
                fz += k * z21;
                count++;
            }
        }
    }
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    fz /= count > 0 ? count : 1;
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("z", z1 + fz);
    FLAMEGPU->setVariable<float>("drift", sqrtf(fx*fx + fy*fy + fz*fz));
    return flamegpu::ALIVE;
}
FLAMEGPU_STEP_FUNCTION(Validation) {
    static float prevTotalDrift = FLT_MAX;
    static unsigned int driftDropped = 0;
    static unsigned int driftIncreased = 0;
    // This value should decline? as the model moves towards a steady equlibrium state
    // Once an equilibrium state is reached, it is likely to oscillate between 2-4? values
    float totalDrift = FLAMEGPU->agent("Circle").sum<float>("drift");
    if (totalDrift <= prevTotalDrift)
        driftDropped++;
    else
        driftIncreased++;
    prevTotalDrift = totalDrift;
    // printf("Avg Drift: %g\n", totalDrift / FLAMEGPU->agent("Circle").count());
    printf("%.2f%% Drift correct\n", 100 * driftDropped / static_cast<float>(driftDropped + driftIncreased));

    // Change radius
    static float radius_swap = 4.0f;
    if (!((FLAMEGPU->getStepCounter()+1) % 200)) {
        auto msg = FLAMEGPU->message<flamegpu::MessageSpatial3D>("location");
        const float t = msg.getRadius();
        msg.setRadius(radius_swap);
        radius_swap = t;
    }
}
int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("Circles_Spatial3D_example");

    const unsigned int AGENT_COUNT = 16384;
    const float ENV_MAX = static_cast<float>(floor(cbrt(AGENT_COUNT)));
    const float RADIUS = 2.0f;
    {   // Location message
        flamegpu::MessageSpatial3D::Description &message = model.newMessage<flamegpu::MessageSpatial3D>("location");
        message.newVariable<flamegpu::id_t>("id");
        message.setRadius(RADIUS);
        message.setMin(0, 0, 0);
        message.setMax(ENV_MAX, ENV_MAX, ENV_MAX);
    }
    {   // Circle agent
        flamegpu::AgentDescription  &agent = model.newAgent("Circle");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newVariable<float>("drift");  // Store the distance moved here, for validation
        agent.newFunction("output_message", output_message).setMessageOutput("location");
        agent.newFunction("move", move).setMessageInput("location");
    }

    /**
     * GLOBALS
     */
    {
        flamegpu::EnvironmentDescription  &env = model.Environment();
        env.newProperty("repulse", 0.05f);
    }

    /**
     * Control flow
     */     
    {   // Attach init/step/exit functions and exit condition
        model.addStepFunction(Validation);
    }

    {   // Layer #1
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(output_message);
    }
    {   // Layer #2
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(move);
    }

    /**
     * Create Model Runner
     */
    flamegpu::CUDASimulation  cudaSimulation(model, argc, argv);

    /**
     * Create visualisation
     */
#ifdef VISUALISATION
    flamegpu::visualiser::ModelVis  &m_vis = cudaSimulation.getVisualisation();
    {
        const float INIT_CAM = ENV_MAX * 1.25F;
        m_vis.setInitialCameraLocation(INIT_CAM, INIT_CAM, INIT_CAM);
        m_vis.setCameraSpeed(0.01f);
        auto &circ_agt = m_vis.addAgent("Circle");
        // Position vars are named x, y, z; so they are used by default
        circ_agt.setModel(flamegpu::visualiser::Stock::Models::ICOSPHERE);
        circ_agt.setModelScale(1/10.0f);
        // Render the Subdivision of spatial messaging
        {
            const float ENV_MIN = 0;
            const int DIM = static_cast<int>(ceil((ENV_MAX - ENV_MIN) / RADIUS));  // Spatial partitioning scales up to fit none exact environments
            const float DIM_MAX = DIM * RADIUS;
            auto pen = m_vis.newLineSketch(1, 1, 1, 0.2f);  // white
            // X lines
            for (int y = 0; y <= DIM; y++) {
                for (int z = 0; z <= DIM; z++) {
                    pen.addVertex(ENV_MIN, y * RADIUS, z * RADIUS);
                    pen.addVertex(DIM_MAX, y * RADIUS, z * RADIUS);
                }
            }
            // Y axis
            for (int x = 0; x <= DIM; x++) {
                for (int z = 0; z <= DIM; z++) {
                    pen.addVertex(x * RADIUS, ENV_MIN, z * RADIUS);
                    pen.addVertex(x * RADIUS, DIM_MAX, z * RADIUS);
                }
            }
            // Z axis
            for (int x = 0; x <= DIM; x++) {
                for (int y = 0; y <= DIM; y++) {
                    pen.addVertex(x * RADIUS, y * RADIUS, ENV_MIN);
                    pen.addVertex(x * RADIUS, y * RADIUS, DIM_MAX);
                }
            }
        }
    }
    m_vis.activate();
#endif
    /**
     * Initialisation
     */
    if (cudaSimulation.getSimulationConfig().input_file.empty()) {
        // Currently population has not been init, so generate an agent population on the fly
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, ENV_MAX);
        flamegpu::AgentVector population(model.Agent("Circle"), AGENT_COUNT);
        for (unsigned int i = 0; i < AGENT_COUNT; i++) {
            flamegpu::AgentVector::Agent instance = population[i];
            instance.setVariable<float>("x", dist(rng));
            instance.setVariable<float>("y", dist(rng));
            instance.setVariable<float>("z", dist(rng));
        }
        cudaSimulation.setPopulationData(population);
    }

    /**
     * Execution
     */
    cudaSimulation.simulate();

    /**
     * Export Pop
     */
    cudaSimulation.exportData("end.xml");

#ifdef VISUALISATION
    m_vis.join();
#endif
    return 0;
}
