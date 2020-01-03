#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"
#include "flamegpu/io/factory.h"

FLAMEGPU_AGENT_FUNCTION(output_message) {
    FLAMEGPU->addMessage<int>("id", FLAMEGPU->getVariable<float>("id"));
    FLAMEGPU->addMessage<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->addMessage<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->addMessage<float>("z", FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(move) {
    const int ID = FLAMEGPU->getVariable<int>("id");
    const float REPULSE_FACTOR = FLAMEGPU->environment.get<float>("repulse");
    const float RADIUS = FLAMEGPU->environment.get<float>("radius");

    float fx = 0.0;
    float fy = 0.0;
    float fz = 0.0;
    const float x1 = FLAMEGPU->getVariable<float>("x");
    const float y1 = FLAMEGPU->getVariable<float>("y");
    const float z1 = FLAMEGPU->getVariable<float>("z");
    int count = 0;
    for (auto &message : FLAMEGPU->GetMessageIterator("location")) {
        if (message.getVariable<int>("id") != ID) {
            const float x2 = message.getVariable<float>("x");
            const float y2 = message.getVariable<float>("y");
            const float z2 = message.getVariable<float>("z");
            float x21 = x2 - x1;
            float y21 = y2 - y1;
            float z21 = z2 - z1;
            float separation = sqrt(x21*x21 + y21*y21 + z21*z21);
            if (separation < RADIUS && separation > 0.0f) {
                float k = sinf((separation / RADIUS)*3.141*-2)*REPULSE_FACTOR;
                // Normalise without recalulating separation
                x21 /= separation;
                y21 /= separation;
                z21 /= separation;
                fx += k * x21;
                fy += k * y21;
                fz += k * z21;
            }
        }
    }
    fx /= count > 0 ? count : 1;
    fy /= count > 0 ? count : 1;
    fz /= count > 0 ? count : 1;
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("z", z1 + fz);
    return ALIVE;
}

int main(int argc, const char ** argv) {
    ModelDescription model("Circles_BruteForce_example");

    {  // Location message
        MessageDescription &message = model.newMessage("location");
        message.newVariable<int>("id");
        message.newVariable<float>("x");
        message.newVariable<float>("y");
        message.newVariable<float>("z");
    }
    {  // Circle agent
        AgentDescription &agent = model.newAgent("Circle");
        agent.newVariable<int>("id");
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
        agent.newVariable<float>("z");
        agent.newFunction("output_message", output_message).setMessageOutput("location");
        agent.newFunction("move", move).setMessageInput("location");
    }


    /**
     * GLOBALS
     */
    {
        EnvironmentDescription &env = model.Environment();
        env.add("repulse", 0.05f);
        env.add("radius", 1.0f);
    }

    /**
     * Control flow
     */     
     {  // Attach init/step/exit functions and exit condition
        // Model has none
     }

     {  // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(output_message);
     }
     {  // Layer #2
         LayerDescription &layer = model.newLayer();
         layer.addAgentFunction(move);
     }

    /**
     * Initialisation
     */
    // Currently not init (should init from XML file)
    // AgentPopulation population(model.Agent("circle"), AGENT_COUNT);
    // for (unsigned int i = 0; i < AGENT_COUNT; i++) {
    //     AgentInstance instance = population.getNextInstance();
    //    instance.setVariable<float>("x", static_cast<float>(i));
    //     instance.setVariable<int>("a", i % 2 == 0 ? 1 : 0);
    // }

    /**
     * Execution
     */
    CUDAAgentModel cuda_model(model);
    cuda_model.initialise(argc, argv);
    // cuda_model.setPopulationData(population);
    cuda_model.simulate();

    /**
     * Export Pop
     */
    // Based on Simulation::output() // That can't currently be called
    std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    auto a = std::make_shared<AgentPopulation>(model.getAgent("Circle"));  // Not sure if this workls, due to copy construction
    cuda_model.getPopulationData(*a);
    pops.emplace("Circle", a);
    StateWriter *write__ = WriterFactory::createWriter(pops, "end.xml");  // TODO (pair model format with its data?)
    write__->writeStates();

    getchar();
    return 0;
}
