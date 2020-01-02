#include <iostream>
#include <cstdio>
#include <cstdlib>

#include "flamegpu/flame_api.h"
#include "flamegpu/runtime/flamegpu_api.h"

FLAMEGPU_AGENT_FUNCTION(outputdata) {
    FLAMEGPU->addMessage<int>("id", FLAMEGPU->getVariable<float>("id"));
    FLAMEGPU->addMessage<float>("x", FLAMEGPU->getVariable<float>("x"));
    FLAMEGPU->addMessage<float>("y", FLAMEGPU->getVariable<float>("y"));
    FLAMEGPU->addMessage<float>("z", FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(inputdata) {
    const int ID = FLAMEGPU->getVariable<int>("id");
    const float kr = 0.1f; /* Stiffness variable for repulsion */
    const float ka = 0.0f; /* Stiffness variable for attraction */
    const float RADIUS = 2.0f;

    float x1, y1, x2, y2, fx, fy;  // z1, z2, fz
    float location_distance, separation_distance;
    float k;
    x1 = FLAMEGPU->getVariable<float>("x");
    fx = 0.0;
    y1 = FLAMEGPU->getVariable<float>("y");
    fy = 0.0;
    // z1 = FLAMEGPU->getVariable<float>("z");
    // fz = 0.0;
    for (auto &message : FLAMEGPU->GetMessageIterator("location")) {
        if (message.getVariable<int>("id") != ID) {
            x2 = message.getVariable<float>("x");
            y2 = message.getVariable<float>("y");
            // z2 = message.getVariable<float>("z");
            // Deep (expensive) check
            location_distance = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2));
            separation_distance = (location_distance - RADIUS);
            if (separation_distance < RADIUS) {
                k = separation_distance > 0.0 ? ka : -kr;

                fx += k*(separation_distance)*((x1 - x2) / RADIUS);
                fy += k*(separation_distance)*((y1 - y2) / RADIUS);
            }
        }
    }
    FLAMEGPU->setVariable("fx", fx);
    FLAMEGPU->setVariable("fy", fy);
    // FLAMEGPU->setVariable("fz", fz);
    return ALIVE;
}
FLAMEGPU_AGENT_FUNCTION(move) {
    FLAMEGPU->setVariable<float>("x", FLAMEGPU->getVariable<float>("x") + FLAMEGPU->getVariable<float>("fx"));
    FLAMEGPU->setVariable<float>("y", FLAMEGPU->getVariable<float>("y") + FLAMEGPU->getVariable<float>("fy"));
    // FLAMEGPU->setVariable<float>("z", FLAMEGPU->getVariable<float>("z") + FLAMEGPU->getVariable<float>("fz"));
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
        agent.newVariable<float>("fx");
        agent.newVariable<float>("fy");
        // agent.newVariable<float>("fz");  // FGPU1 model is 2D, but has z component?
        agent.newFunction("outputdata", outputdata).setMessageOutput("location");
        agent.newFunction("inputdata", inputdata).setMessageInput("location");
        agent.newFunction("move", move);
    }


    /**
     * GLOBALS
     */
    {
        // EnvironmentDescription &envProperties = model.Environment();
        // Model has none
    }

    /**
     * Control flow
     */     
     {  // Attach init/step/exit functions and exit condition
        // Model has none
     }

     {  // Layer #1
        LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(outputdata);
     }
     {  // Layer #2
         LayerDescription &layer = model.newLayer();
         layer.addAgentFunction(inputdata);
     }
     {  // Layer #3
         LayerDescription &layer = model.newLayer();
         layer.addAgentFunction(move);
     }

    /**
     * Initialisation
     */
    // Currently not init (should init from XML file)
    // AgentPopulation population(model.Agent("agent"), AGENT_COUNT);
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

    getchar();
    return 0;
}
