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
            const float separation = cbrt(x21*x21 + y21*y21 + z21*z21);
            if (separation < RADIUS && separation > 0.0f) {
                float k = sinf((separation / RADIUS)*3.141*-2)*REPULSE_FACTOR;
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
    if(blockIdx.x == 0 && threadIdx.x == 4) {
        printf("(%f, %f, %f)(%f, %f, %f)(%f, %f, %f)\n", x1, y1, z1, fx, fy, fz, x1 + fx, y1 + fy, z1 + fz);
    }
    FLAMEGPU->setVariable<float>("x", x1 + fx);
    FLAMEGPU->setVariable<float>("y", y1 + fy);
    FLAMEGPU->setVariable<float>("z", z1 + fz);
    FLAMEGPU->setVariable<float>("drift", sqrt(fx*fx + fy*fy + fz*fz));
    return ALIVE;
}
FLAMEGPU_STEP_FUNCTION(Validation) {
    // This value should decline? as the model moves towards a steady equlibrium state
    // Once an equilibrium state is reached, it is likely to oscillate between 2-4? values
    float totalDrift = FLAMEGPU->agent("Circle").sum<float>("drift");
    //printf("Drift: %g\n", totalDrift);
}
void export_data(std::shared_ptr<AgentPopulation> pop, const char *filename);
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
        agent.newVariable<float>("drift");  // Store the distance moved here, for validation
        agent.newFunction("output_message", output_message).setMessageOutput("location");
        agent.newFunction("move", move).setMessageInput("location");
    }


    /**
     * GLOBALS
     */
    {
        EnvironmentDescription &env = model.Environment();
        env.add("repulse", 0.05f);
        env.add("radius", 2.5f);
    }

    /**
     * Control flow
     */     
     {  // Attach init/step/exit functions and exit condition
        model.addStepFunction(Validation);
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
    std::default_random_engine rng;
    std::uniform_real_distribution<float> dist(0.0, 10.0);
    const unsigned int AGENT_COUNT = 1024;
    AgentPopulation population(model.Agent("Circle"), AGENT_COUNT);
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<int>("id", i);
        instance.setVariable<float>("x", dist(rng));
        instance.setVariable<float>("y", dist(rng));
        instance.setVariable<float>("z", dist(rng));
    }

    /**
     * Execution
     */
    CUDAAgentModel cuda_model(model);
    cuda_model.initialise(argc, argv);
    cuda_model.setPopulationData(population);
    while(cuda_model.getStepCounter() < cuda_model.getSimulationConfig().steps && cuda_model.step()) {
        std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
        auto a = std::make_shared<AgentPopulation>(model.getAgent("Circle"));
        cuda_model.getPopulationData(*a);
        export_data(a, (std::to_string(cuda_model.getStepCounter())+".bin").c_str());
    }
    
    // cuda_model.simulate();


    /**
     * Export Pop
     */
    // Based on Simulation::output() // That can't currently be called
    //std::unordered_map<std::string, std::shared_ptr<AgentPopulation>> pops;
    //auto a = std::make_shared<AgentPopulation>(model.getAgent("Circle"));  // Not sure if this workls, due to copy construction
    //cuda_model.getPopulationData(*a);
    //pops.emplace("Circle", a);
    //StateWriter *write__ = WriterFactory::createWriter(pops, cuda_model.getStepCounter(), "end.xml");  // TODO (pair model format with its data?)
    //write__->writeStates();
    //export_data(a, "test.bin");
    getchar();
    return 0;
}

#include <fstream>

void export_data(std::shared_ptr<AgentPopulation> pop, const char *filename) {
    // Basic binary export function, so that I can use the visualiser i made for kenneths model
    std::ofstream ofs;
    ofs.open(filename, std::ofstream::out | std::ofstream::trunc | std::ofstream::binary);
    if (ofs.is_open())
    {
        float garbage[6];  // Need to begin with 6 floats, doesn't matter what they are
        //Write data
        ofs.write((char*)garbage, sizeof(float)*6);
        ofs.write((char*)pop->getReadOnlyStateMemory().getReadOnlyMemoryVector("x").getReadOnlyDataPtr(), sizeof(float)*pop->getCurrentListSize());
        ofs.write((char*)pop->getReadOnlyStateMemory().getReadOnlyMemoryVector("y").getReadOnlyDataPtr(), sizeof(float)*pop->getCurrentListSize());
        ofs.write((char*)pop->getReadOnlyStateMemory().getReadOnlyMemoryVector("z").getReadOnlyDataPtr(), sizeof(float)*pop->getCurrentListSize());
        ofs.write((char*)pop->getReadOnlyStateMemory().getReadOnlyMemoryVector("drift").getReadOnlyDataPtr(), sizeof(float)*pop->getCurrentListSize());
        ofs.close();
    }
}