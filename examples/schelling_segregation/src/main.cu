#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>

#include "flamegpu/flamegpu.h"

// Configurable properties
constexpr unsigned int GRID_WIDTH = 50u;
constexpr float THRESHOLD = 0.5;


enum class AGENT_TYPE
{
    A,
    B,
    UNOCCUPIED
};

int32_t as_int(AGENT_TYPE type) {
    switch(type) {
        case AGENT_TYPE::A:
           return 0;  
           
        case AGENT_TYPE::B:
            return 1;

        case AGENT_TYPE::UNOCCUPIED:
            return 2;
    }
    
    return -1;
}




FLAMEGPU_AGENT_FUNCTION(do_nothing, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

// Agents output their type
FLAMEGPU_AGENT_FUNCTION(output_type, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<int>("type", FLAMEGPU->getVariable<int>("type"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}

// Agents decide whether they are happy or not.
FLAMEGPU_AGENT_FUNCTION(is_happy, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    unsigned int same_type_neighbours = 0;
    unsigned int diff_type_neighbours = 0;
    // Iterate 3x3 Moore neighbourhood (this does not include the central cell)
    int my_type = FLAMEGPU->getVariable<int>("type");
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y)) {
        int message_type = message.getVariable<int>("type");
        same_type_neighbours += my_type == message_type;
        diff_type_neighbours += (my_type != message_type) && (message_type != 2);
    }

    FLAMEGPU->setVariable<int>("is_happy", ((float)same_type_neighbours / (same_type_neighbours + diff_type_neighbours)) > THRESHOLD);
    if (FLAMEGPU->getVariable<int>("type") == 2) {
        FLAMEGPU->setVariable<int>("is_happy", 2);
    }
    return flamegpu::ALIVE;
}

// Returns true for unhappy and unoccupied agents
FLAMEGPU_AGENT_FUNCTION_CONDITION(is_available) {
    int happy = FLAMEGPU->getVariable<int>("is_happy");
    return (happy == 0) || (happy == 2);
}

// Unhappy and unoccupied agents output a message to signify their location is available to be moved to
FLAMEGPU_AGENT_FUNCTION(output_available_locations, flamegpu::MessageNone, flamegpu::MessageArray) {
    int tid = blockIdx.x *blockDim.x + threadIdx.x;
    FLAMEGPU->message_out.setVariable<unsigned int>("x", FLAMEGPU->getVariable<unsigned int, 2>("pos", 0));
    FLAMEGPU->message_out.setVariable<unsigned int>("y", FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    FLAMEGPU->message_out.setIndex(tid);
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION_CONDITION(is_unhappy) {
    return FLAMEGPU->getVariable<int>("is_happy") == 0;
}

// Unhappy agents select new locations and output an intent to move, clear current location
FLAMEGPU_AGENT_FUNCTION(move_if_unhappy, flamegpu::MessageArray, flamegpu::MessageArray2D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto& message = FLAMEGPU->message_in.at(tid);
    FLAMEGPU->message_out.setIndex(message.getVariable<unsigned int>("x"), message.getVariable<unsigned int>("y"));
    FLAMEGPU->message_out.setVariable<int>("type", FLAMEGPU->getVariable<int>("type"));
    FLAMEGPU->message_out.setVariable<int>("is_real_message", 1);
    FLAMEGPU->setVariable<int>("type", 2);
    return flamegpu::ALIVE;
}

// Agents move
FLAMEGPU_AGENT_FUNCTION(process_moves, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const auto& message = FLAMEGPU->message_in.at(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    if (message.getVariable<int>("is_real_message")) {
        FLAMEGPU->setVariable<int>("type", message.getVariable<int>("type"));
    }
    return flamegpu::ALIVE;
}

int main(int argc, const char ** argv) {
    NVTX_RANGE("main");
    NVTX_PUSH("ModelDescription");

    flamegpu::ModelDescription model("Schelling_segregation");
    
    /**
     * Messages
     */
    {
        flamegpu::MessageArray2D::Description &message = model.newMessage<flamegpu::MessageArray2D>("type_message");
        message.newVariable<int>("type");
        message.setDimensions(GRID_WIDTH, GRID_WIDTH);
        
        flamegpu::MessageArray::Description &locationAvailableMessage = model.newMessage<flamegpu::MessageArray>("location_available_message");
        locationAvailableMessage.newVariable<unsigned int>("x");
        locationAvailableMessage.newVariable<unsigned int>("y");
        locationAvailableMessage.setLength(GRID_WIDTH * GRID_WIDTH);
        
        flamegpu::MessageArray2D::Description& moveIntentMessage = model.newMessage<flamegpu::MessageArray2D>("move_intent_message");
        moveIntentMessage.newVariable<int>("type");
        moveIntentMessage.newVariable<int>("is_real_message");
        moveIntentMessage.setDimensions(GRID_WIDTH, GRID_WIDTH);
    }
    /**
     * Agents
     */
    {   // Per cell agent
        flamegpu::AgentDescription  &agent = model.newAgent("agent");
        agent.newVariable<unsigned int, 2>("pos");
        agent.newVariable<int>("agent_id");
        agent.newVariable<int>("type");
        agent.newVariable<int>("is_happy");
#ifdef VISUALISATION
        // Redundant seperate floating point position vars for vis
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
#endif
        // Functions
        agent.newFunction("output_type", output_type).setMessageOutput("type_message");
        
        agent.newFunction("is_happy", is_happy).setMessageInput("type_message");
        flamegpu::AgentFunctionDescription& outputAvailableLocations = agent.newFunction("output_available_locations", output_available_locations);
        outputAvailableLocations.setMessageOutput("location_available_message");
        outputAvailableLocations.setFunctionCondition(is_available);
        
        flamegpu::AgentFunctionDescription& moveIfUnhappy = agent.newFunction("move_if_unhappy", move_if_unhappy);
        moveIfUnhappy.setMessageInput("location_available_message");
        moveIfUnhappy.setMessageOutput("move_intent_message");
        moveIfUnhappy.setFunctionCondition(is_unhappy);
        
        agent.newFunction("process_moves", process_moves).setMessageInput("move_intent_message");

    }

    /**
     * Control flow
     */
    {   // Layer #1
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(output_type);
    }
    {   // Layer #2
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(is_happy);
    }
    {
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(output_available_locations);
    }
    {
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(move_if_unhappy);
    }
    {
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(process_moves);
    }
    NVTX_POP();

    /**
     * Create Model Runner
     */
    NVTX_PUSH("CUDAAgentModel creation");
    flamegpu::CUDASimulation  cudaSimulation(model);
    NVTX_POP();

    /**
     * Create visualisation
     * @note FLAMEGPU2 doesn't currently have proper support for discrete/2d visualisations
     */
#ifdef VISUALISATION
    flamegpu::visualiser::ModelVis  &visualisation = cudaSimulation.getVisualisation();
    {
        visualisation.setSimulationSpeed(2);
        visualisation.setInitialCameraLocation(GRID_WIDTH / 2.0f, GRID_WIDTH / 2.0f, 225.0f);
        visualisation.setInitialCameraTarget(GRID_WIDTH / 2.0f, GRID_WIDTH /2.0f, 0.0f);
        visualisation.setCameraSpeed(0.001f * GRID_WIDTH);
        visualisation.setViewClips(0.1f, 5000);
        auto &agt = visualisation.addAgent("agent");
        // Position vars are named x, y, z; so they are used by default
        agt.setModel(flamegpu::visualiser::Stock::Models::CUBE);  // 5 unwanted faces!
        agt.setModelScale(1.0f);

        flamegpu::visualiser::DiscreteColor<int> cell_colors = flamegpu::visualiser::DiscreteColor<int>("type", flamegpu::visualiser::Color{"#666"});
        cell_colors[as_int(AGENT_TYPE::A)] = flamegpu::visualiser::Stock::Colors::RED;
        cell_colors[as_int(AGENT_TYPE::B)] = flamegpu::visualiser::Stock::Colors::BLUE;
        agt.setColor(cell_colors);
    }
    visualisation.activate();
#endif

    /**
     * Initialisation
     */
    NVTX_PUSH("CUDAAgentModel initialisation");
    cudaSimulation.initialise(argc, argv);
    if (cudaSimulation.getSimulationConfig().input_file.empty()) {
        std::default_random_engine rng;

        // Currently population has not been init, so generate an agent population on the fly
        const unsigned int CELL_COUNT = GRID_WIDTH * GRID_WIDTH;
        std::uniform_real_distribution<float> normal(0, 1);
        unsigned int i = 0;
        unsigned int agent_id = 0;
        flamegpu::AgentVector init_pop(model.Agent("agent"), CELL_COUNT);
        for (unsigned int x = 0; x < GRID_WIDTH; ++x) {
            for (unsigned int y = 0; y < GRID_WIDTH; ++y) {
                flamegpu::AgentVector::Agent instance = init_pop[i++];
                instance.setVariable<unsigned int, 2>("pos", { x, y });
                // TODO: How should these values be init?
                // agent specific variables
                // 10% chance of cell holding an agent
                if (normal(rng) < 0.8) {
                    instance.setVariable<int>("agent_id", agent_id++);
                    unsigned int type = normal(rng) < 0.5 ? as_int(AGENT_TYPE::A): as_int(AGENT_TYPE::B);
                    instance.setVariable<int>("type", type);
                    instance.setVariable<int>("is_happy", 0);
                } else {
                    instance.setVariable<int>("agent_id", -1);
                    instance.setVariable<int>("type", as_int(AGENT_TYPE::UNOCCUPIED));
                    instance.setVariable<int>("is_happy", 2);
                }
#ifdef VISUALISATION
                // Redundant separate floating point position vars for vis
                instance.setVariable<float>("x", static_cast<float>(x));
                instance.setVariable<float>("y", static_cast<float>(y));
#endif
            }
        }
        cudaSimulation.setPopulationData(init_pop);
    }
    NVTX_POP();

    /**
     * Execution
     */
    cudaSimulation.simulate();

    /**
     * Export Pop
     */
    // cudaSimulation.exportData("end.xml");

#ifdef VISUALISATION
    visualisation.join();
#endif
    return 0;
}
