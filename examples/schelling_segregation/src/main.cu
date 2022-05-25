#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <array>

#include "flamegpu/flamegpu.h"

// Configurable properties
constexpr unsigned int GRID_WIDTH = 100;
constexpr float THRESHOLD = 0.70f;

constexpr unsigned int A = 0;
constexpr unsigned int B = 1;
constexpr unsigned int UNOCCUPIED = 2;

// Agents output their type
FLAMEGPU_AGENT_FUNCTION(output_type, flamegpu::MessageNone, flamegpu::MessageArray2D) {
    FLAMEGPU->message_out.setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("type"));
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getVariable<unsigned int, 2>("pos", 0), FLAMEGPU->getVariable<unsigned int, 2>("pos", 1));
    return flamegpu::ALIVE;
}

// Agents decide whether they are happy or not and whether or not their space is available
FLAMEGPU_AGENT_FUNCTION(determine_status, flamegpu::MessageArray2D, flamegpu::MessageNone) {
    const unsigned int my_x = FLAMEGPU->getVariable<unsigned int, 2>("pos", 0);
    const unsigned int my_y = FLAMEGPU->getVariable<unsigned int, 2>("pos", 1);

    unsigned int same_type_neighbours = 0;
    unsigned int diff_type_neighbours = 0;

    // Iterate 3x3 Moore neighbourhood (this does not include the central cell)
    unsigned int my_type = FLAMEGPU->getVariable<unsigned int>("type");
    for (auto &message : FLAMEGPU->message_in.wrap(my_x, my_y)) {
        int message_type = message.getVariable<unsigned int>("type");
        same_type_neighbours += my_type == message_type;
        diff_type_neighbours += (my_type != message_type) && (message_type != UNOCCUPIED);
    }

    int isHappy = (static_cast<float>(same_type_neighbours) / (same_type_neighbours + diff_type_neighbours)) > THRESHOLD;
    FLAMEGPU->setVariable<unsigned int>("happy", isHappy);
    unsigned int my_next_type = ((my_type != UNOCCUPIED) && isHappy) ? my_type : UNOCCUPIED;
    FLAMEGPU->setVariable<unsigned int>("next_type", my_next_type);
    FLAMEGPU->setVariable<unsigned int>("movement_resolved", (my_type == UNOCCUPIED) || isHappy);
    unsigned int my_availability = (my_type == UNOCCUPIED) || (isHappy == 0);
    FLAMEGPU->setVariable<unsigned int>("available", my_availability);

    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION_CONDITION(is_available) {
    return FLAMEGPU->getVariable<unsigned int>("available");
}
FLAMEGPU_AGENT_FUNCTION(output_available_locations, flamegpu::MessageNone, flamegpu::MessageArray) {
    FLAMEGPU->message_out.setIndex(FLAMEGPU->getThreadIndex());
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    return flamegpu::ALIVE;
}

FLAMEGPU_HOST_FUNCTION(count_available_spaces) {
    FLAMEGPU->environment.setProperty<unsigned int>("spaces_available", FLAMEGPU->agent("agent").count<unsigned int>("available", 1));
}

FLAMEGPU_AGENT_FUNCTION_CONDITION(is_moving) {
    bool movementResolved = FLAMEGPU->getVariable<unsigned int>("movement_resolved");
    return !movementResolved;
}
FLAMEGPU_AGENT_FUNCTION(bid_for_location, flamegpu::MessageArray, flamegpu::MessageBucket) {
    // Select a location
    unsigned int selected_index = FLAMEGPU->random.uniform<unsigned int>(0, FLAMEGPU->environment.getProperty<unsigned int>("spaces_available") - 1);

    // Get the location at that index
    const auto& message = FLAMEGPU->message_in.at(selected_index);
    const flamegpu::id_t selected_location = message.getVariable<flamegpu::id_t>("id");

    // Bid for that location
    FLAMEGPU->message_out.setKey(selected_location - 1);
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("type"));
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(select_winners, flamegpu::MessageBucket, flamegpu::MessageArray) {
    // First agent in the bucket wins
    for (const auto& message : FLAMEGPU->message_in(FLAMEGPU->getID() - 1)) {
        flamegpu::id_t winning_id = message.getVariable<flamegpu::id_t>("id");
        FLAMEGPU->setVariable<unsigned int>("next_type", message.getVariable<unsigned int>("type"));
        FLAMEGPU->setVariable<unsigned int>("available", 0);
        FLAMEGPU->message_out.setIndex(winning_id - 1);
        FLAMEGPU->message_out.setVariable<unsigned int>("won", 1);
        break;
    }
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(has_moved, flamegpu::MessageArray, flamegpu::MessageNone) {
    const auto& message = FLAMEGPU->message_in.at(FLAMEGPU->getID() - 1);
    if (message.getVariable<unsigned int>("won")) {
        FLAMEGPU->setVariable<unsigned int>("movement_resolved", 1);
    }
    return flamegpu::ALIVE;
}

FLAMEGPU_EXIT_CONDITION(movement_resolved) {
    return (FLAMEGPU->agent("agent").count<unsigned int>("movement_resolved", 0) == 0) ? flamegpu::EXIT : flamegpu::CONTINUE;
}

FLAMEGPU_AGENT_FUNCTION(update_locations, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<unsigned int>("type", FLAMEGPU->getVariable<unsigned int>("next_type"));
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
        {
            flamegpu::MessageArray2D::Description &message = model.newMessage<flamegpu::MessageArray2D>("type_message");
            message.newVariable<unsigned int>("type");
            message.setDimensions(GRID_WIDTH, GRID_WIDTH);
        }
    }

    /**
     * Agents
     */
    {   // Per cell agent
        flamegpu::AgentDescription  &agent = model.newAgent("agent");
        agent.newVariable<unsigned int, 2>("pos");
        agent.newVariable<unsigned int>("type");
        agent.newVariable<unsigned int>("next_type");
        agent.newVariable<unsigned int>("happy");
        agent.newVariable<unsigned int>("available");
        agent.newVariable<unsigned int>("movement_resolved");
#ifdef VISUALISATION
        // Redundant seperate floating point position vars for vis
        agent.newVariable<float>("x");
        agent.newVariable<float>("y");
#endif
        // Functions
        agent.newFunction("output_type", output_type).setMessageOutput("type_message");
        agent.newFunction("determine_status", determine_status).setMessageInput("type_message");
        agent.newFunction("update_locations", update_locations);
    }

    /**
     * Movement resolution submodel
     */

    flamegpu::ModelDescription submodel("plan_movement");
    submodel.addExitCondition(movement_resolved);
    {
        // Environment
        {
            flamegpu::EnvironmentDescription& env = submodel.Environment();
            env.newProperty<unsigned int>("spaces_available", 0);
        }

        // Message types
        {
            {
                flamegpu::MessageArray::Description &message = submodel.newMessage<flamegpu::MessageArray>("available_location_message");
                message.newVariable<flamegpu::id_t>("id");
                message.setLength(GRID_WIDTH*GRID_WIDTH);
            }
            {
                flamegpu::MessageBucket::Description &message = submodel.newMessage<flamegpu::MessageBucket>("intent_to_move_message");
                message.newVariable<flamegpu::id_t>("id");
                message.newVariable<unsigned int>("type");
                message.setBounds(0, GRID_WIDTH * GRID_WIDTH);
            }
            {
                flamegpu::MessageArray::Description &message = submodel.newMessage<flamegpu::MessageArray>("movement_won_message");
                message.newVariable<unsigned int>("won");
                message.setLength(GRID_WIDTH*GRID_WIDTH);
            }
        }

        // Agent types
        {
            flamegpu::AgentDescription  &agent = submodel.newAgent("agent");
            agent.newVariable<unsigned int, 2>("pos");
            agent.newVariable<unsigned int>("type");
            agent.newVariable<unsigned int>("next_type");
            agent.newVariable<unsigned int>("happy");
            agent.newVariable<unsigned int>("available");
            agent.newVariable<unsigned int>("movement_resolved");

            // Functions
            auto& outputLocationsFunction = agent.newFunction("output_available_locations", output_available_locations);
            outputLocationsFunction.setMessageOutput("available_location_message");
            outputLocationsFunction.setFunctionCondition(is_available);

            auto& bidFunction = agent.newFunction("bid_for_location", bid_for_location);
            bidFunction.setFunctionCondition(is_moving);
            bidFunction.setMessageInput("available_location_message");
            bidFunction.setMessageOutput("intent_to_move_message");

            auto& selectWinnersFunction = agent.newFunction("select_winners", select_winners);
            selectWinnersFunction.setMessageInput("intent_to_move_message");
            selectWinnersFunction.setMessageOutput("movement_won_message");
            selectWinnersFunction.setMessageOutputOptional(true);

            agent.newFunction("has_moved", has_moved).setMessageInput("movement_won_message");
        }
        // Control flow
        {
            // Available agents output their location (indexed by thread ID)
            {
                flamegpu::LayerDescription &layer = submodel.newLayer();
                layer.addAgentFunction(output_available_locations);
            }
            // Count the number of available spaces
            {
                flamegpu::LayerDescription &layer = submodel.newLayer();
                layer.addHostFunction(count_available_spaces);
            }
            // Unhappy agents bid for a new location
            {
                flamegpu::LayerDescription &layer = submodel.newLayer();
                layer.addAgentFunction(bid_for_location);
            }
            // Available locations check if anyone wants to move to them. If so, approve one and mark as unavailable
            // Update next type to the type of the mover
            // Output a message to inform the mover that they have been successful
            {
                flamegpu::LayerDescription &layer = submodel.newLayer();
                layer.addAgentFunction(select_winners);
            }
            // Movers mark themselves as resolved
            {
                flamegpu::LayerDescription &layer = submodel.newLayer();
                layer.addAgentFunction(has_moved);
            }
        }
    }
    flamegpu::SubModelDescription& plan_movement = model.newSubModel("plan_movement", submodel);
    {
        plan_movement.bindAgent("agent", "agent", true, true);
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
        layer.addAgentFunction(determine_status);
    }
    {
        flamegpu::LayerDescription &layer = model.newLayer();
        layer.addSubModel(plan_movement);
    }
    {
        flamegpu::LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(update_locations);
    }
    {   // Trying calling this again to fix vis
        flamegpu::LayerDescription  &layer = model.newLayer();
        layer.addAgentFunction(determine_status);
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

        flamegpu::visualiser::DiscreteColor<unsigned int> cell_colors = flamegpu::visualiser::DiscreteColor<unsigned int>("type", flamegpu::visualiser::Color{"#666"});
        cell_colors[A] = flamegpu::visualiser::Stock::Colors::RED;
        cell_colors[B] = flamegpu::visualiser::Stock::Colors::BLUE;
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
        flamegpu::AgentVector init_pop(model.Agent("agent"), CELL_COUNT);
        for (unsigned int x = 0; x < GRID_WIDTH; ++x) {
            for (unsigned int y = 0; y < GRID_WIDTH; ++y) {
                flamegpu::AgentVector::Agent instance = init_pop[i++];
                instance.setVariable<unsigned int, 2>("pos", { x, y });
                // Will this cell be occupied
                if (normal(rng) < 0.8) {
                    unsigned int type = normal(rng) < 0.5 ? A : B;
                    instance.setVariable<unsigned int>("type", type);
                    instance.setVariable<unsigned int>("happy", 0);
                } else {
                    instance.setVariable<unsigned int>("type", UNOCCUPIED);
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
