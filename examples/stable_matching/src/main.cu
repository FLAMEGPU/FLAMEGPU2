#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <algorithm>
#include <numeric>

#include "flamegpu/flamegpu.h"

// @todo - support more than a single round via submodel once available
// @todo - make an issue about large agent array performance - interleaving would be better? Maybe only interleave for relatively larger array? > glm's max size?
// @todo - rename Agents/variables to a generic stable_matching problem rather than marriage

// Constants
#define NOT_ENGAGED UINT_MAX

// size of each population, which is related to the size of preference lists hence compile time definition.
#define POPULATION_SIZE 1024



FLAMEGPU_AGENT_FUNCTION_CONDITION(man_make_proposals_condition) {
    // If not provisionally engaged (but not actually engaged?)
    return FLAMEGPU->getVariable<flamegpu::id_t>("engaged_to") == NOT_ENGAGED;
}
FLAMEGPU_AGENT_FUNCTION(man_make_proposals, flamegpu::MessageNone, flamegpu::MessageBruteForce ) {
    uint32_t r = FLAMEGPU->getVariable<uint32_t>("round");

    // If the round is valid, select the next proposal and output a message.
    if (r < POPULATION_SIZE) {
        // Get the next preferable woman
        uint32_t woman = FLAMEGPU->getVariable<flamegpu::id_t, POPULATION_SIZE>("preferred_woman", r);

        // Make a proposal
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("from", FLAMEGPU->getID());
        FLAMEGPU->message_out.setVariable<flamegpu::id_t>("to", woman);

        // Update the round.
        FLAMEGPU->setVariable<flamegpu::id_t>("round", r);
    }
    return flamegpu::ALIVE;
}


FLAMEGPU_AGENT_FUNCTION(woman_check_proposals, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    flamegpu::id_t suitor = NOT_ENGAGED;
    uint32_t current_suitor_rank = FLAMEGPU->getVariable<uint32_t>("current_suitor_rank");
    flamegpu::id_t id = FLAMEGPU->getID();
    // Iterate proposals to find the best suitor so far for this round of proposals.
    for (auto &message : FLAMEGPU->message_in) {
        // If the message is for the current agent, proceed
        if (id == message.getVariable<flamegpu::id_t>("to")) {
            // Make sure the message is in bounds for the agent array
            const uint32_t from = message.getVariable<flamegpu::id_t>("from");
            if (from < POPULATION_SIZE) {
                // Find the authors rank
                const uint32_t rank = FLAMEGPU->getVariable<uint32_t, POPULATION_SIZE>("preferred_man", from);

                // If the rank is lower than current (or current rank is invalid, implied because UINT_MAX.)
                if (rank < current_suitor_rank) {
                    current_suitor_rank = rank;
                    suitor = from;
                }
            }
        }
    }
    // Update agent data
    if (suitor != NOT_ENGAGED) {
        FLAMEGPU->setVariable<flamegpu::id_t>("current_suitor", suitor);
        FLAMEGPU->setVariable<uint32_t>("current_suitor_rank", current_suitor_rank);
    }
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION_CONDITION(woman_notify_suitors_condition) {
    // Have a proposal to accept (i.e. provisional engagement)
    return FLAMEGPU->getVariable<uint32_t>("current_suitor") != NOT_ENGAGED;
}
FLAMEGPU_AGENT_FUNCTION(woman_notify_suitors, flamegpu::MessageNone, flamegpu::MessageBruteForce) {
    // Condition means this is only called if a suitor is found.
    // @todo - alternatively could just have in if statement in here / in the end of the prev function, skipping th eneed for 2 extra kernel launches...
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("from", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<flamegpu::id_t>("to", FLAMEGPU->getVariable<flamegpu::id_t>("current_suitor"));
    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(man_check_notifications, flamegpu::MessageBruteForce, flamegpu::MessageNone) {
    // Reset the engaged property
    flamegpu::id_t engaged_to = NOT_ENGAGED;

    flamegpu::id_t id = FLAMEGPU->getID();

    // Iterate messages looking for notifications of accepeted proposals.
    for (auto &message : FLAMEGPU->message_in) {
        // If this agent is the recipient
        if (id == message.getVariable<flamegpu::id_t>("to")) {
            // Store who it is from - There should only be one message per key.
            // @todo - optimise this to use a key-based message scheme with only a single message per key.
            engaged_to = message.getVariable<flamegpu::id_t>("from");
        }
    }

    // Update agent data in global memory.
    FLAMEGPU->setVariable<flamegpu::id_t>("engaged_to", engaged_to);
    return flamegpu::ALIVE;
}

FLAMEGPU_EXIT_CONDITION(check_resolved_exit_condition) {
    // If anyone is not engaged, keep going
    uint32_t unengaged_count = FLAMEGPU->agent("man", "unengaged").count<uint32_t>("engaged_to", NOT_ENGAGED);

    // uint32_t unengaged_wcount = FLAMEGPU->agent("woman", "default").count<uint32_t>("current_suitor", NOT_ENGAGED);
    // printf("Iter %u: %u unengaged men, %u unengaged women\n", FLAMEGPU->getStepCounter(), unengaged_count, unengaged_wcount);

    if (unengaged_count == 0) {
        return flamegpu::EXIT;
    }
    return flamegpu::CONTINUE;
}

// Exit function which will output the resolved status?
FLAMEGPU_EXIT_FUNCTION(exit_resolved_or_not) {
    uint32_t unengaged_count = FLAMEGPU->agent("man", "unengaged").count<uint32_t>("engaged_to", NOT_ENGAGED);

    // Output success / error message.
    if (unengaged_count == 0) {
        printf("Completed in %u iterations\n", FLAMEGPU->getStepCounter());
    } else {
        printf("%u remaining unengaged pairs after %u iterations\n", unengaged_count, FLAMEGPU->getStepCounter());
    }
}

int main(int argc, const char ** argv) {
    flamegpu::ModelDescription model("stable_matching");

    {   // proposal message list
        flamegpu::MessageBruteForce::Description &proposal_message = model.newMessage<flamegpu::MessageBruteForce>("proposal");
        proposal_message.newVariable<flamegpu::id_t>("from");
        proposal_message.newVariable<flamegpu::id_t>("to");
    }

    {   // notificaiton message list
        flamegpu::MessageBruteForce::Description &notification_message = model.newMessage<flamegpu::MessageBruteForce>("notification");
        notification_message.newVariable<flamegpu::id_t>("from");
        notification_message.newVariable<flamegpu::id_t>("to");
    }

    {   // Man agent
        flamegpu::AgentDescription &man_agent = model.newAgent("man");

        // Declare variables
        man_agent.newVariable<flamegpu::id_t>("engaged_to", NOT_ENGAGED);
        man_agent.newVariable<uint32_t>("round");
        man_agent.newVariable<flamegpu::id_t, POPULATION_SIZE>("preferred_woman");

        // Declare states
        man_agent.newState("engaged");
        man_agent.newState("unengaged");

        // make_proposals function.
        auto& make_proposals = man_agent.newFunction("make_proposals", man_make_proposals);
        make_proposals.setInitialState("unengaged");
        make_proposals.setEndState("unengaged");
        make_proposals.setFunctionCondition(man_make_proposals_condition);
        make_proposals.setMessageOutput("proposal");

        // Check notifcations from other population
        auto& check_notifications = man_agent.newFunction("check_notifications", man_check_notifications);
        check_notifications.setInitialState("unengaged");
        check_notifications.setEndState("unengaged");
        check_notifications.setMessageInput("notification");
    }

    {   // Woman agent
        flamegpu::AgentDescription &woman_agent = model.newAgent("woman");

        // Declare variables
        woman_agent.newVariable<flamegpu::id_t>("current_suitor", NOT_ENGAGED);
        woman_agent.newVariable<uint32_t>("current_suitor_rank", NOT_ENGAGED);
        woman_agent.newVariable<flamegpu::id_t, POPULATION_SIZE>("preferred_man");

        // Declare states
        woman_agent.newState("default");

        // make_proposals function.
        auto& check_proposals = woman_agent.newFunction("check_proposals", woman_check_proposals);
        check_proposals.setInitialState("default");
        check_proposals.setEndState("default");
        check_proposals.setMessageInput("proposal");

        // Check notifcations from other population
        auto& notify_suitors = woman_agent.newFunction("notify_suitors", woman_notify_suitors);
        notify_suitors.setInitialState("default");
        notify_suitors.setEndState("default");
        notify_suitors.setMessageOutput("notification");
        notify_suitors.setFunctionCondition(woman_notify_suitors_condition);
    }

    // Exit Condition and Function
    {
        model.addExitCondition(check_resolved_exit_condition);
        model.addExitFunction(exit_resolved_or_not);
    }


    /**
     * GLOBALS
     */
    {
        // EnvironmentDescription &env = model.Environment();
    }

    /**
     * Control flow
     */
    {
        flamegpu::LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(man_make_proposals);
    }
    {
        flamegpu::LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(woman_check_proposals);
    }
    {
        flamegpu::LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(woman_notify_suitors);
    }
    {
        flamegpu::LayerDescription &layer = model.newLayer();
        layer.addAgentFunction(man_check_notifications);
    }

    /**
     * Create Model Runner
     */
    flamegpu::CUDASimulation cuda_simulation(model);

    /**
     * Initialisation
     */
    cuda_simulation.initialise(argc, argv);
    // If no agents were provided from disk, dynamically create the populations.
    if (cuda_simulation.getSimulationConfig().input_file.empty()) {
        // Currently population has not been init, so generate an agent population on the fly.
        // There must be a balanced population of men and women.

        // Prepare an array of rank values, to be shuffled for each agent.
        std::array<uint32_t, POPULATION_SIZE> ranks;
        std::iota(ranks.begin(), ranks.end(), 0);

        const uint32_t seed = cuda_simulation.getSimulationConfig().random_seed;
        std::mt19937 rank_rng(seed);

        {
            flamegpu::AgentVector population(model.Agent("man"), POPULATION_SIZE);
            for (uint32_t idx = 0; idx < POPULATION_SIZE; idx++) {
                flamegpu::AgentVector::Agent instance = population[idx];
                instance.setVariable<uint32_t>("round", 0);
                instance.setVariable<flamegpu::id_t>("engaged_to", NOT_ENGAGED);

                // Shuffle the ranks
                std::shuffle(ranks.begin(), ranks.end(), rank_rng);
                instance.setVariable<flamegpu::id_t, POPULATION_SIZE>("preferred_woman", ranks);
            }
            cuda_simulation.setPopulationData(population, "unengaged");
        }

        {
            flamegpu::AgentVector population(model.Agent("woman"), POPULATION_SIZE);
            for (uint32_t idx = 0; idx < POPULATION_SIZE; idx++) {
                flamegpu::AgentVector::Agent instance = population[idx];
                instance.setVariable<flamegpu::id_t>("current_suitor", NOT_ENGAGED);
                instance.setVariable<uint32_t>("current_suitor_rank", NOT_ENGAGED);

                // Shuffle the ranks
                std::shuffle(ranks.begin(), ranks.end(), rank_rng);
                instance.setVariable<flamegpu::id_t, POPULATION_SIZE>("preferred_man", ranks);
            }
            cuda_simulation.setPopulationData(population, "default");
        }

    } else {
        // @todo - validate that there are balanced populations with unique IDs.
    }

    /**
     * Execution
     */
    cuda_simulation.simulate();

    /**
     * Export Pop
     */
    // TODO
    return 0;
}
