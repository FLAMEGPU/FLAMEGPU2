#include "flamegpu/flamegpu.h"
#include "flamegpu/detail/compute_capability.cuh"

#include "gtest/gtest.h"

namespace flamegpu {

// Tests for detecting concurrency within CUDASimulation.

// @todo - Get information about the current device, in order to (accurately) determine a sensible population size. Doing this accuratly without using the occupancy calculator for the kernel(s) might be a touch awkward.
// @todo - switch to per-layer timing - this might not actually be required if the test case has been constructed in a way that leads to net simulation step speedup.
// @todo - modify all tests to actually verify that agent functions executed? Probably just set an int which is incremented each time an agent executes a function, reduce and check for positive values.

// These tests are only meaningful in release builds (measure performance), so use a macro to disable them in non-release builds.
// @todo - expand RTC variant coverage, although this will dramatically increase runtime.


#ifndef _DEBUG
#define RELEASE_ONLY_TEST(TestSuiteName, TestName)\
    TEST(TestSuiteName, TestName)
#else
#define RELEASE_ONLY_TEST(TestSuiteName, TestName)\
    TEST(TestSuiteName, DISABLED_ ## TestName)
#endif

// if seatbelts and not debug, run the test, otherwise disable.
#if defined(FLAMEGPU_SEATBELTS) && FLAMEGPU_SEATBELTS && !defined(_DEBUG)
#define RELEASE_ONLY_SEATBELTS_TEST(TestSuiteName, TestName)\
    TEST(TestSuiteName, TestName)
#else
#define RELEASE_ONLY_SEATBELTS_TEST(TestSuiteName, TestName)\
    TEST(TestSuiteName, DISABLED_ ## TestName)
#endif


namespace test_cuda_simulation_concurrency {

// Threshold speedup to determine if concurrency was achieved.
const float SPEEDUP_THRESHOLD = 1.5;

// Number of repetitions to time, to improve accuracy of time evaluation. More is better (within reason)
const int TIMING_REPETITIONS = 3;

// Number of conccurent agent functions
const int CONCURRENCY_DEGREE = 4;

// Number of agents per population - i.e how many threads should be used per concurreny kernel.
// This needs to be sufficiently small that streams will actually be concurrent.
const unsigned int POPULATION_SIZES = 512;


/** 
 * Utility function to time N repetitions of a simulation, returning the mean (but skipping the first)
 */
float meanSimulationTime(const int REPETITIONS, CUDASimulation &s, std::vector<AgentVector *> const &populations) {
    double total_time = 0.f;
    for (int r = 0; r < REPETITIONS + 1; r++) {
        // re-set each population
        for (AgentVector* pop : populations) {
            s.setPopulationData(*pop);
        }
        // Run and time the simulation
        s.simulate();
        // Store the time if not the 0th rep of the model.
        if (r > 0) {
            total_time += s.getElapsedTimeSimulation();
        }
    }
    return static_cast<float>(total_time / REPETITIONS);
}

/** 
 * Utility function checking for a speedup after running a sim with and without concurrency.
 */
float concurrentLayerSpeedup(const int REPETITIONS, CUDASimulation &s, std::vector<AgentVector*> const &populations) {
    // Set a single step.
    s.SimulationConfig().steps = 1;

    // Set the flag saying don't use concurrency.
    s.CUDAConfig().inLayerConcurrency = false;
    s.applyConfig();
    EXPECT_EQ(s.CUDAConfig().inLayerConcurrency, false);

    // Time the simulation multiple times to get an average
    float mean_sequential_time = meanSimulationTime(REPETITIONS, s, populations);

    // set the flag saying to use streams for agnet function concurrency.
    s.CUDAConfig().inLayerConcurrency = true;
    s.applyConfig();
    EXPECT_EQ(s.CUDAConfig().inLayerConcurrency, true);

    float mean_concurrent_time = meanSimulationTime(REPETITIONS, s, populations);

    // Calculate a speedup value.
    float speedup = mean_sequential_time / mean_concurrent_time;
    return speedup;
}



/**
 * Slow and uninteresting agent function which will take a while to run for accurate timing
 */
FLAMEGPU_AGENT_FUNCTION(SlowAgentFunction, MessageNone, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    return ALIVE;
}

/**
 * Agent function which outputs to a message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowMessageOutputAgentFunction, MessageNone, MessageBruteForce) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    FLAMEGPU->message_out.setVariable("v", FLAMEGPU->getVariable<float>("v"));
    return ALIVE;
}

/**
 * Agent function which inputs from a message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowMessageInputAgentFunction, MessageBruteForce, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    float vSum = 0.f;
    for (const auto &message : FLAMEGPU->message_in) {
        vSum += message.getVariable<float>("v");
    }
    FLAMEGPU->setVariable("v", vSum);
    return ALIVE;
}

/**
 * Agent function which outputs to an optional message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowOptionalMessageOutputAgentFunction, MessageNone, MessageBruteForce) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    if (FLAMEGPU->getVariable<unsigned int>("id") % 2 == 0) {
        FLAMEGPU->message_out.setVariable("v", FLAMEGPU->getVariable<float>("v"));
    }
    return ALIVE;
}

/**
 * Agent function which inputs from an optional message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowOptionalMessageInputAgentFunction, MessageBruteForce, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    float vSum = 0.f;
    unsigned int count = 0;
    for (const auto &message : FLAMEGPU->message_in) {
        vSum += message.getVariable<float>("v");
        count += 1;
    }
    // if(FLAMEGPU->getVariable<unsigned int>("id") == 0) {
    //     printf("agent 0 read %u messages\n", count);
    // }
    FLAMEGPU->setVariable("v", vSum);
    return ALIVE;
}

/**
 * Agent function which outputs to a Spatial2D message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowMessageOutputAgentFunctionSpatial2D, MessageNone, MessageSpatial2D) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    FLAMEGPU->message_out.setVariable("v", FLAMEGPU->getVariable<float>("v"));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"));
    return ALIVE;
}

/**
 * Agent function which inputs from a Spatial2D message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowMessageInputAgentFunctionSpatial2D, MessageSpatial2D, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    float vSum = 0.f;
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y)) {
        vSum += message.getVariable<float>("v");
    }
    FLAMEGPU->setVariable("v", vSum);
    return ALIVE;
}

/**
 * Agent function which outputs to a Spatial3D message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowMessageOutputAgentFunctionSpatial3D, MessageNone, MessageSpatial3D) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    FLAMEGPU->message_out.setVariable("v", FLAMEGPU->getVariable<float>("v"));
    FLAMEGPU->message_out.setLocation(
        FLAMEGPU->getVariable<float>("x"),
        FLAMEGPU->getVariable<float>("y"),
        FLAMEGPU->getVariable<float>("z"));
    return ALIVE;
}

/**
 * Agent function which inputs from a Spatial3D message list + some slow work.
 */
FLAMEGPU_AGENT_FUNCTION(SlowMessageInputAgentFunctionSpatial3D, MessageSpatial3D, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    float vSum = 0.f;
    float agent_x = FLAMEGPU->getVariable<float>("x");
    float agent_y = FLAMEGPU->getVariable<float>("y");
    float agent_z = FLAMEGPU->getVariable<float>("z");
    for (const auto &message : FLAMEGPU->message_in(agent_x, agent_y, agent_z)) {
        vSum += message.getVariable<float>("v");
    }
    FLAMEGPU->setVariable("v", vSum);
    return ALIVE;
}

/**
 * Slow and uninteresting agent function which will take a while to run for accurate timing.
 * Agents birth an agent of the same type.
 */
FLAMEGPU_AGENT_FUNCTION(SlowAgentFunctionBirth, MessageNone, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    // A new agent is born.
    FLAMEGPU->agent_out.setVariable<float>("v", 1.0f);
    return ALIVE;
}

/**
 * Slow and uninteresting agent function which will take a while to run for accurate timing.
 * Agents die.
 */
FLAMEGPU_AGENT_FUNCTION(SlowAgentFunctionDeath, MessageNone, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    // Agents all die.
    return DEAD;
}

/**
 * Agent function condition which disabled all agents
 */
FLAMEGPU_AGENT_FUNCTION_CONDITION(SlowConditionAllFalse) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    float v = 1.f;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Can't write, so just read and increment locally
        v = v + FLAMEGPU->getVariable<float>("v");
    }
    // Use v in the return val so the loop doesn't get optimised out.
    return v < 0.f;
}


/**
 * Agent function condition which enables all agents
 */
FLAMEGPU_AGENT_FUNCTION_CONDITION(SlowConditionAllTrue) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    float v = 1.f;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Can't write, so just read and increment locally
        v = v + FLAMEGPU->getVariable<float>("v");
    }
    // Use v in the return val so the loop doesn't get optimised out.
    return v > 0.0f;
}

/**
 * Agent function condition which enables all agents
 */
FLAMEGPU_AGENT_FUNCTION_CONDITION(SlowCondition5050) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    float v = 1.f;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Can't write, so just read and increment locally
        v = v + FLAMEGPU->getVariable<float>("v");
    }
    const int fifytfifty = threadIdx.x + blockIdx.x * blockDim.x;
    // Use v in the return val so the loop doesn't get optimised out.
    return fifytfifty ? v > 0.0f : v < 1.0f;
}

/**
 * Agent function which causes a device exception. Slow just to ensure it runs at the same time. No way of timing accurately.)
 */
FLAMEGPU_AGENT_FUNCTION(SlowAgentFunctionWithDeviceException, MessageNone, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 65536;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        // Read from the wrong data type (with a bad size) which should trigger the exception.
        double v = FLAMEGPU->getVariable<double>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    return ALIVE;
}


/**
 * Fast but uninteresting agent function. not useful to time, but allows timing of post processing (to a certain degree)
 */
FLAMEGPU_AGENT_FUNCTION(FastAgentFunction, MessageNone, MessageNone) {
    // Repeatedly do some pointless maths on the value in register
    const int INTERNAL_REPETITIONS = 1;
    for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
        // Read and write all the way to global mem each time to make this intentionally slow
        float v = FLAMEGPU->getVariable<float>("v");
        FLAMEGPU->setVariable("v", v + v);
    }
    return ALIVE;
}

/**
 * Agent function condition which disabled all agents
 */
FLAMEGPU_AGENT_FUNCTION_CONDITION(FastConditionAllFalse) {
    return false;
}

/**
 * Agent function condition which enables all agents
 */
FLAMEGPU_AGENT_FUNCTION_CONDITION(FastConditionAllTrue) {
    return true;
}
/**
 * Agent function condition which enables 50% of agents
 */
FLAMEGPU_AGENT_FUNCTION_CONDITION(FastCondition5050) {
    return (threadIdx.x + blockIdx.x * blockDim.x) % 2;
}

/**
 * Test dectecting concurrency for the simple unintersting case.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, LayerConcurrency) {
    // Define a model with multiple agent types
    ModelDescription m("concurrency_test");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_slowAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, SlowAgentFunction);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test dectecting concurrency for the simple unintersting case.
 * This executes a fast agent function, so might never be useful?
 * Disabled as it is not expected to pass till pinned memory + lots of refactoring is implementd (for sub 1ms improvements per function per layer per iteration per step)
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, DISABLED_FastLayerConcurrency) {
    // Define a model with multiple agent types
    ModelDescription m("concurrency_test");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_fastAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, FastAgentFunction);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Time agents outputting messages to separate lists in the same layer.
 * The same list should not be possible.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConcurrentMessageOutput) {
    // Define a model with multiple agent types
    ModelDescription m("ConcurrentMessageOutput");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_SlowMessageOutputAgentFunction");
        std::string message_name(agent_name + "_messages");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        MessageBruteForce::Description message = m.newMessage(message_name);
        message.newVariable<float>("v");

        auto f = a.newFunction(agent_function, SlowMessageOutputAgentFunction);
        f.setMessageOutput(message);

        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}


/**
 * Time agents outputting and inputting from independeant message lists
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConcurrentMessageOutputInput) {
    // Define a model with multiple agent types
    ModelDescription m("ConcurrentMessageOutputInput");

    // Create two layers.
    LayerDescription layer0  = m.newLayer();
    LayerDescription layer1  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function_out(agent_name + "_SlowMessageOutputAgentFunction");
        std::string agent_function_in(agent_name + "_SlowMessageInputAgentFunction");
        std::string message_name(agent_name + "_messages");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        MessageBruteForce::Description message = m.newMessage(message_name);
        message.newVariable<float>("v");

        auto f_out = a.newFunction(agent_function_out, SlowMessageOutputAgentFunction);
        f_out.setMessageOutput(message);

        layer0.addAgentFunction(f_out);

        auto f_in = a.newFunction(agent_function_in, SlowMessageInputAgentFunction);
        f_in.setMessageInput(message);

        layer1.addAgentFunction(f_in);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Time agents outputting to independant lists, then inputting from a common list (agent_0_messages)
 * With access to per-layer timing, it would be possible to only output to one message list, as the timing of that layer would not effect the potential measured speedup
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConcurrentMessageOutputInputSameList) {
    // Define a model with multiple agent types
    ModelDescription m("ConcurrentMessageOutputInputSameList");

    // Create two layers.
    LayerDescription layer0  = m.newLayer();
    LayerDescription layer1  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    std::string message_in_name("agent_0_messages");

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function_out(agent_name + "_SlowMessageOutputAgentFunction");
        std::string agent_function_in(agent_name + "_SlowMessageInputAgentFunction");
        std::string message_name(agent_name + "_messages");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        MessageBruteForce::Description message = m.newMessage(message_name);
        message.newVariable<float>("v");

        MessageBruteForce::Description message_in = m.Message(message_in_name);

        auto f_out = a.newFunction(agent_function_out, SlowMessageOutputAgentFunction);
        f_out.setMessageOutput(message);

        layer0.addAgentFunction(f_out);

        auto f_in = a.newFunction(agent_function_in, SlowMessageInputAgentFunction);
        f_in.setMessageInput(message_in);

        layer1.addAgentFunction(f_in);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Time agents outputting to independant lists, then inputting from a common list (agent_0_messages), using optional messaging.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConcurrentOptionalMessageOutputInputSameList) {
    // Define a model with multiple agent types
    ModelDescription m("ConcurrentMessageOutputInputSameList");

    // Create two layers.
    LayerDescription layer0  = m.newLayer();
    LayerDescription layer1  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    std::string message_in_name("agent_0_messages");

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function_out(agent_name + "_SlowOptionalMessageOutputAgentFunction");
        std::string agent_function_in(agent_name + "_SlowOptionalMessageInputAgentFunction");
        std::string message_name(agent_name + "_messages");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<unsigned int>("id");
        a.newVariable<float>("v");
        MessageBruteForce::Description message = m.newMessage(message_name);
        message.newVariable<float>("v");

        MessageBruteForce::Description message_in = m.Message(message_in_name);

        auto f_out = a.newFunction(agent_function_out, SlowOptionalMessageOutputAgentFunction);
        f_out.setMessageOutputOptional(true);
        f_out.setMessageOutput(message);

        layer0.addAgentFunction(f_out);

        auto f_in = a.newFunction(agent_function_in, SlowOptionalMessageInputAgentFunction);
        f_in.setMessageInput(message_in);

        layer1.addAgentFunction(f_in);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<unsigned int>("id", j);
            agent.setVariable<unsigned int>("id", j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}


/**
 * Time agents outputting to independant lists, then inputting from a common list (agent_0_messages) using spatial messaging.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConcurrentMessageOutputInputSpatial2D) {
    const float MESSAGE_BOUNDS_MIN = 0.f;
    const float MESSAGE_BOUNDS_MAX = 9.f;
    const float MESSAGE_BOUNDS_RADIUS = 1.f;

    // Define a model with multiple agent types
    ModelDescription m("ConcurrentMessageOutputInputSpatial2D");

    // Create two layers.
    LayerDescription layer0  = m.newLayer();
    LayerDescription layer1  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function_out(agent_name + "_SlowMessageOutputAgentFunctionSpatial2D");
        std::string agent_function_in(agent_name + "_SlowMessageInputAgentFunctionSpatial2D");
        std::string message_name(agent_name + "_messages");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        a.newVariable<float>("x");
        a.newVariable<float>("y");

        MessageSpatial2D::Description message = m.newMessage<MessageSpatial2D>(message_name);
        message.newVariable<float>("v");
        // message.newVariable<float>("x");
        // message.newVariable<float>("y");
        message.setMin(MESSAGE_BOUNDS_MIN, MESSAGE_BOUNDS_MIN);
        message.setMax(MESSAGE_BOUNDS_MAX, MESSAGE_BOUNDS_MAX);
        message.setRadius(MESSAGE_BOUNDS_RADIUS);

        auto f_out = a.newFunction(agent_function_out, SlowMessageOutputAgentFunctionSpatial2D);
        f_out.setMessageOutput(message);

        layer0.addAgentFunction(f_out);

        auto f_in = a.newFunction(agent_function_in, SlowMessageInputAgentFunctionSpatial2D);
        f_in.setMessageInput(message);

        layer1.addAgentFunction(f_in);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, 11.0f);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
            agent.setVariable<float>("x", dist(rng));
            agent.setVariable<float>("y", dist(rng));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}


/**
 * Time agents outputting to independant lists, then inputting from a common list (agent_0_messages) using spatial messaging.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConcurrentMessageOutputInputSpatial3D) {
    const float MESSAGE_BOUNDS_MIN = 0.f;
    const float MESSAGE_BOUNDS_MAX = 9.f;
    const float MESSAGE_BOUNDS_RADIUS = 1.f;

    // Define a model with multiple agent types
    ModelDescription m("ConcurrentMessageOutputInputSpatial3D");

    // Create two layers.
    LayerDescription layer0  = m.newLayer();
    LayerDescription layer1  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function_out(agent_name + "_SlowMessageOutputAgentFunctionSpatial3D");
        std::string agent_function_in(agent_name + "_SlowMessageInputAgentFunctionSpatial3D");
        std::string message_name(agent_name + "_messages");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        a.newVariable<float>("x");
        a.newVariable<float>("y");
        a.newVariable<float>("z");

        MessageSpatial3D::Description message = m.newMessage<MessageSpatial3D>(message_name);
        message.newVariable<float>("v");
        // message.newVariable<float>("x");
        // message.newVariable<float>("y");
        // message.newVariable<float>("z");
        message.setMin(MESSAGE_BOUNDS_MIN, MESSAGE_BOUNDS_MIN, MESSAGE_BOUNDS_MIN);
        message.setMax(MESSAGE_BOUNDS_MAX, MESSAGE_BOUNDS_MAX, MESSAGE_BOUNDS_MAX);
        message.setRadius(MESSAGE_BOUNDS_RADIUS);

        auto f_out = a.newFunction(agent_function_out, SlowMessageOutputAgentFunctionSpatial3D);
        f_out.setMessageOutput(message);

        layer0.addAgentFunction(f_out);

        auto f_in = a.newFunction(agent_function_in, SlowMessageInputAgentFunctionSpatial3D);
        f_in.setMessageInput(message);

        layer1.addAgentFunction(f_in);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        std::mt19937_64 rng;
        std::uniform_real_distribution<float> dist(0.0f, 11.0f);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
            agent.setVariable<float>("x", dist(rng));
            agent.setVariable<float>("y", dist(rng));
            agent.setVariable<float>("z", dist(rng));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test for agent birth (to unique lists). Each agent type executes a function, and birth an agent to it's own population.
 * @note Disabled since AgentID PR (#512), this PR adds a memcpy (before and) after agent birth.
 * @see CUDAFatAgent::getDeviceNextID(): This is called before any agent functon with device birth enabled, however only memcpys on first step or after host agent birth
 * @see CUDAFatAgent::notifyDeviceBirths(unsigned int): This  is called after any agent function with device birth enabled
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, DISABLED_LayerConcurrencyBirth) {
    // Define a model with multiple agent types
    ModelDescription m("LayerConcurrencyBirth");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_SlowAgentFunctionBirth");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, SlowAgentFunctionBirth);
        f.setAgentOutput(a);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test for agent death. Each agent type executes a function and dies.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, LayerConcurrencyDeath) {
    // Define a model with multiple agent types
    ModelDescription m("LayerConcurrencyDeath");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_SlowAgentFunctionDeath");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, SlowAgentFunctionDeath);
        f.setAllowAgentDeath(true);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test if Agent Function conditions are executed in parallel, with all agents disabled so the agent function condition is the only bit being timed.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConditionConcurrencyAllDisabled) {
    // Define a model with multiple agent types
    ModelDescription m("ConditionConcurrencyAllDisabled");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_SlowAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, SlowAgentFunction);
        f.setFunctionCondition(SlowConditionAllFalse);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test if Agent Function conditions are executed in parallel, with all agents enabled so the condition and agent function are being verified as being in parallel.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConditionConcurrencyAllEnabled) {
    // Define a model with multiple agent types
    ModelDescription m("ConditionConcurrencyAllEnabled");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_SlowAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, SlowAgentFunction);
        f.setFunctionCondition(SlowConditionAllTrue);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test if Agent Function conditions are executed in parallel, with all agents enabled so the condition and agent function are being verified as being in parallel.
 * Disabled as it is not expected to pass till pinned memory + lots of refactoring is implementd (for sub 1ms improvements per function per layer per iteration per step)
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, ConditionConcurrency5050) {
    // Define a model with multiple agent types
    ModelDescription m("ConditionConcurrency5050");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_SlowAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, SlowAgentFunction);
        f.setFunctionCondition(SlowCondition5050);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test condition concurrency with fast kernels - i.e. only measure overheads.
 * Disabled as it is not expected to pass till pinned memory + lots of refactoring is implementd (for sub 1ms improvements per function per layer per iteration per step)
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, DISABLED_FastConditiRELEASE_ONLYonConcurrencyAllDisabled) {
    // Define a model with multiple agent types
    ModelDescription m("FastConditionConcurrencyAllDisabled");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_FastAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, FastAgentFunction);
        f.setFunctionCondition(FastConditionAllFalse);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test condition concurrency with fast kernels - i.e. only measure overheads.
 * Disabled as it is not expected to pass till pinned memory + lots of refactoring is implementd (for sub 1ms improvements per function per layer per iteration per step)
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, DISABLED_FastConditionConcurrencyAllEnabled) {
    // Define a model with multiple agent types
    ModelDescription m("FastConditionConcurrencyAllEnabled");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_FastAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, FastAgentFunction);
        f.setFunctionCondition(FastConditionAllTrue);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * Test condition concurrency with fast kernels - i.e. only measure overheads.
 * Disabled as it is not expected to pass till pinned memory + lots of refactoring is implementd (for sub 1ms improvements per function per layer per iteration per step)
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, DISABLED_FastConditionConcurrency5050) {
    // Define a model with multiple agent types
    ModelDescription m("FastConditionConcurrency5050");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector *> populations = std::vector<AgentVector *>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_FastAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, FastAgentFunction);
        f.setFunctionCondition(FastCondition5050);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector * a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

/**
 * If FLAMEGPU_SEATBELTS are on, try to get a device exception from parallel agent functions.
 */
RELEASE_ONLY_SEATBELTS_TEST(TestCUDASimulationConcurrency, LayerConcurrencyDeviceException) {
    // Define a model with multiple agent types
    ModelDescription m("LayerConcurrencyDeviceException");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_SlowAgentFunctionWithDeviceException");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");
        auto f = a.newFunction(agent_function, SlowAgentFunctionWithDeviceException);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);
    s.SimulationConfig().steps = 1;

    // set the flag saying to use streams for agnet function concurrency.
    s.CUDAConfig().inLayerConcurrency = true;
    s.applyConfig();
    EXPECT_EQ(s.CUDAConfig().inLayerConcurrency, true);

    // re-set each population
    for (AgentVector* pop : populations) {
        s.setPopulationData(*pop);
    }
    // Run and time the simulation, expecting a throw.
    EXPECT_THROW(s.simulate(), exception::DeviceError);
}


/************* 
 * RTC Tests *
 *************/

/**
 * Slow and uninteresting RTC agent function which will take a while to run for accurate timing
 */
const char* rtc_slowAgentFunction = R"###(
    FLAMEGPU_AGENT_FUNCTION(rtc_slowAgentFunction, flamegpu::MessageNone, flamegpu::MessageNone) {
        // Repeatedly do some pointless maths on the value in register
        const int INTERNAL_REPETITIONS = 1048576;  // More repetations for RTC to get reliable timing, it's (much) faster than non rtc. 
        for (int i = 0; i < INTERNAL_REPETITIONS; i++) {
            // Reead and write all the way to global mem each time to make this intentionally slow
            float v = FLAMEGPU->getVariable<float>("v");
            FLAMEGPU->setVariable("v", v + 1);
        }
        return flamegpu::ALIVE;
    }
    )###";

/**
 * Test dectecting RTC concurrency for the simple unintersting case.
 */
RELEASE_ONLY_TEST(TestCUDASimulationConcurrency, RTCLayerConcurrency) {
    // Define a model with multiple agent types
    ModelDescription m("rtc_concurrency_test");

    // Create a layer, which contains one function for each agent type - with no dependencies this is allowed.
    LayerDescription layer  = m.newLayer();

    std::vector<AgentVector*> populations = std::vector<AgentVector*>();

    // Add a few agent types, each with a single agent function.
    for (int i = 0; i < CONCURRENCY_DEGREE; i++) {
        // Generate the agent type
        std::string agent_name("agent_" + std::to_string(i));
        std::string agent_function(agent_name + "_slowAgentFunction");
        AgentDescription a = m.newAgent(agent_name);
        a.newVariable<float>("v");

        // @bug - first argument as the same string for many agent types leads to not all functions executing. Issue #378
        auto f = a.newRTCFunction(agent_function.c_str(), rtc_slowAgentFunction);
        layer.addAgentFunction(f);

        // Generate an iniital population.
        AgentVector* a_pop = new AgentVector(a, POPULATION_SIZES);
        for (unsigned int j = 0; j < POPULATION_SIZES; ++j) {
            auto agent = a_pop->at(j);
            agent.setVariable<float>("v", static_cast<float>(j));
        }
        populations.push_back(a_pop);
    }

    // Convert the model to a simulation
    CUDASimulation s(m);

    // Run the simulation many times, with and without concurrency to get an accurate speedup
    float speedup = concurrentLayerSpeedup(TIMING_REPETITIONS, s, populations);
    // Assert that a speedup was achieved.
    EXPECT_GE(speedup, SPEEDUP_THRESHOLD);
}

}  // namespace test_cuda_simulation_concurrency
}  // namespace flamegpu
