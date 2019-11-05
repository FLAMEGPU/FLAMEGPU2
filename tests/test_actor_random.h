/**
 * @copyright  2017 University of Sheffield
 *
 *
 * @file       test_actor_random.h
 * @authors    Robert Chisholm
 * @brief      Test suite for validating methods in simulation folder
 *
 * @see        https://github.com/FLAMEGPU/FLAMEGPU2_dev
 * @bug        No known bugs
 */

#include <flamegpu/flame_api.h>

#include "device_functions.h"

using namespace std;

BOOST_AUTO_TEST_SUITE(AgentRandomTest) //name of the test suite is SimTest

/**
 * @brief      To verify the correctness of actor random
 *
 * This test checks whether ActorRandom and Random behave correctly
*/
BOOST_AUTO_TEST_CASE(ActorRandomCheck)
{
    BOOST_TEST_MESSAGE("\nTesting ActorRandom and Random Name ..");

    const int AGENT_COUNT = 5;


    ModelDescription model("random_model");
    AgentDescription agent("agent");

    agent.addAgentVariable<float>("a");
    agent.addAgentVariable<float>("b");
    agent.addAgentVariable<float>("c");

    AgentFunctionDescription do_random("do_random");
    //do_random.setFunction(&random1);
    attach_random1_func(do_random);
    agent.addAgentFunction(do_random);

    model.addAgent(agent);


    AgentPopulation population(agent, AGENT_COUNT);
    for (int i = 0; i< AGENT_COUNT; i++)
    {
        AgentInstance instance = population.getNextInstance("default");
        //AgentInstance instance = population.getInstanceAt(i);
        instance.setVariable<float>("a", 0);
        instance.setVariable<float>("b", 0);
        instance.setVariable<float>("c", 0);
    }

    Simulation simulation(model);

    SimulationLayer layer(simulation, "layer");
    layer.addAgentFunction("do_random");
    simulation.addSimulationLayer(layer);

    CUDAAgentModel cuda_model(model);
    cuda_model.setInitialPopulationData(population);
    char *args_1[4] = { "process.exe", "input.xml", "-r", "0" };
    char *args_2[4] = { "process.exe", "input.xml", "-r", "1" };
    std::string _t_unused = std::string();
    std::vector<std::tuple<float, float, float>> results1, results2;
    {
        /**
        * Test Model 1
        * Do agents generate different random numbers
        * Does random number change each time it's called
        */
        //Seed random
        simulation.checkArgs(4, args_1, _t_unused);
        
        cuda_model.simulate(simulation);

        cuda_model.getPopulationData(population);

        float a1 = -1, b1 = -1, c1 = -1, a2 = -1, b2 = -1, c2 = -1;
        for (int i = 0; i < population.getCurrentListSize(); i++)
        {
            if (i != 0)
            {
                a2 = a1;
                b2 = b1;
                c2 = c1;
            }
            AgentInstance instance = population.getInstanceAt(i);
            a1 = instance.getVariable<float>("a");
            b1 = instance.getVariable<float>("b");
            c1 = instance.getVariable<float>("c");
            results1.push_back({a1, b1, c1});
            if (i != 0)
            {
                //Different agents get different random numbers
                BOOST_CHECK(a1 != a2);
                BOOST_CHECK(b1 != b2);
                BOOST_CHECK(c1 != c2);
            }
            //Multiple calls get multiple random numbers
            BOOST_CHECK(a1 != b1);
            BOOST_CHECK(b1 != c1);
            BOOST_CHECK(a1 != c1);
        }
        BOOST_CHECK(results1.size() == AGENT_COUNT);
    }
    {
        /**
         * Test Model 2
         * Different seed produces different random numbers
         */
        //Seed random
        simulation.checkArgs(4, args_2, _t_unused);

        cuda_model.simulate(simulation);

        cuda_model.getPopulationData(population);

        for (int i = 0; i < population.getCurrentListSize(); i++)
        {
            AgentInstance instance = population.getInstanceAt(i);
            results2.push_back({ 
                instance.getVariable<float>("a"), 
                instance.getVariable<float>("b"), 
                instance.getVariable<float>("c") 
            });
        }
        BOOST_CHECK(results2.size() == AGENT_COUNT);

        for(int i = 0; i<results1.size();++i)
        {
            //Different seed produces different results
            BOOST_CHECK(results1[i]!=results2[i]);
        }
    }
    {
        /**
        * Test Model 3
        * Different seed produces different random numbers
        */
        results2.clear();
        //Seed random
        simulation.checkArgs(4, args_1, _t_unused);

        cuda_model.simulate(simulation);

        cuda_model.getPopulationData(population);

        for (int i = 0; i < population.getCurrentListSize(); i++)
        {
            AgentInstance instance = population.getInstanceAt(i);
            results2.push_back({
                instance.getVariable<float>("a"),
                instance.getVariable<float>("b"),
                instance.getVariable<float>("c")
            });
        }
        BOOST_CHECK(results2.size() == AGENT_COUNT);

        for (int i = 0; i<results1.size(); ++i)
        {
            //Same seed produces different results
            BOOST_CHECK(results1[i] == results2[i]);
        }
    }
}


BOOST_AUTO_TEST_CASE(ActorRandomFunctionsNoExcept)
{
    BOOST_TEST_MESSAGE("\nTesting ActorRandom functions all work");

    const int AGENT_COUNT = 5;


    ModelDescription model("random_model");
    AgentDescription agent("agent");

    agent.addAgentVariable<float>("uniform_float");
    agent.addAgentVariable<double>("uniform_double");

    agent.addAgentVariable<float>("normal_float");
    agent.addAgentVariable<double>("normal_double");

    agent.addAgentVariable<float>("logNormal_float");
    agent.addAgentVariable<double>("logNormal_double");

    // char
    agent.addAgentVariable<char>("uniform_char");
    agent.addAgentVariable<unsigned char>("uniform_u_char");
    // short
    agent.addAgentVariable<short>("uniform_short");
    agent.addAgentVariable<unsigned short>("uniform_u_short");
    // int
    agent.addAgentVariable<int>("uniform_int");
    agent.addAgentVariable<unsigned int>("uniform_u_int");
    // long
    agent.addAgentVariable<long>("uniform_long");
    agent.addAgentVariable<unsigned long>("uniform_u_long");
    // long long
    agent.addAgentVariable<long long>("uniform_longlong");
    agent.addAgentVariable<unsigned long long>("uniform_u_longlong");

    AgentFunctionDescription do_random("do_random");
    //do_random.setFunction(&random1);
    attach_random2_func(do_random);
    agent.addAgentFunction(do_random);

    model.addAgent(agent);


    AgentPopulation population(agent, AGENT_COUNT);
    for (int i = 0; i< AGENT_COUNT; i++)
    {
        //Actually create the agents
        AgentInstance instance = population.getNextInstance("default");
        //Don't bother initialising
    }

    Simulation simulation(model);

    SimulationLayer layer(simulation, "layer");
    layer.addAgentFunction("do_random");
    simulation.addSimulationLayer(layer);

    CUDAAgentModel cuda_model(model);
    cuda_model.setInitialPopulationData(population);
    cuda_model.simulate(simulation);
    //Success if we get this far without an exception being thrown.
}

BOOST_AUTO_TEST_CASE(ActorRandomArrayResizeNoExcept)
{
    BOOST_TEST_MESSAGE("\nTesting d_random scales up/down without breaking");
    
    std::vector<int> AGENT_COUNTS = {1024, 512 * 1024, 1024, 1024 * 1024};

    //TODO: Can't yet control agent population up/down

    //Success if we get this far without an exception being thrown.
}


BOOST_AUTO_TEST_SUITE_END()

