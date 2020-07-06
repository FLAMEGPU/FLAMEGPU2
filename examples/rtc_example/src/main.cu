/******************************************************************************
 * main.cu is a host function that prepares data array and passes it to the CUDA kernel.
 * This main.cu would either be specified by a user or automatically generated from the model.xml.
 * Each of the API functions will have a 121 mapping with XML elements
 * The API is very similar to FLAME 2. The directory structure and general project is set out very similarly.

 * Single Agent model example

 ******************************************************************************
 * Author  Paul Richmond, Mozhgan Kabiri Chimeh
 * Date    Feb 2017
 *****************************************************************************/

#include "flamegpu/flame_api.h"


/* must be compiled separately using FLAME GPU builder
 * This will generate object files for different architecture targets as well as ptx info for each agent function (registers, memory use etc.)
 * http://stackoverflow.com/questions/12388207/interpreting-output-of-ptxas-options-v
 */

#define AGENT_COUNT 32
#define EXPECT_EQ(x, y) if (x != y) printf("%d not equal to %d", x, y)


const char* rtc_func = R"###(
FLAMEGPU_AGENT_FUNCTION(MandatoryOutput, MsgNone, MsgNone) {
    unsigned int id = FLAMEGPU->getVariable<unsigned int>("id") + 1;
    FLAMEGPU->agent_out.setVariable<float>("x", id + 12.0f);
    FLAMEGPU->agent_out.setVariable<unsigned int>("id", id);
    return ALIVE;
}
)###";

/**
 * Test an RTC function to an agent function condition (where the condition is not compiled using RTC)
 */
int main() {
    // Define model
    ModelDescription model("Spatial3DMsgTestModel");
    AgentDescription &agent = model.newAgent("agent");
    AgentDescription &agent2 = model.newAgent("agent2");
    agent.newState("a");
    agent.newState("b");
    agent.newVariable<float>("x");
    agent.newVariable<unsigned int>("id");
    agent2.newVariable<float>("x");
    agent2.newVariable<unsigned int>("id");
    AgentFunctionDescription &function = agent2.newRTCFunction("output", rtc_func);
    function.setAgentOutput(agent, "b");
    LayerDescription &layer1 = model.newLayer();
    layer1.addAgentFunction(function);
    // Init agent pop
    CUDAAgentModel cuda_model(model);
    AgentPopulation population(model.Agent("agent2"), AGENT_COUNT);
    // Initialise agents
    for (unsigned int i = 0; i < AGENT_COUNT; i++) {
        AgentInstance instance = population.getNextInstance();
        instance.setVariable<float>("x", i + 1.0f);
        instance.setVariable<unsigned int>("id", i);
    }
    cuda_model.setPopulationData(population);
    // Execute model
    cuda_model.step();
    // Test output
    AgentPopulation newPopulation(model.Agent("agent"));
    cuda_model.getPopulationData(population);
    cuda_model.getPopulationData(newPopulation);
    // Validate each agent has same result
    EXPECT_EQ(population.getCurrentListSize(), AGENT_COUNT);
    EXPECT_EQ(newPopulation.getCurrentListSize("b"), AGENT_COUNT);
    unsigned int is_1_mod2_0 = 0;
    unsigned int is_1_mod2_1 = 0;
    for (unsigned int i = 0; i < population.getCurrentListSize(); ++i) {
        AgentInstance ai = population.getInstanceAt(i);
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 1.0f);
        if (ai.getVariable<unsigned int>("id") % 2 == 0) {
            is_1_mod2_0++;
        } else {
            is_1_mod2_1++;
        }
    }
    EXPECT_EQ(is_1_mod2_0, AGENT_COUNT / 2);
    EXPECT_EQ(is_1_mod2_1, AGENT_COUNT / 2);
    unsigned int is_12_mod2_0 = 0;
    unsigned int is_12_mod2_1 = 0;
    for (unsigned int i = 0; i < newPopulation.getCurrentListSize("b"); ++i) {
        AgentInstance ai = newPopulation.getInstanceAt(i, "b");
        EXPECT_EQ(ai.getVariable<float>("x") - ai.getVariable<unsigned int>("id"), 12.0f);
        if (ai.getVariable<unsigned int>("id") % 2 == 0) {
            is_12_mod2_0++;
        } else {
            is_12_mod2_1++;
        }
    }
    EXPECT_EQ(is_12_mod2_0, AGENT_COUNT / 2);
    EXPECT_EQ(is_12_mod2_1, AGENT_COUNT / 2);
}
