#ifndef INCLUDE_FLAMEGPU_GPU_CUDASUBAGENT_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASUBAGENT_H_

#include <memory>
#include <string>

#include "flamegpu/gpu/CUDAAgent.h"

struct AgentData;
struct SubAgentData;


/**
 * CUDAAgent for Agents within SubModels
 * This is a largely rewritten version, whereby the CUDASubAgent matches the size of 'master_agent' at all times
 * Some or all CUDAAgentStateLists may depend on state lists within master_agent
 */
class CUDASubAgent : public CUDAAgent {
 public:
    /**
     * Constructs a new CUDASubAgent
     * @param description Agent description from the model description hierarchy
     * @param master_agent The CUDAAgent which this Agent is bound to and inherits from
     * @param mapping Mapping information, which lists the state and variable mappings
     */
    CUDASubAgent(const AgentData &description, const CUDAAgentModel& cuda_model, const std::unique_ptr<CUDAAgent> &master_agent, const std::shared_ptr<SubAgentData> &mapping);
    /**
     * Instead triggers resize on master_agent, then propagates the resize back to this agent for mapped variables
     * Resize is then called on this agent, to use the master agent's size and update unmapped vars (and condition lists)
     */
    void resize(const unsigned int &newSize, const unsigned int &streamId) override;
    /**
     * I don't think it should actually be possible to call this method on a submodel's agents
     * However, it should work if required
     */
    void setPopulationData(const AgentPopulation& population) override;
    /**
     * Instead returns max size from master_agent, as we copy master_agent size throughout
     */
    unsigned int getMaximumListSize() const override;
    /**
     * Translates master state to bound sub state and swaps the two lists (if possible)
     */
    void swapListsMaster(const std::string &master_state_source, const std::string &master_state_dest);
    /**
     * Maps master agent states to sub agent states
     * Then appends unmapped variables to the merge list
     * It then even propagates to a dependent agent, or performs a scatter all operation
     */
    void appendScatterMaps(
        const std::string &master_state_source,
        CUDAMemoryMap &merge_d_list,
        const std::string &master_state_dest,
        CUDAMemoryMap &merge_d_swap_list,
        VariableMap &merge_var_list,
        unsigned int streamId,
        unsigned int srcListSize,
        unsigned int destListSize);

 private:
    const std::unique_ptr<CUDAAgent> &master_agent;
    const std::shared_ptr<const SubAgentData> &mapping;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASUBAGENT_H_
