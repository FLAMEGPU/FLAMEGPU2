#ifndef INCLUDE_FLAMEGPU_GPU_CUDASUBAGENTSTATELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASUBAGENTSTATELIST_H_

#include <memory>
#include <string>

#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/model/Variable.h"

class CUDAAgent;
class CUDASubAgent;
struct SubAgentData;

// TODO: How is current_list_size synchronised with master_list->current_list_size

/**
 * CUDAAgentStateList for CUDASubAgents (Agents within SubModels)
 * This is a largely rewritten version, whereby some variables are inherited from 'master_list'
 * d_new_list is wholy owned by the subagent state list, even mapped vars
 */
class CUDASubAgentStateList : public CUDAAgentStateList {
 public:
   /**
    * Creates a new CUDASubAgentStateList
    * Initialises unmapped variables, and sets itself as a dependent of master_list (which should init all mapped variables)
    * @param cuda_agent The parent CUDAAgent of the state list, this holds the agent description
    * @param master_list The CUDAAgentStateList which this list borrows variables from
    * @param mapping This contains a map of which sub list variables are mapped to which master list variables
    */
    CUDASubAgentStateList(CUDASubAgent& cuda_agent, const std::shared_ptr<CUDAAgentStateList> &master_list, const std::shared_ptr<SubAgentData> &mapping);
    void setLists(const std::string &var_name, void *list, void *swap_list);
    /**
     * Triggers a resize of all unmapped variables in d_list and d_swap_list
     * Then updates all condition lists
     */
    void resize(bool retain_d_list_data = true) override;

    void setAgentData(const AgentStateMemory &state_memory) override;

    unsigned int scatter(const unsigned int &streamId, const unsigned int out_offset = 0, const ScatterMode &mode = Death) override;
    /**
     * Maps master agent states to sub agent states
     * Then appends unmapped variables to the merge list
     */
    void appendScatterMaps(CUDAMemoryMap &d_list, CUDAMemoryMap &d_swap_list, VariableMap &var_list);
    /**
     * Swaps d_list and d_swap_list (and conditional versions), also steals current_list_size, as it has likely changed
     * Called by master_list when a swap occurs
     */
    void swap();
    /**
     * Increments current_list_size by count and initialises the agents with default value
     * @param new_births Number of new agents
     */
    void addAgents(const unsigned int &new_births, const unsigned int &streamId);

 protected:
    /**
     * Allocates all unmapped variables in the named list
     */
    void allocateDeviceAgentList(CUDAMemoryMap &agent_list, const unsigned int &allocSize) override;
    /**
     * Released all unmapped variables in the named list
     */
    void releaseDeviceAgentList(CUDAMemoryMap &agent_list) override;
    /**
     * Resizes all unmapped variables in the named list
     */
    void resizeDeviceAgentList(CUDAMemoryMap &agent_list, const unsigned int &newSize, bool copyData) override;

 private:
    std::shared_ptr<CUDAAgentStateList> master_list;
    std::shared_ptr<SubAgentData> mapping;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASUBAGENTSTATELIST_H_
