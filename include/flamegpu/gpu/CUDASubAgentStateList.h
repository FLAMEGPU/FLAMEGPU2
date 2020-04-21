#ifndef INCLUDE_FLAMEGPU_GPU_CUDASUBAGENTSTATELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASUBAGENTSTATELIST_H_

#include <memory>
#include <string>

#include "flamegpu/gpu/CUDAAgentStateList.h"

class CUDAAgent;
class CUDASubAgent;
struct SubAgentData;

/**
 * CUDAAgentStateList for CUDASubAgents (Agents within SubModels)
 * This is a largely rewritten version, whereby some variables are inherited from 'master_list'
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
    void setLists(const std::string &var_name, void *list, void *swap_list, void *new_list);

 protected:
    /**
     * Allocates all unmapped variables in the named list
     */
    void allocateDeviceAgentList(CUDAMemoryMap &agent_list, const unsigned int &allocSize) override;
   /**
    * Released all unmapped variables in the named list
    */
    void releaseDeviceAgentList(CUDAMemoryMap &agent_list) override;

 private:
    std::shared_ptr<CUDAAgentStateList> master_list;
    std::shared_ptr<SubAgentData> mapping;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDASUBAGENTSTATELIST_H_
