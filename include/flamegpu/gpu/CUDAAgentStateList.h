 /**
 * @file CUDAAgentList.h
 * @author
 * @date    Feb 2017
 * @brief
 *
 * \todo longer description
 */

#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <utility>

class CUDAAgent;
class AgentStateMemory;

// #define UNIFIED_GPU_MEMORY

typedef std::map <std::string, void*> CUDAMemoryMap;        // map of pointers to gpu memory for each variable name
typedef std::pair <std::string, void*> CUDAMemoryMapPair;

class CUDAAgentStateList {
 public:
    explicit CUDAAgentStateList(CUDAAgent& cuda_agent);
    virtual ~CUDAAgentStateList();

    // cant be done in destructor as it requires access to the parent CUDAAgent object
    void cleanupAllocatedData();

    void setAgentData(const AgentStateMemory &state_memory);

    void getAgentData(AgentStateMemory &state_memory);

    void* getAgentListVariablePointer(std::string variable_name) const;

    void zeroAgentData();

    unsigned int getCUDAStateListSize() const;
    /**
     * Perform a compaction using d_agent_scan_flag and d_agent_position
     */
    void scatter();

 protected:
    /*
     * The purpose of this function is to allocate on the device a block of memory for each variable. These vectors are stored within a hash list using the cuRVE technique so that the location of the vectors can be quickly determined at runtime by FLAME GPU functions.
     */
    void allocateDeviceAgentList(CUDAMemoryMap &agent_list);

    void releaseDeviceAgentList(CUDAMemoryMap &agent_list);

    void zeroDeviceAgentList(CUDAMemoryMap &agent_list);

 private:
    CUDAMemoryMap d_list;
    CUDAMemoryMap d_swap_list;
    CUDAMemoryMap d_new_list;

    unsigned int current_list_size;  // ???

    CUDAAgent& agent;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
