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

/**
 * Holds a the data for an agent of a single state
 * CUDAAgent owns one of these per agent state
 */
class CUDAAgentStateList {
 public:
    enum ScatterMode{ Death, FunctionCondition, FunctionCondition2};
    explicit CUDAAgentStateList(CUDAAgent& cuda_agent);
    virtual ~CUDAAgentStateList();

    // cant be done in destructor as it requires access to the parent CUDAAgent object
    void cleanupAllocatedData();

    /**
     * Resizes the internal memory, only retains existing data in non scratch buffers
     * @note Size is taken from parent CUDAAgent::max_list_size
     * @see resizeDeviceAgentList(CUDAMemoryMap &, bool)
     */
    void resize();

    void setAgentData(const AgentStateMemory &state_memory);

    void getAgentData(AgentStateMemory &state_memory);

    void* getAgentListVariablePointer(std::string variable_name) const;

    void zeroAgentData();
    /**
     * The number of active and alive agents
     */
    unsigned int getCUDAStateListSize() const;
    /**
     * The number of alive agents (including those temporarily disabled by agent function conditions)
     */
    unsigned int getCUDATrueStateListSize() const;
    /**
     * Sets the number of active agents (does not affect the count of disabled agents)
     */
    void setCUDAStateListSize(const unsigned int &newCount);
    /**
     * Perform a compaction using d_agent_scan_flag and d_agent_position, from d_list to d_swap_list
     * @param streamId The streamId with d_agent_scan_flag, d_agent_position
     * @param out_offset The offset applied to the output index, useful if output array has data which should be retained
     * @param mode The mode variable changes behaviour with the result
     * @param mode Death Reduces the list size, so agents with scan_flag = false are dropped from the list
     * @param mode FunctionCondition Does nothing, it is assumed that the caller will configure the condition state with the return value
     * @return The number of agents with scan flag set to true
     * @note This function could do with some cleanup, e.g. move the mode specific stuff to the calling functions
     */
    unsigned int scatter(const unsigned int &streamId, const unsigned int out_offset = 0, const ScatterMode &mode = Death);

    const CUDAMemoryMap &getReadList() { return condition_d_list; }
    const CUDAMemoryMap &getWriteList() { return condition_d_swap_list; }
    const CUDAMemoryMap &getNewList() { return d_new_list; }

    void setConditionState(const unsigned int &disabledAgentCt);

 protected:
    /*
     * The purpose of this function is to allocate on the device a block of memory for each variable. These vectors are stored within a hash list using the cuRVE technique so that the location of the vectors can be quickly determined at runtime by FLAME GPU functions.
     */
    void allocateDeviceAgentList(CUDAMemoryMap &agent_list);

    void releaseDeviceAgentList(CUDAMemoryMap &agent_list);

    void zeroDeviceAgentList(CUDAMemoryMap &agent_list);

    /**
     * Resizes the specified CUDAMemoryMap
     * Only copies across old ddata if copyData is set to True
     * @see resize()
     */
    void resizeDeviceAgentList(CUDAMemoryMap &agent_list, bool copyData);

 private:
    /**
     * This value is the number of temporarily 'disabled' agents at the start of the state list
     * It is used to temporarily ignore agents which fail agent function conditions
     */
    unsigned int condition_state;
    /**
     * Clones of d_list, d_swap_list
     * These have pointers modified according to condition_state
     * When condition_state==0, they are identical
     */
    CUDAMemoryMap condition_d_list, condition_d_swap_list;

    /**
     * This is the default memoryspace, for each variable, that should be read from
     */
    CUDAMemoryMap d_list;
    /**
     * This is the default memoryspace, for each variable, that should be written to
     */
    CUDAMemoryMap d_swap_list;
    /**
     * This is the default memoryspace, for each variable, that new agents should be written to
     */
    CUDAMemoryMap d_new_list;
    /**
     * The total number of alive agents in this state
     * This includes temporarily 'disabled' agents (agent function conditions)
     */
    unsigned int current_list_size;

    CUDAAgent& agent;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
