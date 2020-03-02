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

struct VarOffsetStruct;
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
     * @note Does not resize new list
     * @see resizeDeviceAgentList(CUDAMemoryMap &, const unsigned int &, bool)
     */
    void resize();
    /**
     * Resizes the internal newList
     * @param newSize Number of agents to support (probably agent.getMaximumListSize())
     * @see resizeDeviceAgentList(CUDAMemoryMap &, const unsigned int &, bool)
     */
    void resizeNewList(const unsigned int &newSize);

    void setAgentData(const AgentStateMemory &state_memory);

    void getAgentData(AgentStateMemory &state_memory);

    /**
     * Returns pointer to variable from d_list
     * @param variable_name Name of the variable to return
     * @return nullptr is variable not found
     */
    void* getAgentListVariablePointer(const std::string &variable_name) const;
    /**
     * Returns pointer to variable from d_new_list
     * @param variable_name Name of the variable to return
     * @return nullptr is variable not found
     */
    void* getAgentNewListVariablePointer(const std::string &variable_name) const;

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
    /**
     * Scatters agents from d_list_new to d_list
     * Used for device agent creation
     * @param newSize The max possible number of new agents
     * @param streamId Stream index for stream safe operations
     */
    void scatterNew(const unsigned int &newSize, const unsigned int &streamId);
    /**
     * Scatters agents from AoS to d_list SoA
     * Used by host agent creation
     * @param newSize The number of new agents to be copied to d_list
     */
    void scatterHostCreation(const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets);
    /**
     * Initialises newSize agents in d_new_list with their default variable values
     * @param newSize The max possible number of new agents
     * @param streamId Stream index for stream safe operations
     */
    void initNew(const unsigned int &newSize, const unsigned int &streamId);

 protected:
    /*
     * The purpose of this function is to allocate on the device a block of memory for each variable. 
     * These vectors are stored within a hash list using the cuRVE technique so that the location of the vectors can be quickly determined at runtime by FLAME GPU functions.
     * 
     * @param agent_list The agent list to allocate memory within (d_list, d_swap_list, d_new_list)
     * @param allocSize The number of agents that the list needs to be able to hold (probably agent.getMaximumListSize())
     */
    void allocateDeviceAgentList(CUDAMemoryMap &agent_list, const unsigned int &allocSize);

    void releaseDeviceAgentList(CUDAMemoryMap &agent_list);

    void zeroDeviceAgentList(CUDAMemoryMap &agent_list);

    /**
     * Resizes the specified CUDAMemoryMap
     * Only copies across old ddata if copyData is set to True
     * @see resize()
     */
    void resizeDeviceAgentList(CUDAMemoryMap &agent_list, const unsigned int &newSize, bool copyData);

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
     * The number of agents that can fit in d_new_list
     */
    unsigned int d_new_list_alloc_size;
    /**
     * The total number of alive agents in this state
     * This includes temporarily 'disabled' agents (agent function conditions)
     * Not the size of memory allocated, that is agent.getMaximumListSize()
     */
    unsigned int current_list_size;

    CUDAAgent& agent;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
