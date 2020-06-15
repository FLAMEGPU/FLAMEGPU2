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
#include <unordered_map>

#include "flamegpu/gpu/CUDAFatAgentStateList.h"
#include "flamegpu/pop/AgentStateMemory.h"

struct VarOffsetStruct;
class CUDAAgent;

/**
 * Manages data for an agent state
 * Forwards alot of operations to parent statelist if they need to affected mapped vars too
 */
class CUDAAgentStateList {
 public:
    /**
     * Construct a CUDAAgentStateList view into the provided fat_list
     * @param fat_list The parent CUDAFatAgentStateList
     * @param cuda_agent The parent CUDAAgent
     * @param _fat_index The parent CUDAAgent's fat index within the CUDAFatAgent
     * @param description The parent CUDAAgent's description, used to identify variables
     */
    CUDAAgentStateList(
        const std::shared_ptr<CUDAFatAgentStateList> &fat_list,
        CUDAAgent& cuda_agent,
        const unsigned int &_fat_index,
        const AgentData& description);
    void resize(const unsigned int &minimumSize, const bool &retainData);
    /**
     * Returns the number of alive and active agents in the state list
     */
    unsigned int getSize() const;
    /**
     * Returns the maximum number of agents that can be stored based on the current buffer allocations
     */
    unsigned int getAllocatedSize() const;
    /**
     * Returns the device pointer for the named variable
     */
    void *getVariablePointer(const std::string &variable_name);
    /**
     * Store agent data from agent state memory into state list
     */
    void setAgentData(const AgentStateMemory &data);
    void getAgentData(AgentStateMemory &data);
    /**
     * Initialises the specified number of new agents based on agent data from a device buffer
     * Variables in mapped agents are also initialised to their default values
     * Also updates the count of alive agents to accommodate the new agents
     * @param newSize The number of new agents to initialise
     * @param d_inBuff device pointer to buffer of agent init data
     * @param offsets Offset data explaining the layout of d_inBuff
     */
    void scatterHostCreation(const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets);
    void scatterNew(void * d_newBuff, const unsigned int &newSize, const unsigned int &streamId);

 private:
    std::unordered_map<std::string, std::shared_ptr<VariableBuffer>> variables;
    /**
     * Index of the parent agent within the CUDAFatAgent
     */
    const unsigned int fat_index;
    /**
     * The parent agent which owns this statelist
     */
    CUDAAgent &agent;
    /**
     * The parent state list, which holds var buffers of all mapped agent vars
     */
    const std::shared_ptr<CUDAFatAgentStateList> parent_list;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
