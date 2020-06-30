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
#include <map>
#include <list>

#include "flamegpu/gpu/CUDAFatAgentStateList.h"
#include "flamegpu/pop/AgentStateMemory.h"

class CUDAScatter;
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
     * @param cuda_agent The owning CUDAAgent
     * @param _fat_index The owning CUDAAgent's fat index within the CUDAFatAgent
     * @param description The owning CUDAAgent's description, used to identify variables
     * @param _isSubStateList If true, the statelist is mapped to a master agent (and will be reset after CUDAAgentModel::simulate())
     * @param mapping Mapping definition for how the variables are to their master state.
     */
    CUDAAgentStateList(
        const std::shared_ptr<CUDAFatAgentStateList> &fat_list,
        CUDAAgent& cuda_agent,
        const unsigned int &_fat_index,
        const AgentData& description,
        bool _isSubStateList = false);
    CUDAAgentStateList(
        const std::shared_ptr<CUDAFatAgentStateList> &fat_list,
        CUDAAgent& cuda_agent,
        const unsigned int &_fat_index,
        const AgentData& description,
        bool _isSubStateList,
        const SubAgentData::Mapping &mapping);
    /**
     * Resize all variable buffers within the parent CUDAFatAgentStateList
     * @param minimumSize The minimum number of agents that must be representable
     * @param retainData If true existing buffer data is retained
     * @see CUDAFatAgentStateList::resize(const unsigned int &, const bool &)
     */
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
     * @data data Source for agent data
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId This is required for scan compaction arrays and async
     */
    void setAgentData(const AgentStateMemory &data, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Retrieve agent data from the agent state list into agent state memory
     * @data data Destination for agent data
     */
    void getAgentData(AgentStateMemory &data);
    /**
     * Initialises the specified number of new agents based on agent data from a device buffer
     * Variables in mapped agents are also initialised to their default values
     * Also updates the count of alive agents to accommodate the new agents
     * @param newSize The number of new agents to initialise
     * @param d_inBuff device pointer to buffer of agent init data
     * @param offsets Offset data explaining the layout of d_inBuff
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId This is required for scan compaction arrays and async
     */
    void scatterHostCreation(const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Sorts all agent variables according to the positions stored inside Message Output scan buffer
     * @param scatter Scatter instance and scan arrays to be used (CUDAAgentModel::singletons->scatter)
     * @param streamId The stream in which the corresponding agent function has executed
     */
    void scatterSort(CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Scatters agents from the currently assigned device agent birth buffer (see member variable newBuffs)
     * The device buffer must be packed in the same format as CUDAAgent::mapNewRuntimeVariables(const AgentFunctionData&, const unsigned int &, const unsigned int &)
     * @param d_newBuff The buffer holding the new agent data
     * @param newSize The maximum number of new agents (this will be the size of the agent state executing func)
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId This is required for scan compaction arrays and async
     */
    void scatterNew(void * d_newBuff, const unsigned int &newSize, CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Returns true if the state list is not the primary statelist (and is mapped to a master agent state)
     */
    bool getIsSubStatelist();
    /**
     * Returns any unmapped variables (for alive agents) to their default value
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId This is required for scan compaction arrays and async
     */
    void initUnmappedVars(CUDAScatter &scatter, const unsigned int &streamId);
    /**
     * Returns the statelist to an empty state
     * This resets the size to 0.
     */
    void clear();

 private:
    /**
     * map rather than unordered_map is used here intentionally
     * Variable iteration order affects how variables are stored within new buffer for device agent creation
     */
    std::map<std::string, std::shared_ptr<VariableBuffer>> variables;
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
    /**
     * If true, the statelist is mapped to a master agent (and will be reset after CUDAAgentModel::simulate())
     */
    const bool isSubStateList;
    /**
     * These variables are not mapped to a master agent
     * Hence they are reset each time CUDAAgentModel::simulate() is called
     */
    std::list<std::shared_ptr<VariableBuffer>> unmappedBuffers;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
