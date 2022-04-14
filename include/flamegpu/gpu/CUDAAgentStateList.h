#ifndef INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_

#include <string>
#include <memory>
#include <map>
#include <list>

#include "flamegpu/gpu/CUDAFatAgentStateList.h"

namespace flamegpu {

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
     * @param _isSubStateList If true, the statelist is mapped to a master agent (and will be reset after CUDASimulation::simulate())
     */
    CUDAAgentStateList(
        const std::shared_ptr<CUDAFatAgentStateList> &fat_list,
        CUDAAgent& cuda_agent,
        const unsigned int &_fat_index,
        const AgentData& description,
        bool _isSubStateList = false);
    /**
     * Construct a CUDAAgentStateList view into the provided fat_list
     * @param fat_list The parent CUDAFatAgentStateList
     * @param cuda_agent The owning CUDAAgent
     * @param _fat_index The owning CUDAAgent's fat index within the CUDAFatAgent
     * @param description The owning CUDAAgent's description, used to identify variables
     * @param _isSubStateList If true, the statelist is mapped to a master agent (and will be reset after CUDASimulation::simulate())
     * @param mapping Mapping definition for how the variables are to their master state.
     */
    CUDAAgentStateList(
        const std::shared_ptr<CUDAFatAgentStateList> &fat_list,
        CUDAAgent& cuda_agent,
        const unsigned int &_fat_index,
        const AgentData& description,
        bool _isSubStateList,
        const SubAgentData::Mapping &mapping);
    /**
     * Resize all variable buffers within the parent CUDAFatAgentStateList
     * Only initialises unmapped agent data
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
     * @param data data Source for agent data
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void setAgentData(const AgentVector &data, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t& stream);
    /**
     * Retrieve agent data from the agent state list into agent state memory
     * @param data data Destination for agent data
     */
    void getAgentData(AgentVector&data) const;
    /**
     * Initialises the specified number of new agents based on agent data from a device buffer
     * Variables in mapped agents are also initialised to their default values
     * Also updates the count of alive agents to accommodate the new agents
     * @param newSize The number of new agents to initialise
     * @param d_inBuff device pointer to buffer of agent init data
     * @param offsets Offset data explaining the layout of d_inBuff
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void scatterHostCreation(const unsigned int &newSize, char *const d_inBuff, const VarOffsetStruct &offsets, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Sorts all agent variables according to the positions stored inside Message Output scan buffer
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void scatterSort_async(CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Scatters agents from the currently assigned device agent birth buffer (see member variable newBuffs)
     * The device buffer must be packed in the same format as CUDAAgent::mapNewRuntimeVariables(const AgentFunctionData&, const unsigned int &, const unsigned int &)
     * @param d_newBuff The buffer holding the new agent data
     * @param newSize The maximum number of new agents (this will be the size of the agent state executing func)
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @return The number of newly birthed agents
     */
    unsigned int scatterNew(void * d_newBuff, const unsigned int &newSize, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Returns true if the state list is not the primary statelist (and is mapped to a master agent state)
     */
    bool getIsSubStatelist();
    /**
     * Returns any unmapped variables (for alive agents) to their default value
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void initUnmappedVars(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Initialises any agent variables within the CUDAFatAgentStateList which are not present in this CUDAAgentStateList
     * @param count Number of variables to init
     * @param offset Offset into the buffer of agents to init
     * @param scatter Scatter instance and scan arrays to be used
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void initExcludedVars(const unsigned int& count, const unsigned int& offset, CUDAScatter& scatter, const unsigned int& streamId, const cudaStream_t& stream);
    /**
     * Returns the statelist to an empty state
     * This resets the size to 0.
     */
    void clear();
    /**
     * Updates the number of alive agents, does not affect disabled agents or change agent data
     * @param newSize Number of active agents
     * @throw exception::InvalidMemoryCapacity If the new number of disabled + active agents would exceed currently allocated buffer capacity
     */
    void setAgentCount(const unsigned int& newSize);
    /**
     * Returns a list of variable buffers attached to bound agents, not available in this agent
     * @note This access is only intended for DeviceAgentVector's correctly handling of subagents
     */
    std::list<std::shared_ptr<VariableBuffer>> getUnboundVariableBuffers();

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
     * If true, the statelist is mapped to a master agent (and will be reset after CUDASimulation::simulate())
     */
    const bool isSubStateList;
    /**
     * These variables are not mapped to a master agent
     * Hence they are reset each time CUDASimulation::simulate() is called
     */
    std::list<std::shared_ptr<VariableBuffer>> unmappedBuffers;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAAGENTSTATELIST_H_
