#ifndef INCLUDE_FLAMEGPU_GPU_CUDAFATAGENTSTATELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAFATAGENTSTATELIST_H_

#include <memory>
#include <list>
#include <unordered_map>
#include <set>
#include <string>
#include <utility>

#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/SubAgentData.h"

namespace flamegpu {

class CUDAScatter;

/**
 * This is used to identify a variable that belongs to specific agent
 * This agent's unsigned int is assigned by the parent CUDAFatAgent
 * This is required, as two mapped agents might have variables of the same name which are not mapped
 */
struct AgentVariable{
    const unsigned int agent;
    const std::string variable;
    /**
     * Basic comparison operator, required for use of std::map etc
     */
    bool operator==(const AgentVariable &other) const {
        return (agent == other.agent && variable == other.variable);
    }
};

/**
 * Hash function so that AgentVariable can be used as a key in a map.
 */
struct AgentVariableHash {
    std::size_t operator()(const flamegpu::AgentVariable& k) const noexcept {
        return ((std::hash<unsigned int>()(k.agent)
            ^ (std::hash<std::string>()(k.variable) << 1)) >> 1);
    }
};

/**
 * This represents a raw buffer
 */
struct VariableBuffer {
    /**
     * Pointer to the device buffer of data
     */
    void *data;
    /**
     * Pointer to an offset within data, this should be tied to the number of disabled agents
     */
    void *data_condition;
    /**
     * Pointer to a spare device buffer used when scattering
     */
    void *data_swap;
    /**
     * The type of the variable
     */
    const std::type_index type;
    /**
     * The size of the variable's type (e.g. sizeof(T))
     */
    const size_t type_size;
    /**
     * Is the variable array type
     */
    const size_t elements;
    /**
     * Pointer to the default value of the variable
     * The length of the allocation is equal to elements * type_size
     * @note The memory pointed to by this pointer is allocated and free'd by the instance
     */
    const void *const default_value;
    VariableBuffer(const std::type_index &_type, const size_t &_type_size, const void * const _default_value, const size_t &_elements = 1, void *_data = nullptr, void *_data_swap = nullptr)
        : data(_data)
        , data_condition(_data)
        , data_swap(_data_swap)
        , type(_type)
        , type_size(_type_size)
        , elements(_elements)
        , default_value(buildDefaultValue(_default_value)) { }
    /**
     * Copy constructor
     */
    VariableBuffer(const VariableBuffer&other)
        : data(other.data)
        , data_condition(other.data_condition)
        , data_swap(other.data_swap)
        , type(other.type)
        , type_size(other.type_size)
        , elements(other.elements)
        , default_value(buildDefaultValue(other.default_value)) { }
    /**
     * Destructor
     */
    ~VariableBuffer() {
        free(const_cast<void*>(default_value));
    }
    /**
     * Swaps the buffers pointed to by the two VariableBuffers
     * This is used when performing a state transition to an empty statelist
     * The two buffers must point to the same variable
     */
    void swap(VariableBuffer *other) {
        std::swap(data, other->data);
        std::swap(data_condition, other->data_condition);
        std::swap(data_swap, other->data_swap);
        assert(type == other->type);
        assert(type_size == other->type_size);
        assert(elements == other->elements);
        // assert(default_value == other->default_value);  // Would need to evaluate the value not ptr
    }

 private:
    /**
     * Returns a copy of the paramater allocated with malloc
     * It is expected that source points to atleast type_size * elements memory
     * @param source The source location of the default value
     * @return The destination location of the default value copy
     */
    void *buildDefaultValue(const void * const source) const {
        void *rtn = malloc(type_size * elements);
        memcpy(rtn, source, type_size * elements);
        return rtn;
    }
};

/**
 * Represents state list properties shared between mapped agents
 * @todo resize() Improve data retention, by launching a single scatter all kernel, rather than individually memcpy each variable
 */
class CUDAFatAgentStateList {
 public:
    /**
     * Constructs a new state list with variables from the provided description
     * Memory for buffers is not allocated until resize() is called
     */
    explicit CUDAFatAgentStateList(const AgentData& description);
    /**
     * Copy constructor, this clones an existing CUDAFatAgentStateList
     * However buffers must be uninitialised (bufferLen == 0)
     */
    CUDAFatAgentStateList(const CUDAFatAgentStateList& other);
    /**
     * Destructor
     */
    ~CUDAFatAgentStateList();
    /**
     * Adds variables from a sub agent to the state list
     * Variables which are not mapped will have a new VariableBuffer created
     * Variables which are mapped will be assigned to the corresponding VariableBuffer
     * @param description Agent description of the sub agent
     * @param master_fat_index Fat index of the sub agent's mapped (parent) agent within the CUDAFatAgent
     * @param sub_fat_index Fat index of the sub agent within the CUDAFatAgent
     * @param mapping Mapping definition for how this agent is connected with its master agent.
     */
    void addSubAgentVariables(
      const AgentData &description,
      const unsigned int &master_fat_index,
      const unsigned int &sub_fat_index,
      const std::shared_ptr<SubAgentData> &mapping);
    /**
     * Returns the VariableBuffer for the corresponding agent variable
     * @param fat_index Fat index of the corresponding agent
     * @param name Name of the variable within the corresponding agent
     */
    std::shared_ptr<VariableBuffer> getVariableBuffer(const unsigned int &fat_index, const std::string &name);
    /**
     * Resize all variable buffers
     * @param minSize The minimum number of agents that must be representable
     * @param retainData If true existing buffer data is retained
     */
    void resize(const unsigned int &minSize, const bool &retainData);
    /**
     * Returns the number of alive and active agents in the state list
     */
    unsigned int getSize() const;
    /**
     * Returns the total number of agents in the state list, including disabled agents
     */
    unsigned int getSizeWithDisabled() const;
    /**
     * Returns the maximum number of agents that can be stored based on the current buffer allocations
     */
    unsigned int getAllocatedSize() const;
    /**
     * Updates the number of alive agents within the list
     * @param newCount New number of alive (and active agents)
     * @param resetDisabled If true, the disableAgents count will be set to 0, else it will be added to newCount for the internal true alive count
     */
    void setAgentCount(const unsigned int &newCount, const bool &resetDisabled = false);
    /**
     * Scatters all living agents (including disabled, according to the provided stream's death flag)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @return The number of agents that are still alive (this includes temporarily disabled agents due to agent function condition)
     */
    unsigned int scatterDeath(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Scatters all living agents which failed the agent function condition into the swap buffer (there should be no disabled at this time)
     * This does not swap buffers or update disabledAgent)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @return The number of agents that were scattered (the number of agents which failed the condition)
     * @see scatterAgentFunctionConditionTrue(const unsigned int &, const unsigned int &)
     */
    unsigned int scatterAgentFunctionConditionFalse(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Scatters all living agents which passed the agent function condition into the swap buffer (there should be no disabled at this time)
     * Also swaps the buffers and sets the number of disabled agents
     * @param conditionFailCount The number of agents which failed the condition (they should already have been scattered to swap)
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     * @return The number of agents that were scattered (the number of agents which passed the condition)
     * @see scatterAgentFunctionConditionFalse(const unsigned int &)
     * @see setConditionState(const unsigned int &)
     */
    unsigned int scatterAgentFunctionConditionTrue(const unsigned int &conditionFailCount, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Sorts all agent variables according to the positions stored inside Message Output scan buffer
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void scatterSort(CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Set the number of disabled agents within the state list
     * Updates member var disabledAgents and data_condition for every item inside variables_unique
     * @param numberOfDisabled The new number of disabled agents (this can increase of decrease)
     */
    void setDisabledAgents(const unsigned int &numberOfDisabled);
    /**
     * Resets the value of all variables not present in exclusionSet to their defaults
     *
     * This is useful for when an agent in a submodel is birthed, and it's parent model requires variables initialising and other similar features
     * @param exclusionSet Set of variable buffers not to be initialised
     * @param initCount Total number of agents which require variables to be initialised
     * @param initOffset Offset into the agent list to initialise variables from
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void initVariables(std::set<std::shared_ptr<VariableBuffer>> &exclusionSet, const unsigned int initCount, const unsigned initOffset, CUDAScatter &scatter, const unsigned int &streamId, const cudaStream_t &stream);
    /**
     * Returns the collection of unique variable buffers held by this CUDAFatAgentStateList
     */
    std::list<std::shared_ptr<VariableBuffer>> &getUniqueVariables();
    /**
     * Swaps the internals of the two buffers
     * This is used when performing a state transition to an empty statelist
     */
    void swap(CUDAFatAgentStateList*other);
    /**
     * Returns a list of variable buffers managed by the CUDAFatAgentStateList, which do not occur in exclusionSet
     * @param exclusionSet A set of buffers to compare against
     * @note Comparison checks for identical shared_ptr, not equality of VariableBuffer
     * @note This access is only intended for DeviceAgentVector's correctly handling of subagents
     */
    std::list<std::shared_ptr<VariableBuffer>> getBuffers(std::set<std::shared_ptr<VariableBuffer>>& exclusionSet);

 private:
    /**
     * Count includes disabled agents
     */
    unsigned int aliveAgents;
    /**
     * Ignored agents at the start of buffer, due to agent function condition
     * This should always return to 0 before an agent function's processing returns
     */
    unsigned int disabledAgents;
    /**
     * Max number of agents buffer can currently store
     * This is represented on a per FatAgentStateList level, rather then per VariableBuffer
     */
    unsigned int bufferLen;
    /**
     * Mapping from {agent fat index, variable name} to variable buffer
     */
    std::unordered_map<AgentVariable, std::shared_ptr<VariableBuffer>, AgentVariableHash> variables;
    /**
     * The collection of unique variables represented
     * This is a list, however it contains no duplicates
     */
    std::list<std::shared_ptr<VariableBuffer>> variables_unique;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAFATAGENTSTATELIST_H_
