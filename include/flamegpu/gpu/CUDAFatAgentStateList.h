#ifndef INCLUDE_FLAMEGPU_GPU_CUDAFATAGENTSTATELIST_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAFATAGENTSTATELIST_H_

#include <memory>
#include <list>
#include <unordered_map>
#include <set>
#include <string>

#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/SubAgentData.h"

/**
 * This is used to identify a variable that belongs to specific agent
 * This agent's unsigned int is assigned by the parent CUDAFatAgent
 * This is required, as two mapped agents might have variables of the same name which are not mapped
 */
struct AgentVariable{
    const unsigned int agent;
    const std::string variable;
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
    const std::type_index type;
    const size_t type_size;
    /**
     * Is the variable array type
     */
    const size_t elements;
    const void *const default_value;
    VariableBuffer(const std::type_index &_type, const size_t &_type_size, const void * const _default_value, const size_t &_elements = 1, void *_data = nullptr, void *_data_swap = nullptr)
        : data(_data)
        , data_swap(_data_swap)
        , type(_type)
        , type_size(_type_size)
        , elements(_elements)
        , default_value(buildDefaultValue(_default_value)) { }

    VariableBuffer(const VariableBuffer&other)
        : data(other.data)
        , data_swap(other.data_swap)
        , type(other.type)
        , type_size(other.type_size)
        , elements(other.elements)
        , default_value(buildDefaultValue(other.default_value)) { }

    ~VariableBuffer() {
        free(const_cast<void* const>(default_value));
    }

 private:
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
    ~CUDAFatAgentStateList();
    void addSubAgentVariables(
      const AgentData &description,
      const unsigned int &master_fat_index,
      const unsigned int &sub_fat_index,
      const std::shared_ptr<SubAgentData> &mapping);
    std::shared_ptr<VariableBuffer> getVariableBuffer(const unsigned int &fat_index, const std::string &name);
    void resize(const unsigned int &minSize, const bool &retainData);
    /**
     * Returns the number of alive and active agents in the state list
     */
    unsigned int getSize() const;
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
     * @param streamId The stream in which the corresponding agent function has executed
     * @return The number of agents that are still alive (this includes temporarily disabled agents due to agent function condition)
     */
    unsigned int scatterDeath(const unsigned int &streamId);
    /**
     * Scatters all living agents which failed the agent function condition into the swap buffer (there should be no disabled at this time)
     * This does not swap buffers or update disabledAgent)
     * @param streamId The stream in which the corresponding agent function has executed
     * @return The number of agents that were scattered (the number of agents which failed the condition)
     * @see scatterAgentFunctionConditionTrue(const unsigned int &, const unsigned int &)
     */
    unsigned int scatterAgentFunctionConditionFalse(const unsigned int &streamId);
    /**
     * Scatters all living agents which passed the agent function condition into the swap buffer (there should be no disabled at this time)
     * Also swaps the buffers and sets the number of disabled agents
     * @param streamId The stream in which the corresponding agent function has executed
     * @param conditionFailCount The number of agents which failed the condition (they should already have been scattered to swap)
     * @return The number of agents that were scattered (the number of agents which passed the condition)
     * @see scatterAgentFunctionConditionFalse(const unsigned int &)
     * @see setConditionState(const unsigned int &)
     */
    unsigned int scatterAgentFunctionConditionTrue(const unsigned int &conditionFailCount, const unsigned int &streamId);
    /**
     * Set the number of disabled agents within the state list
     * Updates member var disabledAgents and data_condition for every item inside variables_unique
     * @param numberOfDisabled The new number of disabled agents (this can increase of decrease)
     */
    void setDisabledAgents(const unsigned int numberOfDisabled);
    /**
     * Resets the value of all variables not present in exclusionSet to their defaults
     */
    void initVariables(std::set<std::shared_ptr<VariableBuffer>> &exclusionSet, const unsigned int initCount, const unsigned initOffset, const unsigned int &streamId);

    std::list<std::shared_ptr<VariableBuffer>> &getUniqueVariables();

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

    std::unordered_map<AgentVariable, std::shared_ptr<VariableBuffer>> variables;
    std::list<std::shared_ptr<VariableBuffer>> variables_unique;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAFATAGENTSTATELIST_H_
