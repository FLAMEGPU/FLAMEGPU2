#ifndef INCLUDE_FLAMEGPU_GPU_CUDAFATAGENT_H_
#define INCLUDE_FLAMEGPU_GPU_CUDAFATAGENT_H_

#include <memory>
#include <unordered_map>
#include <set>
#include <mutex>
#include <string>


#include "flamegpu/gpu/CUDAAgentStateList.h"
#include "flamegpu/gpu/CUDAFatAgentStateList.h"
#include "flamegpu/model/SubAgentData.h"

namespace flamegpu {

class HostAPI;

/**
 * This is a shared CUDAFatAgent
 * It manages the buffers for the variables of all agent-state combinations for a group of mapped agents
 * This enables operations applied to mapped agents to impact all affected variables even if not present in the current CUDAAgent
 */
class CUDAFatAgent {
    /**
     * This is used to identify a state that belongs to specific agent
     * This agent's unsigned int is assigned by the CUDAFatAgent
     * However only the parent CUDAAgent knows it's own agent id
     * The agent at the top of the mapped agent hierarchy will always be id 0
     * With each sub agent having the consecutive index
     */
    struct AgentState{
        /**
         * Index assigned to the agent when it is added to the CUDAFatAgent
         * @note An index is used as two mapped agents from different models may share the same name
         * @see CUDAFatAgent::addSubAgent(const AgentData &, unsigned int, const std::shared_ptr<SubAgentData> &)
         */
        const unsigned int agent;
        /**
         * The name of the state within the associated agent
         */
        const std::string state;
        /**
         * Basic comparison operator, required for use of std::map etc
         */
        bool operator==(const AgentState &other) const {
            return (agent == other.agent && state == other.state);
        }
    };
    /**
     * Hash operator for AgentVariable, required for use of std::map etc
     */
    struct AgentState_hash {
        std::size_t operator()(const AgentState& k) const noexcept {
            return ((std::hash<unsigned int>()(k.agent)
                ^ (std::hash<std::string>()(k.state) << 1)) >> 1);
        }
    };

 public:
    /**
     * Constructs a new CUDAFatAgent, by creating a statelist for each of the provided agent's states
     * The specified agent becomes fat_index 0
     * @param description The initial agent to be represented by the CUDAFatAgent
     */
    explicit CUDAFatAgent(const AgentData& description);
    /**
     * Destructor
     * Frees any buffers allocated for new agents
     */
    ~CUDAFatAgent();
    /**
     * Adds a second agent to the CUDAFatAgent, this agent should be mapped to an agent already part of the CUDAFatAgent
     * @param description The agent to be added
     * @param master_fat_index fat index of the (parent) agent which this is agent is mapped to
     * @param mapping Mapping definition for how this agent is connected with its master agent.
     * @note It is expected that the new fat_index will be retrieved prior with getMappedAgentCount(), so this value is not returned
     */
    void addSubAgent(
      const AgentData &description,
      unsigned int master_fat_index,
      const std::shared_ptr<SubAgentData> &mapping);
    /**
     * This function builds and returns the state_map required by the named CUDAAgent
     * @param fat_index The index of the CUDAAgent within this CUDAFatAgent
     * @return a statemap suitable for the named agent
     */
    std::unordered_map<std::string, std::shared_ptr<CUDAFatAgentStateList>> getStateMap(unsigned int fat_index);
    /**
     * Scatters all active agents within the named state to remove agents with death flag set
     * This updates the alive agent count
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param state_name The name of the state attached to the named fat agent index
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void processDeath(unsigned int agent_fat_id, const std::string &state_name, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Transitions all active agents from the source state to the destination state
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param _src The name of the source state attached to the named fat agent index
     * @param _dest The name of the destination state attached to the named fat agent index
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void transitionState(unsigned int agent_fat_id, const std::string &_src, const std::string &_dest, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Reads the flags set by an agent function condition in order to sort agents according to whether they passed or failed
     * Failed agents are sorted to the front and marked as disabled, passing agents are then sorted to the back
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param state_name The name of the state attached to the named fat agent index
     * @param scatter Scatter instance and scan arrays to be used (CUDASimulation::singletons->scatter)
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     * @param stream CUDA stream to be used for async CUDA operations
     */
    void processFunctionCondition(unsigned int agent_fat_id, const std::string &state_name, CUDAScatter &scatter, unsigned int streamId, cudaStream_t stream);
    /**
     * Marks the specified number of agents within the specified statelist as disabled
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param state_name The name of the state attached to the named fat agent index
     * @param numberOfDisabled The number of agents to be marked as disabled
     */
    void setConditionState(unsigned int agent_fat_id, const std::string &state_name, unsigned int numberOfDisabled);
    /**
     * Returns a device pointer of atleast type_size x new_agents bytes available
     * @param total_agent_size Total number of bytes required to fit all variables in the agent
     * @param new_agents The maximum number of agents that need to be stored in the buffer
     * @param varCount The total number of different variables the agent has     *
     * @note It is assumed that when splitting the buffer into variables, each variable's sub-buffer will be 64 bit aligned
     * @note New buffers are shared between all states and mapped/unmapped agents
     */
    void *allocNewBuffer(size_t total_agent_size, unsigned int new_agents, size_t varCount);
    /**
     * Marks the named buffer as free
     * @param buff The buffer to free, this must be a pointer returned by allocNewBuffer(size_t, unsigned int, size_t)
     */
    void freeNewBuffer(void *buff);
    /**
     * The number of mapped agents currently represented by this CUDAFatAgent
     */
    unsigned int getMappedAgentCount() const;
    /**
     * Returns the next free agent id, and increments the ID tracker by the specified count
     * @param count Number that will be added to the return value on next call to this function
     * @return An ID that can be assigned to an agent that wil be stored within this CUDAFatAgent
     */
    id_t nextID(unsigned int count = 1);
    /**
     * Returns a device pointer to the value returns by nextID(0)
     * If the device value is changed, then the internal ID counter must be updated via CUDAAgent::scatterNew()
     */
    id_t *getDeviceNextID();
    /**
     * After device agent births have been processed, this function should be called by CUDAAgent::scatterNew()
     * It updated _nextID and hd_nextID, based on the assumed changes to d_nextID
     * @param newCount The number of agents birthed on device
     */
    void notifyDeviceBirths(unsigned int newCount);
    /**
     * Assigns IDs to any agents who's ID has the value ID_NOT_SET
     * @param hostapi HostAPI object, this is used to provide cub temp storage
     * @param scatter Scatter instance and cub temp memory to be used (CUDASimulation::singletons->scatter)
     * @param stream The CUDAStream to use for CUDA operations
     * @param streamId The stream index to use for accessing stream specific resources such as scan compaction arrays and buffers
     */
    void assignIDs(HostAPI& hostapi, CUDAScatter& scatter, cudaStream_t stream, unsigned int streamId);
    /**
     * Resets the flag agent_ids_have_init
     */
    void markIDsUnset() { agent_ids_have_init = false; }
    /**
     * Resets the ID counter (_nextID) to the first valid ID
     * Useful for submodels which will keep resetting an agent population
     * @note This will fail silently if it called if any state contains agents
     */
    void resetIDCounter();

 private:
    /**
     * Each agent-state pair maps to a CUDAFatStateList
     * Where an agent state is mapped, it will share CUDAFatStateList with another AgentState
     */
    std::unordered_map<AgentState, std::shared_ptr<CUDAFatAgentStateList>, AgentState_hash> states;
    /**
     * Each unique statelist owned by the CUDAFatAgent
     * This is used when applying operations to all mapped variables at once
     */
    std::set<std::shared_ptr<CUDAFatAgentStateList>> states_unique;

    /**
     * Represents a buffer stored in d_newLists
     */
    struct NewBuffer {
        size_t size;
        void *data;
        bool in_use;
        bool operator<(const NewBuffer &other) const { return size < other.size; }
    };
    /**
     * States share newlists, they are created on demand
     * They are are sorted so smallest buffer comes first
     * Potentially even move these upto CUDAAgent model level?
     */
    std::multiset<NewBuffer> d_newLists;
    /**
     * Mutex for accessing d_newLists
     */
    std::mutex d_newLists_mutex;
    /**
     * Counts the total number of agents represented by this CUDAFatAgent
     * This is used to assign fat_index's to newly represented agents
     */
    unsigned int mappedAgentCount;
    /**
     * Holds the next ID to be assigned
     * IDs are assigned in consecutive order, however it cannot be guaranteed that IDs will always be consecutive due to agent death, parallel birth and multiple states.
     */
    id_t _nextID;
    /**
     * Device mirror of _nextID(), allocated when first requested
     */
    id_t *d_nextID;
    /**
     * Host copy of the value of d_nextID
     * This is updated when a user requests d_nextID, or notifies of device births
     */
    id_t hd_nextID;
    /**
     * Set to false whenever an agent population is imported from outside
     * Checked before init functions and when step() is called by a user
     */
    bool agent_ids_have_init = true;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAFATAGENT_H_
