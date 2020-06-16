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

/**
 * This holds the data for all agents and agent states which are mapped to one another
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
        const unsigned int agent;
        const std::string state;

      bool operator==(const AgentState &other) const
      { return (agent == other.agent
                && state == other.state);
      }
    };
    /**
     * Hash operator for AgentVariable
    */
    struct AgentState_hash {
        std::size_t operator()(const AgentState& k) const noexcept {
            return ((std::hash<unsigned int>()(k.agent)
                ^ (std::hash<std::string>()(k.state) << 1)) >> 1);
        }
    };

 public:
    explicit CUDAFatAgent(const AgentData& description);
    ~CUDAFatAgent();
    /**
     * This function builds and returns the state_map required by the named CUDAAgent
     * @param fat_index The index of the CUDAAgent within this CUDAFatAgent
     * @return a statemap suitable for the named agent
     */
    std::unordered_map<std::string, std::shared_ptr<CUDAFatAgentStateList>> getStateMap(const unsigned int &fat_index);
    /**
     * Processes a mapped agent
     */
    void addSubAgent(
      const AgentData &description,
      const unsigned int &master_fat_index,
      const std::shared_ptr<SubAgentData> &mapping);
    /**
     * Scatters all active agents within the named state to remove agents with death flag set
     * This updates the alive agent count
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param state_name The name of the state attached to the named fat agent index
     * @param streamId Index of the stream used for scan compaction flags (and async cuda ops)
     */
    void processDeath(const unsigned int &agent_fat_id, const std::string &state_name, const unsigned int &streamId);
    /**
     * Transitions all active agents from the source state to the destination state
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param _src The name of the source state attached to the named fat agent index
     * @param _dest The name of the destination state attached to the named fat agent index
     * @param streamId Index of the stream used for scan compaction flags (and async cuda ops)
     */
    void transitionState(const unsigned int &agent_fat_id, const std::string &_src, const std::string &_dest, const unsigned int &streamId);
    /**
     * Reads the flags set by an agent function condition in order to sort agents according to whether they passed or failed
     * Failed agents are sorted to the front and marked as disabled, passing agents are then sorted to the back
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param state_name The name of the state attached to the named fat agent index
     * @param streamId Index of the stream used for scan compaction flags (and async cuda ops)
     */
    void processFunctionCondition(const unsigned int &agent_fat_id, const std::string &state_name, const unsigned int &streamId);
    /**
     * Marks the specified number of agents within the specified statelist as disabled
     * @param agent_fat_id The index of the CUDAAgent within this CUDAFatAgent
     * @param state_name The name of the state attached to the named fat agent index
     * @param numberOfDisabled The number of agents to be marked as disabled
     */
    void setConditionState(const unsigned int &agent_fat_id, const std::string &state_name, const unsigned int numberOfDisabled);
    /**
     * Returns a device pointer of atleast type_size x new_agents bytes available
     *
     * @note New buffers are shared between all states and mapped/unmapped agents
     */
    void *allocNewBuffer(const size_t &total_agent_size, const unsigned int &new_agents);
    /**
     * Marks the named buffer as free
     */
    void freeNewBuffer(void *buff);
    unsigned int getMappedAgentCount() const;


 private:
    /**
     * Each agent-state pair maps to a CUDAFatStateList
     * Where an agent state is mapped, it will share CUDAFatStateList with another AgentState
     */
    std::unordered_map<AgentState, std::shared_ptr<CUDAFatAgentStateList>, AgentState_hash> states;
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
    std::mutex d_newLists_mutex;

    unsigned int mappedAgentCount;
};

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAFATAGENT_H_
