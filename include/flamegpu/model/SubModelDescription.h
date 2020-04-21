#ifndef INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_
#include <string>

#include "LayerDescription.h"

class SubAgentDescription;
struct ModelData;
struct SubModelData;

/**
 * This class holds a submodel and mapping between the parent and sub-model
 */
class SubModelDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct SubModelData;
    /**
     * Accesses internals to validate function description before adding to layer
     */
    friend void LayerDescription::addSubModel(const SubModelDescription &);
    /**
     * Constructor, this should only be called by AgentData
     * @param _model Model at root of model hierarchy
     * @param _submodel Model at root of submodel hierarchy
     */
    SubModelDescription(const ModelData *const _model, SubModelData *const _submodel);
    /**
     * Default copy constructor, not implemented
     */
    SubModelDescription(const SubModelDescription &other_agent) = delete;
    /**
     * Default move constructor, not implemented
     */
    SubModelDescription(SubModelDescription &&other_agent) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    SubModelDescription& operator=(const SubModelDescription &other_agent) = delete;
    /**
     * Default move assignment, not implemented
     */
    SubModelDescription& operator=(SubModelDescription &&other_agent) noexcept = delete;

 public:
    /**
     * Defines which agent from the parent/master model will be mapped to which agent in the submodel
     * Only 1 master agent can be bound to each sub agent, however the same master agent can be bound to many sub agents
     * Binding an agent to a subagent, ensures the population of subagents is always the same size.
     * Furthermore, agent variables can be mapped so that they are shared between the two agents.
     * Returns SubAgentDescription which can be used to manually map variables
     * If auto_map_vars is enabled, variables with matching name, type and length will be automatically mapped
     * @param sub_agent_name Name of the agent in the submodel (must be unique, 1 bind per subagent)
     * @param master_agent_name Name of the agent in the parent/host/master model
     * @param auto_map_vars Whether to automatically map matching variables of the two agents
     * @param auto_map_states Whether to automatically map matching states of the two agents
     * @throws InvalidSubAgentName If the sub agent name does not map to a valid agent
     * @throws InvalidAgentName If the  master agent has already been bound
     * @throws InvalidSubAgentName If the sub agent name does not map to a valid agent
     * @throws InvalidAgentName If the master agent has already been bound
     */
    SubAgentDescription &bindAgent(const std::string &sub_agent_name, const std::string &master_agent_name, bool auto_map_vars = false, bool auto_map_states = true);
    /**
     * Returns a mutable reference to the named SubAgent description if it has already been bound to a master agent
     * @param sub_agent_name Name of the sub agent, who's description to return
     * @return A mutable reference to the named SubAgent description
     * @throws InvalidSubAgentName If the sub_agent_name does not exist within the sub_model and/or has not been bound yet
     * @see SubModelDescription::getSubAgent(const std::string &) for the immutable version
     */
    SubAgentDescription &SubAgent(const std::string &sub_agent_name);
    /**
     * Returns an immutable reference to the named SubAgent description if it has already been bound to a master agent
     * @param sub_agent_name Name of the sub agent, who's description to return
     * @return An immutable reference to the named SubAgent description
     * @throws InvalidSubAgentName If the sub_agent_name does not exist within the sub_model and/or has not been bound yet
     * @see SubModelDescription::SubAgent(const std::string &) for the mutable version
     */
    const SubAgentDescription &getSubAgent(const std::string &sub_agent_name) const;

 private:
    /**
     * Root of the model hierarchy
     */
    const ModelData *const model;
    /**
     * Root of the submodel hierarchy
     */
    SubModelData *const data;
};

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_
