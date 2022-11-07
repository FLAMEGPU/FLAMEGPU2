#ifndef INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_
#include <string>
#include <memory>

#include "LayerDescription.h"
#include "DependencyNode.h"

namespace flamegpu {

class CSubAgentDescription;
class SubAgentDescription;
class SubEnvironmentDescription;
struct ModelData;
struct SubModelData;

class CSubModelDescription : public DependencyNode {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct SubModelData;
    /**
     * Accesses internals to validate function description before adding to layer
     */
    friend void LayerDescription::addSubModel(const CSubModelDescription&);

 public:
    /**
     * Constructor, creates an interface to the SubModelData
     * @param data Data store of this submodel's data
     */
    explicit CSubModelDescription(std::shared_ptr<SubModelData> data);
    explicit CSubModelDescription(std::shared_ptr<const SubModelData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same SubModelData/ModelData
     */
    CSubModelDescription(const CSubModelDescription& other_agent) = default;
    CSubModelDescription(CSubModelDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same SubModelData/ModelData
     */
    CSubModelDescription& operator=(const CSubModelDescription& other_agent) = default;
    CSubModelDescription& operator=(CSubModelDescription&& other_agent) = default;
    /**
     * Equality operator, checks whether SubModelDescription hierarchies are functionally the same
     * @param rhs right hand side
     * @returns True when submodels are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CSubModelDescription& rhs) const;
    /**
     * Equality operator, checks whether SubModelDescription hierarchies are functionally different
     * @param rhs right hand side
     * @returns True when submodels are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CSubModelDescription& rhs) const;
    /**
     * Return the current value of max steps, defaults to 0
     * This is the maximum number of steps per call of the submodel
     * 0 is unlimited, however requires the model to have an exit condition
     */
    unsigned int getMaxSteps() const;
    /**
     * Return the name assigned to the submodel
     * @note This may differ from the name of the ModelDescription for the submodel
     */
    const std::string getName() const;

 protected:
    /**
     * The class which stores all of the submodel's data.
     */
    std::shared_ptr<SubModelData> submodel;
};
/**
 * This class provides an interface to a mapping between the parent and sub-model
 */
class SubModelDescription : public CSubModelDescription {
 public:
    /**
     * Constructor, creates an interface to the SubModelData
     * @param data Data store of this agent's data
     */
    explicit SubModelDescription(std::shared_ptr<SubModelData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same SubModelData/ModelData
     */
    SubModelDescription(const SubModelDescription& other_agent) = default;
    SubModelDescription(SubModelDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same SubModelData/ModelData
     */
    SubModelDescription& operator=(const SubModelDescription& other_agent) = default;
    SubModelDescription& operator=(SubModelDescription&& other_agent) = default;
    /**
     * Equality operator, checks whether SubModelDescription hierarchies are functionally the same
     * @param rhs right hand side
     * @returns True when submodels are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CSubModelDescription& rhs) const;
    /**
     * Equality operator, checks whether SubModelDescription hierarchies are functionally different
     * @param rhs right hand side
     * @returns True when submodels are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CSubModelDescription& rhs) const;

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
     * @throws exception::InvalidSubAgentName If the sub agent name does not map to a valid agent
     * @throws exception::InvalidAgentName If the  master agent has already been bound
     * @throws exception::InvalidSubAgentName If the sub agent name does not map to a valid agent
     * @throws exception::InvalidAgentName If the master agent has already been bound
     */
    SubAgentDescription bindAgent(const std::string &sub_agent_name, const std::string &master_agent_name, bool auto_map_vars = false, bool auto_map_states = true);
    /**
     * Returns a mutable reference to the named SubAgent description if it has already been bound to a master agent
     * @param sub_agent_name Name of the sub agent, who's description to return
     * @return A mutable reference to the named SubAgent description
     * @throws exception::InvalidSubAgentName If the sub_agent_name does not exist within the sub_model and/or has not been bound yet
     * @see SubModelDescription::getSubAgent(const std::string &) for the immutable version
     */
    SubAgentDescription SubAgent(const std::string &sub_agent_name);
    /**
     * Returns an immutable reference to the named SubAgent description if it has already been bound to a master agent
     * @param sub_agent_name Name of the sub agent, who's description to return
     * @return An immutable reference to the named SubAgent description
     * @throws exception::InvalidSubAgentName If the sub_agent_name does not exist within the sub_model and/or has not been bound yet
     * @see SubModelDescription::SubAgent(const std::string &) for the mutable version
     */
    CSubAgentDescription getSubAgent(const std::string &sub_agent_name) const;
    /**
     * Returns an interface for configuring mapped environment properties
     * @param auto_map If true is passed, all properties and macro properties with matching name, type and length between models will be mapped
     */
    SubEnvironmentDescription &SubEnvironment(bool auto_map = false);
    /**
     * Returns an immutable interface for viewing the configuration of mapped environment properties
     * @param auto_map If true is passed, all properties and macro properties with matching name, type and length between models will be mapped
     */
    const SubEnvironmentDescription &getSubEnvironment(bool auto_map = false) const;
    /**
     * Set the maximum number of steps per execution of the submodel
     * If 0 (default), unlimited however an exit condition is required
     */
    void setMaxSteps(unsigned int max_steps);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBMODELDESCRIPTION_H_
