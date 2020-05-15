#ifndef INCLUDE_FLAMEGPU_MODEL_SUBAGENTDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBAGENTDATA_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/model/SubModelData.h"

struct AgentData;

class SubAgentDescription;

/**
 * Holds all the mappings between a an agent in the submodel and the parent model
 * States and variables can be mapped
 */
struct SubAgentData : std::enable_shared_from_this<SubAgentData> {
    /**
    * Description needs full access
    */
    friend class SubAgentDescription;
    /**
    * SubModelDescription needs access to private constructor
    */
    friend class SubModelDescription;
    /**
    * SubModelData needs access to private copy constructor
    */
    friend struct ModelData;
    /**
    * The sub agent which is bound
    */
    std::weak_ptr<AgentData> subAgent;
    /**
    * The master agent which is bound
    */
    std::weak_ptr<AgentData> masterAgent;
    /**
    * Map of submodel item name:parent model item name
    * map<string, string>
    */
    typedef std::unordered_map<std::string, std::string> Mapping;
    /**
    * Holds all of the model's variable mappings
    */
    Mapping variables;
    /**
    * Holds all of the model's state mappings
    */
    Mapping states;
    /**
    * The agent which this function is a member of
    */
    std::weak_ptr<SubModelData> parent;
    /**
    * Description class which provides convenient accessors
    * This may be null if the instance has been cloned
    */
    std::shared_ptr<SubAgentDescription> description;
    /**
    * Equality operator, checks whether SubAgentData hierarchies are functionally the same
    * @returns True when models are the same
    * @note Instead compare pointers if you wish to check that they are the same instance
    */
    bool operator==(const SubAgentData& rhs) const;
    /**
    * Equality operator, checks whether SubAgentData hierarchies are functionally different
    * @returns True when models are not the same
    * @note Instead compare pointers if you wish to check that they are not the same instance
    */
    bool operator!=(const SubAgentData& rhs) const;
    /**
    * Default copy constructor, not implemented
    */
    SubAgentData(const SubAgentData &other) = delete;

 protected:
    /**
    * Copy constructor
    * This should only be called via clone();
    */
    explicit SubAgentData(const ModelData *model, const std::shared_ptr<SubModelData> &parent, const SubAgentData &other);
    /**
    * Normal constructor
    * This should only be called by SubModelDescription
    */
    explicit SubAgentData(const ModelData *model, const std::shared_ptr<SubModelData> &_parent, const std::shared_ptr<AgentData> &subAgent, const std::shared_ptr<AgentData> &masterAgent);
};

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBAGENTDATA_H_
