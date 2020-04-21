#ifndef INCLUDE_FLAMEGPU_MODEL_SUBMODELDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBMODELDATA_H_

#include <memory>
#include <string>
#include <unordered_map>

struct ModelData;
struct SubAgentData;
struct AgentData;

class SubModelDescription;
class SubAgentDescription;

/**
 * Holds all the mappings between a submodel and master/parent/host model
 */
struct SubModelData : std::enable_shared_from_this<SubModelData> {
    /**
     * Description needs full access
     */
    friend class SubModelDescription;
    /**
     * ModelDescription needs access to private copy constructor
     */
    friend struct ModelData;
    /**
     * ModelDescription needs access to private constructor
     */
    friend class ModelDescription;
    /**
     * The SubModel represented by this class
     */
    std::shared_ptr<ModelData> submodel;
    /**
     * Map of name:subagent definition
     * map<string, SubAgentData>
     */
    typedef std::unordered_map<std::string, std::shared_ptr<SubAgentData>> SubAgentMap;
    /**
     * Holds all of the model's subagent definitions
     */
    SubAgentMap subagents;
    /**
     * Description class which provides convenient accessors
     * This may be null if the instance has been cloned
     */
    std::shared_ptr<SubModelDescription> description;
    /**
     * Equality operator, checks whether SubModelData hierarchies are functionally the same
     * @returns True when models are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const SubModelData& rhs) const;
    /**
     * Equality operator, checks whether SubModelData hierarchies are functionally different
     * @returns True when models are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const SubModelData& rhs) const;
    /**
     * Default copy constructor, not implemented
     */
    SubModelData(const SubModelData &other) = delete;
    /**
     * Returns a constant copy of this submodel's hierarchy
     * Does not copy description, sets it to nullptr instead
     */
    std::shared_ptr<const SubModelData> clone() const;

 protected:
    /**
     * Copy constructor
     * This should only be called via clone();
     */
    explicit SubModelData(const ModelData *model, const SubModelData &other);
    /**
     * Normal constructor
     * This should only be called by ModelDescription
     */
    explicit SubModelData(const ModelData *model, std::shared_ptr<ModelData> submodel);
};

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
    friend struct SubModelData;
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
    /**
     * Returns a constant copy of this subagent's hierarchy
     * Does not copy description, sets it to nullptr instead
     */
    std::shared_ptr<const SubAgentData> clone() const;

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

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBMODELDATA_H_
