#ifndef INCLUDE_FLAMEGPU_MODEL_SUBMODELDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBMODELDATA_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "flamegpu/model/SubModelDescription.h"

struct ModelData;
struct SubAgentData;
struct AgentData;
struct SubEnvironmentData;

class SubModelDescription;

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
    std::shared_ptr<const ModelData> submodel;
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
     * The SubModel represented by this class
     */
    std::shared_ptr<SubEnvironmentData> subenvironment;
    /**
     * Name assigned to the submodel at creation
     */
    std::string name;
    /**
     * Description class which provides convenient accessors
     * This may be null if the instance has been cloned
     */
    std::unique_ptr<SubModelDescription> description;
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

 protected:
    /**
     * Copy constructor
     * This should only be called via clone();
     */
    explicit SubModelData(const std::shared_ptr<ModelData> &model, const SubModelData &other);
    /**
     * Normal constructor
     * This should only be called by ModelDescription
     */
    explicit SubModelData(const std::shared_ptr<ModelData> &model, const std::string &submodel_name, const std::shared_ptr<ModelData> &submodel);
};

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBMODELDATA_H_
