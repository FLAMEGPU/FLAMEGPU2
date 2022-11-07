#ifndef INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDATA_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDATA_H_

#include <memory>
#include <string>
#include <unordered_map>

namespace flamegpu {

struct ModelData;
struct SubModelData;
class EnvironmentDescription;
class SubEnvironmentDescription;

/**
 * Holds all the mappings between the environment properties in the submodel and the parent model
 * Properties of matching type and length can be mapped
 */
struct SubEnvironmentData : std::enable_shared_from_this<SubEnvironmentData> {
    /**
     * Description needs full access
     */
    friend class SubEnvironmentDescription;
    /**
     * SubModelDescription needs access to private constructor
     */
    friend class SubModelDescription;
    /**
     * SubModelData needs access to private copy constructor
     */
    friend struct ModelData;
    /**
     * Parent model
     */
    std::weak_ptr<const ModelData> model;
    /**
     * The sub environment which is bound
     */
    std::weak_ptr<EnvironmentDescription> subEnvironment;
    /**
    * The master environment which is bound
    */
    std::weak_ptr<EnvironmentDescription> masterEnvironment;
    /**
    * Map of submodel item name:parent model item name
    * map<string, string>
    */
    typedef std::unordered_map<std::string, std::string> Mapping;
    /**
    * Holds all of the model's environment property mappings
    */
    Mapping properties;
    /**
    * Holds all of the model's environment macro property mappings
    */
    Mapping macro_properties;
    /**
     * The model which this environment is a member of
     */
    std::weak_ptr<SubModelData> parent;
    /**
    * Equality operator, checks whether SubAgentData hierarchies are functionally the same
    * @returns True when models are the same
    * @note Instead compare pointers if you wish to check that they are the same instance
    */
    bool operator==(const SubEnvironmentData& rhs) const;
    /**
    * Equality operator, checks whether SubAgentData hierarchies are functionally different
    * @returns True when models are not the same
    * @note Instead compare pointers if you wish to check that they are not the same instance
    */
    bool operator!=(const SubEnvironmentData& rhs) const;
    /**
    * Default copy constructor, not implemented
    */
    SubEnvironmentData(const SubEnvironmentData &other) = delete;

 protected:
    /**
    * Copy constructor
    * This should only be called via clone();
    */
    explicit SubEnvironmentData(std::shared_ptr<const ModelData> model, const std::shared_ptr<SubModelData> &parent, const SubEnvironmentData &other);
    /**
    * Normal constructor
    * This should only be called by SubModelDescription
    */
    explicit SubEnvironmentData(std::shared_ptr<const ModelData> model, const std::shared_ptr<SubModelData> &_parent, const std::shared_ptr<EnvironmentDescription> &subEnv);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDATA_H_
