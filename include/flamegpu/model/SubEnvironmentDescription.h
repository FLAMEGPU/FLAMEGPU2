#ifndef INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDESCRIPTION_H_

#include <memory>
#include <string>

namespace flamegpu {

struct ModelData;
struct SubModelData;
struct SubEnvironmentData;

/**
 * This class provides an interface to a mapping between a parent and submodel's environment properties
 */
class SubEnvironmentDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct SubEnvironmentData;
    /**
     * Constructor, this should only be called by AgentData
     * @param model Model at root of model hierarchy
     * @param data Data store of this subagent's data
     */
    SubEnvironmentDescription(const std::shared_ptr<const ModelData> &model, SubEnvironmentData *const data);
    /**
     * Default copy constructor, not implemented
     */
    SubEnvironmentDescription(const SubEnvironmentDescription &other_agent) = delete;
    /**
     * Default move constructor, not implemented
     */
    SubEnvironmentDescription(SubEnvironmentDescription &&other_agent) noexcept = delete;
    /**
     * Default copy assignment, not implemented
     */
    SubEnvironmentDescription& operator=(const SubEnvironmentDescription &other_agent) = delete;
    /**
     * Default move assignment, not implemented
     */
    SubEnvironmentDescription& operator=(SubEnvironmentDescription &&other_agent) noexcept = delete;

 public:
    /**
     * Automatically map all compatible properties
     * In order to be compatible, properties must share the same name, type, length (number of elements)
     * Const master properties cannot be mapped to non-const sub properties, however the inverse is permitted
     */
    void autoMapProperties();
    /**
     * Links the named properties between the master and sub environment
     * In order to be compatible, properties must share the same name, type, length (number of elements)
     * Const master properties cannot be mapped to non-const sub properties, however the inverse is permitted
     * @param sub_property_name Name of the property in the sub models agent
     * @param master_property_name Name of the property in the master models agent
     * @throws InvalidParent If the sub agent or master agent weak_ptrs have expired (this should never happen)
     * @throws InvalidEnvProperty If the named property does not exist within the bound sub or master environment
     * @throws InvalidEnvProperty If the named properties do not share the same type and length
     */
    void mapProperty(const std::string &sub_property_name, const std::string &master_property_name);
    /**
     * Returns the master agent property which has been mapped to the name subagent state
     * @param sub_property_name Name of the state in the sub agent to check
     * @return The name of the state within the master agent which is mapped
     * @throws InvalidEnvProperty If the sub environment property does not exist or has not been mapped yet
     */
    std::string getPropertyMapping(const std::string &sub_property_name);

 private:
    /**
     * Root of the model hierarchy
     */
    const std::weak_ptr<const ModelData> model;
    /**
     * The class which stores all of the agent's data.
     */
    SubEnvironmentData *const data;
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDESCRIPTION_H_
