#ifndef INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDESCRIPTION_H_
#define INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDESCRIPTION_H_

#include <memory>
#include <string>

namespace flamegpu {

struct ModelData;
struct SubModelData;
struct SubEnvironmentData;

class CSubEnvironmentDescription {
    /**
     * Data store class for this description, constructs instances of this class
     */
    friend struct SubEnvironmentData;

 public:
    /**
     * Constructor, creates an interface to the SubEnvironmentData
     * @param data Data store of this subenvironment's data
     */
    explicit CSubEnvironmentDescription(std::shared_ptr<SubEnvironmentData> data);
    explicit CSubEnvironmentDescription(std::shared_ptr<const SubEnvironmentData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same SubEnvironmentData/ModelData
     */
    CSubEnvironmentDescription(const CSubEnvironmentDescription& other_agent) = default;
    CSubEnvironmentDescription(CSubEnvironmentDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same SubEnvironmentData/ModelData
     */
    CSubEnvironmentDescription& operator=(const CSubEnvironmentDescription& other_agent) = default;
    CSubEnvironmentDescription& operator=(CSubEnvironmentDescription&& other_agent) = default;
    /**
     * Equality operator, checks whether SubEnvironmentDescription hierarchies are functionally the same
     * @param rhs right hand side
     * @returns True when subenvironments are the same
     * @note Instead compare pointers if you wish to check that they are the same instance
     */
    bool operator==(const CSubEnvironmentDescription& rhs) const;
    /**
     * Equality operator, checks whether SubEnvironmentDescription hierarchies are functionally different
     * @param rhs right hand side
     * @returns True when subenvironments are not the same
     * @note Instead compare pointers if you wish to check that they are not the same instance
     */
    bool operator!=(const CSubEnvironmentDescription& rhs) const;
    /**
     * Returns the name of the master property which has been mapped to the named subenvironment property
     * @param sub_property_name Name of the state in the sub agent to check
     * @return The name of the state within the master agent which is mapped
     * @throws exception::InvalidEnvProperty If the sub environment property does not exist or has not been mapped yet
     */
    std::string getPropertyMapping(const std::string& sub_property_name) const;
    /**
     * Returns the name of the master macro property which has been mapped to the named subenvironment macro property
     * @param sub_property_name Name of the state in the sub agent to check
     * @return The name of the state within the master agent which is mapped
     * @throws exception::InvalidEnvProperty If the sub environment property does not exist or has not been mapped yet
     */
    std::string getMacroPropertyMapping(const std::string& sub_property_name) const;

 protected:
    /**
     * The class which stores all of the subenvironment's data.
     */
    std::shared_ptr<SubEnvironmentData> subenvironment;
};
/**
 * This class provides an interface to a mapping between a parent and submodel's environment properties
 */
class SubEnvironmentDescription : public CSubEnvironmentDescription {
 public:
    /**
     * Constructor, creates an interface to the SubEnvironmentData
     * @param data Data store of this subenvironment's data
     */
    explicit SubEnvironmentDescription(std::shared_ptr<SubEnvironmentData> data);
    /**
     * Copy constructor
     * Creates a new interface to the same SubEnvironmentData/ModelData
     */
    SubEnvironmentDescription(const SubEnvironmentDescription& other_agent) = default;
    SubEnvironmentDescription(SubEnvironmentDescription&& other_agent) = default;
    /**
     * Assignment operator
     * Assigns this interface to the same SubEnvironmentData/ModelData
     */
    SubEnvironmentDescription& operator=(const SubEnvironmentDescription& other_agent) = default;
    SubEnvironmentDescription& operator=(SubEnvironmentDescription&& other_agent) = default;

    /**
     * Automatically map all compatible properties and macro properties
     * In order to be compatible, properties must share the same name, type, dimensions/length (number of elements)
     * Const master properties cannot be mapped to non-const sub properties, however the inverse is permitted
     */
    void autoMap();
    /**
     * Automatically map all compatible properties
     * In order to be compatible, properties must share the same name, type, length (number of elements)
     * Const master properties cannot be mapped to non-const sub properties, however the inverse is permitted
     */
    void autoMapProperties();
    /**
     * Automatically map all compatible macro properties
     * In order to be compatible, properties must share the same name, type, dimensions
     */
    void autoMapMacroProperties();
    /**
     * Links the named properties between the master and sub environment
     * In order to be compatible, properties must share the same name, type, length (number of elements)
     * Const master properties cannot be mapped to non-const sub properties, however the inverse is permitted
     * @param sub_property_name Name of the property in the sub models agent
     * @param master_property_name Name of the property in the master models agent
     * @throws exception::InvalidParent If the sub agent or master agent weak_ptrs have expired (this should never happen)
     * @throws exception::InvalidEnvProperty If the named property does not exist within the bound sub or master environment
     * @throws exception::InvalidEnvProperty If the named properties do not share the same type and length
     */
    void mapProperty(const std::string &sub_property_name, const std::string &master_property_name);
    /**
     * Links the named macro properties between the master and sub environment
     * In order to be compatible, macro properties must share the same name, type, dimensions
     * @param sub_property_name Name of the macro property in the sub models agent
     * @param master_property_name Name of the macro property in the master models agent
     * @throws exception::InvalidParent If the sub agent or master agent weak_ptrs have expired (this should never happen)
     * @throws exception::InvalidEnvProperty If the named macro property does not exist within the bound sub or master environment
     * @throws exception::InvalidEnvProperty If the named macro properties do not share the same type and length
     */
    void mapMacroProperty(const std::string& sub_property_name, const std::string& master_property_name);
};

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_MODEL_SUBENVIRONMENTDESCRIPTION_H_
