#include "flamegpu/model/SubEnvironmentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/model/EnvironmentDescription.h"

namespace flamegpu {

CSubEnvironmentDescription::CSubEnvironmentDescription(std::shared_ptr<SubEnvironmentData> data)
    : subenvironment(std::move(data)) { }
CSubEnvironmentDescription::CSubEnvironmentDescription(std::shared_ptr<const SubEnvironmentData> data)
    : subenvironment(std::move(std::const_pointer_cast<SubEnvironmentData>(data))) { }

bool CSubEnvironmentDescription::operator==(const CSubEnvironmentDescription& rhs) const {
    return *this->subenvironment == *rhs.subenvironment;  // Compare content is functionally the same
}
bool CSubEnvironmentDescription::operator!=(const CSubEnvironmentDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
std::string CSubEnvironmentDescription::getPropertyMapping(const std::string& sub_property_name) const {
    const auto v = subenvironment->properties.find(sub_property_name);
    if (v != subenvironment->properties.end())
        return v->second;
    THROW exception::InvalidAgentState("SubEnvironment property '%s', either does not exist or has not been mapped yet, "
        "in SubEnvironmentDescription::getPropertyMapping()\n", sub_property_name.c_str());
}
std::string CSubEnvironmentDescription::getMacroPropertyMapping(const std::string& sub_property_name) const {
    const auto v = subenvironment->macro_properties.find(sub_property_name);
    if (v != subenvironment->properties.end())
        return v->second;
    THROW exception::InvalidAgentState("SubEnvironment macro property '%s', either does not exist or has not been mapped yet, "
        "in SubEnvironmentDescription::getMacroPropertyMapping()\n", sub_property_name.c_str());
}

/**
 * Constructors
 */
SubEnvironmentDescription::SubEnvironmentDescription(std::shared_ptr<SubEnvironmentData> data)
    : CSubEnvironmentDescription(std::move(data)) { }

bool SubEnvironmentDescription::operator==(const CSubEnvironmentDescription& rhs) const {
    return rhs == *this;  // Forward to superclass's equality
}
bool SubEnvironmentDescription::operator!=(const CSubEnvironmentDescription& rhs) const {
    return !(*this == rhs);
}

void SubEnvironmentDescription::mapProperty(const std::string &sub_property_name, const std::string &master_property_name) {
    // Neither are reserved properties
    if (!sub_property_name.empty() && sub_property_name[0] == '_') {
        THROW exception::ReservedName("Sub-model environment property '%s is internal and cannot be mapped, "
            "in SubEnvironmentDescription::mapProperty()\n", sub_property_name.c_str());
    }
    if (!master_property_name.empty() && master_property_name[0] == '_') {
        THROW exception::ReservedName("Master-model environment property '%s is internal and cannot be mapped, "
            "in SubEnvironmentDescription::mapProperty()\n", master_property_name.c_str());
    }
    // Sub property exists
    auto subEnv = subenvironment->subEnvironment.lock();
    if (!subEnv) {
        THROW exception::InvalidParent("SubEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapProperty()\n");
    }
    const auto subProp = subEnv->properties.find(sub_property_name);
    if (subProp == subEnv->properties.end()) {
        const auto parent = subenvironment->parent.lock();
        THROW exception::InvalidEnvProperty("SubModel '%s's Environment does not contain property '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", parent ? parent->submodel->name.c_str() : "?", sub_property_name.c_str());
    }
    // Master property exists
    auto masterEnv = subenvironment->masterEnvironment.lock();
    if (!masterEnv) {
        THROW exception::InvalidParent("MasterEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapProperty()\n");
    }
    const auto masterProp = masterEnv->properties.find(master_property_name);
    if (masterProp == masterEnv->properties.end()) {
        THROW exception::InvalidEnvProperty("MasterEnvironment does not contain property '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", master_property_name.c_str());
    }
    // Sub property has not been bound yet
    if (subenvironment->properties.find(sub_property_name) != subenvironment->properties.end()) {
        const auto parent = subenvironment->parent.lock();
        THROW exception::InvalidEnvProperty("SubModel '%s's Environment property '%s' has already been mapped, "
            "in SubEnvironmentDescription::mapProperty()\n", parent ? parent->submodel->name.c_str() : "?", sub_property_name.c_str());
    }
    // Master property has already been bound
    for (auto &v : subenvironment->properties) {
        if (v.second == master_property_name) {
            THROW exception::InvalidEnvProperty("MasterEnvironment property '%s' has already been mapped, "
                "in SubEnvironmentDescription::mapProperty()\n", master_property_name.c_str());
        }
    }
    // Check properties are the same
    if (subProp->second.data.type != masterProp->second.data.type) {
        THROW exception::InvalidEnvProperty("Property types do not match, '%s' != '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", subProp->second.data.type.name(), masterProp->second.data.type.name());
    }
    if (subProp->second.data.elements != masterProp->second.data.elements) {
        THROW exception::InvalidEnvProperty("Property lengths do not match, '%u' != '%u'",
            "in SubEnvironmentDescription::mapProperty()\n", subProp->second.data.elements, masterProp->second.data.elements);
    }
    if (masterProp->second.isConst && !subProp->second.isConst) {
        THROW exception::InvalidEnvProperty("SubEnvironment property '%s' must be const, if mapped to const MasterEnvironment property '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", sub_property_name.c_str(), master_property_name.c_str());
    }
    // Properties match, create mapping
    subenvironment->properties.emplace(sub_property_name, master_property_name);
}
void SubEnvironmentDescription::mapMacroProperty(const std::string& sub_property_name, const std::string& master_property_name) {
    // Neither are reserved properties
    if (!sub_property_name.empty() && sub_property_name[0] == '_') {
        THROW exception::ReservedName("Sub-model environment macro property '%s is internal and cannot be mapped, "
            "in SubEnvironmentDescription::mapMacroProperty()\n", sub_property_name.c_str());
    }
    if (!master_property_name.empty() && master_property_name[0] == '_') {
        THROW exception::ReservedName("Master-model environment macro property '%s is internal and cannot be mapped, "
            "in SubEnvironmentDescription::mapMacroProperty()\n", master_property_name.c_str());
    }
    // Sub macro property exists
    auto subEnv = subenvironment->subEnvironment.lock();
    if (!subEnv) {
        THROW exception::InvalidParent("SubEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapMacroProperty()\n");
    }
    const auto subProp = subEnv->macro_properties.find(sub_property_name);
    if (subProp == subEnv->macro_properties.end()) {
        const auto parent = subenvironment->parent.lock();
        THROW exception::InvalidEnvProperty("SubModel '%s's Environment does not contain macro property '%s', "
            "in SubEnvironmentDescription::mapMacroProperty()\n", parent ? parent->submodel->name.c_str() : "?", sub_property_name.c_str());
    }
    // Master macro property exists
    auto masterEnv = subenvironment->masterEnvironment.lock();
    if (!masterEnv) {
        THROW exception::InvalidParent("MasterEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapMacroProperty()\n");
    }
    const auto masterProp = masterEnv->macro_properties.find(master_property_name);
    if (masterProp == masterEnv->macro_properties.end()) {
        THROW exception::InvalidEnvProperty("MasterEnvironment does not contain macro property '%s', "
            "in SubEnvironmentDescription::mapMacroProperty()\n", master_property_name.c_str());
    }
    // Sub macro property has not been bound yet
    if (subenvironment->macro_properties.find(sub_property_name) != subenvironment->macro_properties.end()) {
        const auto parent = subenvironment->parent.lock();
        THROW exception::InvalidEnvProperty("SubModel '%s's Environment macro property '%s' has already been mapped, "
            "in SubEnvironmentDescription::mapMacroProperty()\n", parent ? parent->submodel->name.c_str() : "?", sub_property_name.c_str());
    }
    // Master macro property has already been bound
    for (auto& v : subenvironment->macro_properties) {
        if (v.second == master_property_name) {
            THROW exception::InvalidEnvProperty("MasterEnvironment macro property '%s' has already been mapped, "
                "in SubEnvironmentDescription::mapMacroProperty()\n", master_property_name.c_str());
        }
    }
    // Check macro properties are the same
    if (subProp->second.type != masterProp->second.type) {
        THROW exception::InvalidEnvProperty("Macro property types do not match, '%s' != '%s', "
            "in SubEnvironmentDescription::mapMacroProperty()\n", subProp->second.type.name(), masterProp->second.type.name());
    }
    if (subProp->second.elements != masterProp->second.elements) {
        THROW exception::InvalidEnvProperty("Macro property dimensions do not match, (%u, %u, %u, %u) != (%u, %u, %u, %u)",
            "in SubEnvironmentDescription::mapMacroProperty()\n",
            subProp->second.elements[0], subProp->second.elements[1], subProp->second.elements[2], subProp->second.elements[3],
            masterProp->second.elements[0], masterProp->second.elements[1], masterProp->second.elements[2], masterProp->second.elements[3]);
    }
    // Macro properties match, create mapping
    subenvironment->macro_properties.emplace(sub_property_name, master_property_name);
}


void SubEnvironmentDescription::autoMap() {
    autoMapProperties();
    autoMapMacroProperties();
}
void SubEnvironmentDescription::autoMapProperties() {
    // Sub property exists
    auto subEnv = subenvironment->subEnvironment.lock();
    if (!subEnv) {
        THROW exception::InvalidParent("SubEnvironment pointer has expired, "
            "in SubEnvironmentDescription::autoMapProperties()\n");
    }
    // Master property exists
    auto masterEnv = subenvironment->masterEnvironment.lock();
    if (!masterEnv) {
        THROW exception::InvalidParent("MasterEnvironment pointer has expired, "
            "in SubEnvironmentDescription::autoMapProperties()\n");
    }
    for (auto &subProp : subEnv->properties) {
        // If it's a reserved environment property, don't map it
        // (Previously, there was a bug where _stepCount was being mixed between parent and child models)
        if (subProp.first[0] == '_')
            continue;
        auto masterProp = masterEnv->properties.find(subProp.first);
        // If there exists variable with same name in both environments
        if (masterProp != masterEnv->properties.end()) {
            // Check properties are the same
            if ((subProp.second.data.type == masterProp->second.data.type) &&
                (subProp.second.data.elements == masterProp->second.data.elements) &&
                !(masterProp->second.isConst && !subProp.second.isConst)) {
                subenvironment->properties.emplace(subProp.first, masterProp->first);  // Doesn't actually matter, both strings are equal
            }
        }
    }
}
void SubEnvironmentDescription::autoMapMacroProperties() {
    // Sub property exists
    auto subEnv = subenvironment->subEnvironment.lock();
    if (!subEnv) {
        THROW exception::InvalidParent("SubEnvironment pointer has expired, "
            "in SubEnvironmentDescription::autoMapMacroProperties()\n");
    }
    // Master property exists
    auto masterEnv = subenvironment->masterEnvironment.lock();
    if (!masterEnv) {
        THROW exception::InvalidParent("MasterEnvironment pointer has expired, "
            "in SubEnvironmentDescription::autoMapMacroProperties()\n");
    }
    for (auto& subProp : subEnv->macro_properties) {
        // If it's a reserved environment property, don't map it
        if (subProp.first[0] == '_')
            continue;
        auto masterProp = masterEnv->macro_properties.find(subProp.first);
        // If there exists variable with same name in both environments
        if (masterProp != masterEnv->macro_properties.end()) {
            // Check properties are the same
            if ((subProp.second.type == masterProp->second.type) &&
                (subProp.second.elements == masterProp->second.elements)) {
                subenvironment->macro_properties.emplace(subProp.first, masterProp->first);  // Doesn't actually matter, both strings are equal
            }
        }
    }
}

}  // namespace flamegpu
