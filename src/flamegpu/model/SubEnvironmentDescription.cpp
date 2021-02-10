#include "flamegpu/model/SubEnvironmentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/model/EnvironmentDescription.h"

SubEnvironmentDescription::SubEnvironmentDescription(const std::shared_ptr<const ModelData> &_model, SubEnvironmentData *const _data)
    : model(_model)
    , data(_data) { }

void SubEnvironmentDescription::mapProperty(const std::string &sub_property_name, const std::string &master_property_name) {
    // Sub property exists
    auto subEnv = data->subEnvironment.lock();
    if (!subEnv) {
        THROW InvalidParent("SubEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapProperty()\n");
    }
    const auto subProp = subEnv->properties.find(sub_property_name);
    if (subProp == subEnv->properties.end()) {
        const auto parent = data->parent.lock();
        THROW InvalidEnvProperty("SubModel '%s's Environment does not contain property '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", parent ? parent->submodel->name.c_str() : "?", sub_property_name.c_str());
    }
    // Master property exists
    auto masterEnv = data->masterEnvironment.lock();
    if (!masterEnv) {
        THROW InvalidParent("MasterEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapProperty()\n");
    }
    const auto masterProp = masterEnv->properties.find(master_property_name);
    if (masterProp == masterEnv->properties.end()) {
        THROW InvalidEnvProperty("MasterEnvironment does not contain property '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", master_property_name.c_str());
    }
    // Sub property has not been bound yet
    if (data->properties.find(sub_property_name) != data->properties.end()) {
        const auto parent = data->parent.lock();
        THROW InvalidEnvProperty("SubModel '%s's Environment property '%s' has already been mapped, "
            "in SubEnvironmentDescription::mapProperty()\n", parent ? parent->submodel->name.c_str() : "?", sub_property_name.c_str());
    }
    // Master property has already been bound
    for (auto &v : data->properties) {
        if (v.second == master_property_name) {
            THROW InvalidEnvProperty("MasterEnvironment property '%s' has already been mapped, "
                "in SubEnvironmentDescription::mapProperty()\n", master_property_name.c_str());
        }
    }
    // Check properties are the same
    if (subProp->second.data.type != masterProp->second.data.type) {
        THROW InvalidEnvProperty("Property types do not match, '%s' != '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", subProp->second.data.type.name(), masterProp->second.data.type.name());
    }
    if (subProp->second.data.elements != masterProp->second.data.elements) {
        THROW InvalidEnvProperty("Property lengths do not match, '%u' != '%u'",
            "in SubEnvironmentDescription::mapProperty()\n", subProp->second.data.elements, masterProp->second.data.elements);
    }
    if (masterProp->second.isConst && !subProp->second.isConst) {
        THROW InvalidEnvProperty("SubEnvironment property '%s' must be const, if mapped to const MasterEnvironment property '%s', "
            "in SubEnvironmentDescription::mapProperty()\n", sub_property_name.c_str(), master_property_name.c_str());
    }
    // Properties match, create mapping
    data->properties.emplace(sub_property_name, master_property_name);
}

std::string SubEnvironmentDescription::getPropertyMapping(const std::string &sub_property_name) {
    const auto v = data->properties.find(sub_property_name);
    if (v != data->properties.end())
        return v->second;
    THROW InvalidAgentState("SubEnvironment property '%s', either does not exist or has not been mapped yet, "
        "in SubAgentDSubEnvironmentDescriptionescription::getPropertyMapping()\n", sub_property_name.c_str());
}


void SubEnvironmentDescription::autoMapProperties() {
    // Sub property exists
    auto subEnv = data->subEnvironment.lock();
    if (!subEnv) {
        THROW InvalidParent("SubEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapProperty()\n");
    }
    // Master property exists
    auto masterEnv = data->masterEnvironment.lock();
    if (!masterEnv) {
        THROW InvalidParent("MasterEnvironment pointer has expired, "
            "in SubEnvironmentDescription::mapProperty()\n");
    }
    for (auto &subProp : subEnv->properties) {
        auto masterProp = masterEnv->properties.find(subProp.first);
        // If there exists variable with same name in both environments
        if (masterProp != masterEnv->properties.end()) {
            // Check properties are the same
            if ((subProp.second.data.type == masterProp->second.data.type) &&
                (subProp.second.data.elements == masterProp->second.data.elements) &&
                !(masterProp->second.isConst && !subProp.second.isConst)) {
                data->properties.emplace(subProp.first, masterProp->first);  // Doesn't actually matter, both strings are equal
            }
        }
    }
}
