#include "flamegpu/model/EnvironmentDescription.h"

namespace flamegpu {

EnvironmentDescription::EnvironmentDescription() {
    // Add CUDASimulation specific environment members
    // We do this here, to not break comparing different model description hierarchies before/after CUDASimulation creation
    unsigned int zero = 0;
    newProperty("_stepCount", reinterpret_cast<const char*>(&zero), sizeof(unsigned int), false, 1, typeid(unsigned int));
}

bool EnvironmentDescription::operator==(const EnvironmentDescription& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (properties.size() == rhs.properties.size()) {
        for (auto &v : properties) {
            auto _v = rhs.properties.find(v.first);
            if (_v == rhs.properties.end())
                return false;
            if (v.second != _v->second)
                return false;
        }
        return true;
    }
    if (macro_properties.size() == rhs.macro_properties.size()) {
        for (auto& v : macro_properties) {
            auto _v = rhs.macro_properties.find(v.first);
            if (_v == rhs.macro_properties.end())
                return false;
            if (v.second != _v->second)
                return false;
        }
        return true;
    }
    return false;
}
bool EnvironmentDescription::operator!=(const EnvironmentDescription& rhs) const {
    return !(*this == rhs);
}

void EnvironmentDescription::newProperty(const std::string &name, const char *ptr, size_t length, bool isConst, flamegpu::size_type elements, const std::type_index &type) {
    properties.emplace(name, PropData(isConst, util::Any(ptr, length, type, elements)));
}

bool EnvironmentDescription::getConst(const std::string &name) {
    for (auto &i : properties) {
        if (i.first == name) {
            return i.second.isConst;
        }
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getConst().",
        name.c_str());
}

const std::unordered_map<std::string, EnvironmentDescription::PropData> EnvironmentDescription::getPropertiesMap() const {
    return properties;
}
const std::unordered_map<std::string, EnvironmentDescription::MacroPropData> EnvironmentDescription::getMacroPropertiesMap() const {
    return macro_properties;
}

}  // namespace flamegpu
