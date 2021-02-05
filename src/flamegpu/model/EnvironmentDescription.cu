#include "flamegpu/model/EnvironmentDescription.h"

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
            if (v.second.isConst != _v->second.isConst
                || v.second.data.elements != _v->second.data.elements
                || v.second.data.type != _v->second.data.type)
                return false;
            if (v.second.data.ptr == _v->second.data.ptr &&
                v.second.data.length == _v->second.data.length)
                continue;
            if (v.second.data.length == _v->second.data.length) {
                for (size_t i = 0; i < v.second.data.length; ++i)
                    if (reinterpret_cast<char*>(v.second.data.ptr)[i] != reinterpret_cast<char*>(_v->second.data.ptr)[i])
                        return false;
                continue;
            }
            return false;
        }
        return true;
    }
    return false;
}
bool EnvironmentDescription::operator!=(const EnvironmentDescription& rhs) const {
    return !(*this == rhs);
}

void EnvironmentDescription::newProperty(const std::string &name, const char *ptr, const size_t &length, const bool &isConst, const EnvironmentManager::size_type &elements, const std::type_index &type) {
    properties.emplace(name, PropData(isConst,  Any(ptr, length, type, elements)));
}

bool EnvironmentDescription::getConst(const std::string &name) {
    for (auto &i : properties) {
        if (i.first == name) {
            return i.second.isConst;
        }
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getConst().",
        name.c_str());
}

const std::unordered_map<std::string, EnvironmentDescription::PropData> EnvironmentDescription::getPropertiesMap() const {
    return properties;
}
