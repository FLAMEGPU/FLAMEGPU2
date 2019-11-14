#include "flamegpu/model/EnvironmentDescription.h"

void EnvironmentDescription::add(const std::string &name, const char *ptr, const size_t &length, const bool &isConst, const EnvironmentManager::size_type &elements, const std::type_index &type) {
    properties.emplace(name, PropData(isConst,  elements, Any(ptr, length), type));
}

void EnvironmentDescription::setConst(const std::string &name, const bool &isConst) {
    for (auto &i : properties) {
        if (i.first == name) {
            i.second.isConst = isConst;
            return;
        }
    }
    THROW InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::setConst().",
        name.c_str());
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
