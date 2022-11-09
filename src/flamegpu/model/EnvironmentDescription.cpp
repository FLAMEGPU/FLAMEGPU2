#include "flamegpu/model/EnvironmentDescription.h"

namespace flamegpu {

CEnvironmentDescription::CEnvironmentDescription(std::shared_ptr<EnvironmentData> data)
    : environment(std::move(data)) { }
CEnvironmentDescription::CEnvironmentDescription(std::shared_ptr<const EnvironmentData> data)
    : environment(std::const_pointer_cast<EnvironmentData>(data)) { }

bool CEnvironmentDescription::operator==(const CEnvironmentDescription& rhs) const {
    return *this->environment == *rhs.environment;  // Compare content is functionally the same
}
bool CEnvironmentDescription::operator!=(const CEnvironmentDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
bool CEnvironmentDescription::getConst(const std::string& name) const {
    for (auto& i : environment->properties) {
        if (i.first == name) {
            return i.second.isConst;
        }
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getConst().",
        name.c_str());
}

/**
 * Constructors
 */
EnvironmentDescription::EnvironmentDescription(std::shared_ptr<EnvironmentData> data)
    : CEnvironmentDescription(std::move(data)) { }

bool EnvironmentDescription::operator==(const CEnvironmentDescription& rhs) const {
    return rhs == *this;  // Forward to superclass's equality
}
bool EnvironmentDescription::operator!=(const CEnvironmentDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Accessors
 */
void EnvironmentDescription::newProperty(const std::string &name, const char *ptr, size_t length, bool isConst, flamegpu::size_type elements, const std::type_index &type) {
    environment->properties.emplace(name, EnvironmentData::PropData(isConst, util::Any(ptr, length, type, elements)));
}

}  // namespace flamegpu
