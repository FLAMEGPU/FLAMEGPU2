#include "flamegpu/model/SubEnvironmentData.h"

#include "flamegpu/model/SubEnvironmentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubModelData.h"

namespace flamegpu {

bool SubEnvironmentData::operator==(const SubEnvironmentData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    // Compare members
    if (properties == rhs.properties) {
        return true;
    }
    return false;
}
bool SubEnvironmentData::operator!=(const SubEnvironmentData& rhs) const {
    return !(*this == rhs);
}

SubEnvironmentData::SubEnvironmentData(const std::shared_ptr<const ModelData> &model, const std::shared_ptr<SubModelData> &_parent, const SubEnvironmentData &other)
    : subEnvironment(_parent->submodel->environment)  // These two can only fail if the submodeldesc is cloned before the submodel
    , masterEnvironment(model->environment)
    , parent(_parent)
    , description(model ? new SubEnvironmentDescription(model, this) : nullptr) {
    properties.insert(other.properties.begin(), other.properties.end());
}
SubEnvironmentData::SubEnvironmentData(
    const std::shared_ptr<const ModelData> &model,
    const std::shared_ptr<SubModelData> &_parent,
    const std::shared_ptr<EnvironmentDescription> &_subEnvironment)
    : subEnvironment(_parent->submodel->environment)
    , masterEnvironment(model->environment)
    , parent(_parent)
    , description(new SubEnvironmentDescription(model, this)) { }

}  // namespace flamegpu
