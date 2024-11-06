#include "flamegpu/model/SubEnvironmentData.h"

#include <utility>
#include <memory>

#include "flamegpu/model/SubEnvironmentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubModelData.h"

namespace flamegpu {

SubEnvironmentData::SubEnvironmentData(std::shared_ptr<const ModelData> _model, const std::shared_ptr<SubModelData> &_parent, const SubEnvironmentData &other)
    : model(std::move(_model))
    , subEnvironment(_parent->submodel->environment)  // These two can only fail if the submodeldesc is cloned before the submodel
    , masterEnvironment(_model->environment)
    , parent(_parent) {
    properties.insert(other.properties.begin(), other.properties.end());
    macro_properties.insert(other.macro_properties.begin(), other.macro_properties.end());
    directed_graphs.insert(other.directed_graphs.begin(), other.directed_graphs.end());
}
SubEnvironmentData::SubEnvironmentData(
    std::shared_ptr<const ModelData> _model,
    const std::shared_ptr<SubModelData> &_parent,
    const std::shared_ptr<EnvironmentData> &_subEnvironment)
    : model(std::move(_model))
    , subEnvironment(_parent->submodel->environment)
    , masterEnvironment(_model->environment)
    , parent(_parent) { }

bool SubEnvironmentData::operator==(const SubEnvironmentData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    // Compare members
    if (properties == rhs.properties
        && macro_properties == rhs.macro_properties
        && directed_graphs == rhs.directed_graphs
        // && model.lock() == rhs.model.lock()  // Don't check weak pointers
        // && masterEnvironment.lock() == rhs.masterEnvironment.lock()  // Skipping any equality here feels unsafe
        ) {
        return true;
    }
    return false;
}
bool SubEnvironmentData::operator!=(const SubEnvironmentData& rhs) const {
    return !(*this == rhs);
}

}  // namespace flamegpu
