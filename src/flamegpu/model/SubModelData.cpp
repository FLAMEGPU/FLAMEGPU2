#include "flamegpu/model/SubModelData.h"

#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubAgentData.h"

bool SubModelData::operator==(const SubModelData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    // Compare members
    if (subagents.size() == rhs.subagents.size()
        && (submodel == rhs.submodel || *submodel == *rhs.submodel)) {
        // Compare subagents map
        for (auto &v : subagents) {
            auto _v = rhs.subagents.find(v.first);
            // Find failed, key mismatch
            if (_v == rhs.subagents.end())
                return false;
            // Val mismatch
            if (*v.second != *_v->second)
                return false;
        }
    }
    return false;
}
bool SubModelData::operator!=(const SubModelData& rhs) const {
    return !(*this == rhs);
}
SubModelData::SubModelData(const std::shared_ptr<ModelData> &model, const SubModelData &other)
    : submodel(other.submodel->clone())
    , name(other.name)
    , description(model ? new SubModelDescription(model, this) : nullptr) {
    // Note, this does not init subagents!
    // Note, this does not init subenvironment!
}
SubModelData::SubModelData(const std::shared_ptr<ModelData> &model, const std::string &submodel_name, const std::shared_ptr<ModelData> &_submodel)
    : submodel(_submodel)
    , name(submodel_name)
    , description(new SubModelDescription(model, this)) { }
