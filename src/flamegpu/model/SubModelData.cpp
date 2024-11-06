#include "flamegpu/model/SubModelData.h"

#include <utility>
#include <string>
#include <memory>

#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubAgentData.h"

namespace flamegpu {
SubModelData::SubModelData(std::shared_ptr<ModelData> _model, const SubModelData& other)
    : model(std::move(_model))
    , submodel(other.submodel->clone())
    , max_steps(other.max_steps)
    , name(other.name) {
    // Note, this does not init subagents!
    // Note, this does not init subenvironment!
}
SubModelData::SubModelData(std::shared_ptr<ModelData> _model, const std::string& submodel_name, const std::shared_ptr<ModelData>& _submodel)
    : model(std::move(_model))
    , submodel(_submodel)
    , max_steps(0)
    , name(submodel_name) { }

bool SubModelData::operator==(const SubModelData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    // Compare members
    if (name == rhs.name
        //  && model.lock() == rhs.model.lock()  // Don't check weak pointers
        && subagents.size() == rhs.subagents.size()
        && (submodel == rhs.submodel || *submodel == *rhs.submodel)
        && max_steps == rhs.max_steps) {
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
        return true;
    }
    return false;
}
bool SubModelData::operator!=(const SubModelData& rhs) const {
    return !(*this == rhs);
}

}  // namespace flamegpu
