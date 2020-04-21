#include "flamegpu/model/SubModelData.h"

#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubAgentDescription.h"

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
std::shared_ptr<const SubModelData> SubModelData::clone() const {
    std::shared_ptr<SubModelData> b = std::shared_ptr<SubModelData>(new SubModelData(nullptr, *this));
    // Manually copy construct maps of shared ptr
    for (const auto f : subagents) {
        b->subagents.emplace(f.first, std::shared_ptr<SubAgentData>(new SubAgentData(nullptr, b, *f.second)));
    }
    return b;
}
SubModelData::SubModelData(const ModelData * model, const SubModelData &other)
    : submodel(other.submodel->clone())
    , description(model ? new SubModelDescription(model, this) : nullptr) { }
SubModelData::SubModelData(const ModelData * model, std::shared_ptr<ModelData> _submodel)
    : submodel(std::move(_submodel))
    , description(new SubModelDescription(model, this)) { }


bool SubAgentData::operator==(const SubAgentData &rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    // Compare members
    if (variables.size() == rhs.variables.size()
        && states.size() == rhs.states.size()) {
        // Compare variables map
        for (auto &v : variables) {
            auto _v = rhs.variables.find(v.first);
            // Find failed, key mismatch
            if (_v == rhs.variables.end())
                return false;
            // Val mismatch
            if (v.second != _v->second)
                return false;
        }
        // Compare states map
        for (auto &v : states) {
            auto _v = rhs.states.find(v.first);
            // Find failed, key mismatch
            if (_v == rhs.states.end())
                return false;
            // Val mismatch
            if (v.second != _v->second)
                return false;
        }
    }
    return false;
}
bool SubAgentData::operator!=(const SubAgentData &rhs) const {
    return !(*this == rhs);
}
std::shared_ptr<const SubAgentData> SubAgentData::clone() const {
    std::shared_ptr<SubAgentData> b = std::shared_ptr<SubAgentData>(new SubAgentData(nullptr, nullptr, *this));
    // Manually copy construct maps of shared ptr
    // (None present to copy)
    return b;
}
SubAgentData::SubAgentData(const ModelData *model, const std::shared_ptr<SubModelData> &_parent, const SubAgentData &other)
    : subAgent(_parent->submodel->agents.at(other.subAgent.lock()->name))  // These two can only fail if the submodeldesc is cloned before the submodel
    , masterAgent(model->agents.at(other.masterAgent.lock()->name))
    , parent(_parent)
    , description(model ? new SubAgentDescription(model, this) : nullptr) {
    variables.insert(other.variables.begin(), other.variables.end());
    states.insert(other.states.begin(), other.states.end());
}

SubAgentData::SubAgentData(const ModelData *model, const std::shared_ptr<SubModelData> &_parent, const std::shared_ptr<AgentData> &_subAgent, const std::shared_ptr<AgentData> &_masterAgent)
    : subAgent(_subAgent)
    , masterAgent(_masterAgent)
    , parent(_parent)
    , description(new SubAgentDescription(model, this)) { }
