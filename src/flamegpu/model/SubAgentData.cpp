#include "flamegpu/model/SubAgentData.h"

#include "flamegpu/model/SubAgentDescription.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/ModelData.h"


bool SubAgentData::operator==(const SubAgentData &rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    // Compare members
    if (variables == rhs.variables
        && states ==  rhs.states) {
        // Compare variables map
        return true;
    }
    return false;
}
bool SubAgentData::operator!=(const SubAgentData &rhs) const {
    return !(*this == rhs);
}
SubAgentData::SubAgentData(const std::shared_ptr<const ModelData> &model, const std::shared_ptr<SubModelData> &_parent, const SubAgentData &other)
    : subAgent(_parent->submodel->agents.at(other.subAgent.lock()->name))  // These two can only fail if the submodeldesc is cloned before the submodel
    , masterAgent(model->agents.at(other.masterAgent.lock()->name))
    , parent(_parent)
    , description(model ? new SubAgentDescription(model, this) : nullptr) {
    variables.insert(other.variables.begin(), other.variables.end());
    states.insert(other.states.begin(), other.states.end());
}
SubAgentData::SubAgentData(const std::shared_ptr<const ModelData> &model, const std::shared_ptr<SubModelData> &_parent, const std::shared_ptr<AgentData> &_subAgent, const std::shared_ptr<AgentData> &_masterAgent)
    : subAgent(_subAgent)
    , masterAgent(_masterAgent)
    , parent(_parent)
    , description(new SubAgentDescription(model, this)) { }
