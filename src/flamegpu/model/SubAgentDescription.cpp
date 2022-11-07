#include "flamegpu/model/SubAgentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/model/AgentData.h"

namespace flamegpu {

CSubAgentDescription::CSubAgentDescription(std::shared_ptr<SubAgentData> data)
    : subagent(std::move(data)) { }
CSubAgentDescription::CSubAgentDescription(std::shared_ptr<const SubAgentData> data)
    : subagent(std::move(std::const_pointer_cast<SubAgentData>(data))) { }

bool CSubAgentDescription::operator==(const CSubAgentDescription& rhs) const {
    return *this->subagent == *rhs.subagent;  // Compare content is functionally the same
}
bool CSubAgentDescription::operator!=(const CSubAgentDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
std::string CSubAgentDescription::getStateMapping(const std::string& sub_state_name) const {
    const auto v = subagent->states.find(sub_state_name);
    if (v != subagent->states.end())
        return v->second;
    THROW exception::InvalidAgentState("Sub agent state '%s', either does not exist or has not been mapped yet, "
        "in SubAgentDescription::getStateMapping()\n", sub_state_name.c_str());
}
std::string CSubAgentDescription::getVariableMapping(const std::string& sub_variable_name) const {
    const auto v = subagent->variables.find(sub_variable_name);
    if (v != subagent->variables.end())
        return v->second;
    THROW exception::InvalidAgentVar("Sub agent var '%s', either does not exist or has not been mapped yet, "
        "in SubAgentDescription::getVariableMapping()\n", sub_variable_name.c_str());
}

/**
 * Constructors
 */
SubAgentDescription::SubAgentDescription(std::shared_ptr<SubAgentData> data)
    : CSubAgentDescription(std::move(data)) { }

bool SubAgentDescription::operator==(const CSubAgentDescription& rhs) const {
    return rhs == *this;  // Forward to superclass's equality
}
bool SubAgentDescription::operator!=(const CSubAgentDescription& rhs) const {
    return !(*this == rhs);
}

void SubAgentDescription::mapState(const std::string &sub_state_name, const std::string &master_state_name) {
    // Sub state exists
    auto subAgent = subagent->subAgent.lock();
    if (!subAgent) {
        THROW exception::InvalidParent("SubAgent pointer has expired, "
            "in SubAgentDescription::mapState()\n");
    }
    const auto subVar = subAgent->states.find(sub_state_name);
    if (subVar == subAgent->states.end()) {
        THROW exception::InvalidAgentState("Sub Agent '%s' does not contain state '%s', "
            "in SubAgentDescription::mapState()\n", subAgent->name.c_str(), sub_state_name.c_str());
    }
    // Master states exists
    auto masterAgent = subagent->masterAgent.lock();
    if (!masterAgent) {
        THROW exception::InvalidParent("MasterAgent pointer has expired, "
            "in SubAgentDescription::mapState()\n");
    }
    const auto masterVar = masterAgent->states.find(master_state_name);
    if (masterVar == masterAgent->states.end()) {
        THROW exception::InvalidAgentState("Master Agent '%s' does not contain state '%s', "
            "in SubAgentDescription::mapState()\n", masterAgent->name.c_str(), master_state_name.c_str());
    }
    // Sub state has not been bound yet
    if (subagent->states.find(sub_state_name) != subagent->states.end()) {
        const auto parent = subagent->parent.lock();
        THROW exception::InvalidAgentState("SubModel '%s's Agent '%s' state '%s' has already been mapped, "
            "in SubAgentDescription::mapState()\n", parent ? parent->submodel->name.c_str() : "?", subAgent->name.c_str(), sub_state_name.c_str());
    }
    // Master state has already been bound
    for (auto &v : subagent->states) {
        if (v.second == master_state_name) {
            THROW exception::InvalidAgentState("Master Agent '%s' state '%s' has already been mapped, "
                "in SubAgentDescription::mapState()\n", masterAgent->name.c_str(), master_state_name.c_str());
        }
    }
    // States match, create mapping
    subagent->states.emplace(sub_state_name, master_state_name);
}
void SubAgentDescription::mapVariable(const std::string &sub_variable_name, const std::string &master_variable_name) {
    // Sub variable exists
    auto subAgent = subagent->subAgent.lock();
    if (!subAgent) {
        THROW exception::InvalidParent("SubAgent pointer has expired, "
            "in SubAgentDescription::mapVariable()\n");
    }
    const auto subVar = subAgent->variables.find(sub_variable_name);
    if (subVar == subAgent->variables.end()) {
        THROW exception::InvalidAgentVar("Sub Agent '%s' does not contain Variable '%s', "
            "in SubAgentDescription::mapVariable()\n", subAgent->name.c_str(), sub_variable_name.c_str());
    }
    // Master variable exists
    auto masterAgent = subagent->masterAgent.lock();
    if (!masterAgent) {
        THROW exception::InvalidParent("MasterAgent pointer has expired, "
            "in SubAgentDescription::mapVariable()\n");
    }
    const auto masterVar = masterAgent->variables.find(master_variable_name);
    if (masterVar == masterAgent->variables.end()) {
        THROW exception::InvalidAgentVar("Master Agent '%s' does not contain Variable '%s', "
            "in SubAgentDescription::mapVariable()\n", masterAgent->name.c_str(), master_variable_name.c_str());
    }
    // Sub variable has not been bound yet
    if (subagent->variables.find(sub_variable_name) != subagent->variables.end()) {
        const auto parent = subagent->parent.lock();
        THROW exception::InvalidAgentVar("SubModel '%s's Agent '%s' variable '%s' has already been mapped, "
            "in SubAgentDescription::mapVariable()\n", parent ? parent->submodel->name.c_str() : "?", subAgent->name.c_str(), sub_variable_name.c_str());
    }
    // Master variable has already been bound
    for (auto &v : subagent->variables) {
        if (v.second == master_variable_name) {
            THROW exception::InvalidAgentVar("Master Agent '%s' variable '%s' has already been mapped, "
                "in SubAgentDescription::mapVariable()\n", masterAgent->name.c_str(), master_variable_name.c_str());
        }
    }
    // Check type and length (is it an array var)
    if (subVar->second.type != masterVar->second.type || subVar->second.elements != masterVar->second.elements) {
        THROW exception::InvalidAgentVar("Variable types ('%s', '%s') and/or lengths (%u, %u) do not match, "
            "in SubAgentDescription::mapVariable()\n", subVar->second.type.name(), masterVar->second.type.name(), subVar->second.elements, masterVar->second.elements);
    }
    // Variables match, create mapping
    subagent->variables.emplace(sub_variable_name, master_variable_name);
}

}  // namespace flamegpu
