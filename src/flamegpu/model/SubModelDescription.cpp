#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/model/SubEnvironmentDescription.h"

namespace flamegpu {

CSubModelDescription::CSubModelDescription(std::shared_ptr<SubModelData> data)
    : submodel(std::move(data)) { }
CSubModelDescription::CSubModelDescription(std::shared_ptr<const SubModelData> data)
    : submodel(std::move(std::const_pointer_cast<SubModelData>(data))) { }

bool CSubModelDescription::operator==(const CSubModelDescription& rhs) const {
    return *this->submodel == *rhs.submodel;  // Compare content is functionally the same
}
bool CSubModelDescription::operator!=(const CSubModelDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
unsigned int CSubModelDescription::getMaxSteps() const {
    return submodel->max_steps;
}
const std::string CSubModelDescription::getName() const {
    return submodel->name;
}

/**
 * Constructors
 */
SubModelDescription::SubModelDescription(std::shared_ptr<SubModelData> data)
    : CSubModelDescription(std::move(data)) { }

bool SubModelDescription::operator==(const CSubModelDescription& rhs) const {
    return rhs == *this;  // Forward to superclass's equality
}
bool SubModelDescription::operator!=(const CSubModelDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Accessors
 */
SubAgentDescription &SubModelDescription::bindAgent(const std::string &sub_agent_name, const std::string &master_agent_name, bool auto_map_vars, bool auto_map_states) {
    // Sub agent exists
    const auto subagent = submodel->submodel->agents.find(sub_agent_name);
    if (subagent == submodel->submodel->agents.end()) {
        THROW exception::InvalidSubAgentName("SubModel '%s' does not contain Agent '%s', "
            "in SubModelDescription::bindAgent()\n", submodel->submodel->name.c_str(), sub_agent_name.c_str());
    }
    auto mdl = submodel->model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    // Master agent exists
    const auto masteragent = mdl->agents.find(master_agent_name);
    if (masteragent == mdl->agents.end()) {
        THROW exception::InvalidAgentName("Master Model '%s' does not contain Agent '%s', "
            "in SubModelDescription::bindAgent()\n", mdl->name.c_str(), master_agent_name.c_str());
    }
    // Sub agent has not been bound yet
    {
        const auto subagent_bind = submodel->subagents.find(sub_agent_name);
        if (subagent_bind != submodel->subagents.end()) {
            auto master_agent_ptr = subagent_bind->second->masterAgent.lock();
            THROW exception::InvalidSubAgentName("SubModel '%s's Agent '%s' has already been bound to Master agent '%s', "
                "in SubModelDescription::bindAgent()\n", submodel->submodel->name.c_str(), sub_agent_name.c_str(), master_agent_ptr ? master_agent_ptr->name.c_str() : "?");
        }
    }
    // Master agent has not been bound yet
    for (auto &a : submodel->subagents) {
        const auto master_agent_ptr = a.second->masterAgent.lock();
        if (master_agent_ptr && master_agent_ptr->name == master_agent_name) {
            THROW exception::InvalidAgentName("Master Agent '%s' has already been bound to Sub agent '%s', "
                "in SubModelDescription::bindAgent()\n", master_agent_name.c_str(), a.first.c_str());
        }
    }
    // Create SubAgent
    auto rtn = std::shared_ptr<SubAgentData>(new SubAgentData(mdl, submodel->shared_from_this(), subagent->second, masteragent->second));
    submodel->subagents.emplace(sub_agent_name, rtn);
    // If auto_map, map any matching vars
    // Otherwise map all internal variables that begin _ (e.g. _id)
    for (auto& sub_var : subagent->second->variables) {
        if (auto_map_vars || (!sub_var.first.empty() && sub_var.first[0] == '_')) {
            auto master_var = masteragent->second->variables.find(sub_var.first);
            // If there exists variable with same name in both agents
            if (master_var != masteragent->second->variables.end()) {
                // Check type and length (is it an array var)
                if (sub_var.second.type == master_var->second.type
                    && sub_var.second.elements == master_var->second.elements) {
                    // Variables match, create mapping
                    rtn->variables.emplace(sub_var.first, master_var->first);  // Doesn't actually matter, both strings are equal
                }
            }
        }
    }
    // If auto_map, map any matching states
    if (auto_map_states) {
        for (auto &sub_var : subagent->second->states) {
            auto master_var = masteragent->second->states.find(sub_var);
            // If there exists states with same name in both agents
            if (master_var != masteragent->second->states.end()) {
                // States match, create mapping
                rtn->states.emplace(sub_var, *master_var);  // Doesn't actually matter, both strings are equal
            }
        }
    }
    // return SubAgentDescription
    return *rtn->description;
}

SubAgentDescription &SubModelDescription::SubAgent(const std::string &sub_agent_name) {
    const auto rtn = submodel->subagents.find(sub_agent_name);
    if (rtn != submodel->subagents.end())
        return *rtn->second->description;
    THROW exception::InvalidSubAgentName("SubAgent ('%s') either does not exist, or has not been bound yet, "
        "in SubModelDescription::SubAgent().",
        sub_agent_name.c_str());
}
const SubAgentDescription &SubModelDescription::getSubAgent(const std::string &sub_agent_name) const {
    const auto rtn = submodel->subagents.find(sub_agent_name);
    if (rtn != submodel->subagents.end())
        return *rtn->second->description;
    THROW exception::InvalidSubAgentName("SubAgent ('%s')  either does not exist, or has not been bound yet, "
        "in SubModelDescription::getSubAgent().",
        sub_agent_name.c_str());
}


SubEnvironmentDescription &SubModelDescription::SubEnvironment(bool auto_map) {
    if (!submodel->subenvironment) {
        auto mdl = submodel->model.lock();
        if (!mdl) {
            THROW exception::ExpiredWeakPtr();
        }
        submodel->subenvironment = std::shared_ptr<SubEnvironmentData>(new SubEnvironmentData(mdl, submodel->shared_from_this(), submodel->submodel->environment));
    }
    if (auto_map) {
        submodel->subenvironment->description->autoMapProperties();
        submodel->subenvironment->description->autoMapMacroProperties();
    }
    return *submodel->subenvironment->description;
}
const SubEnvironmentDescription &SubModelDescription::getSubEnvironment(bool auto_map) const {
    if (!submodel->subenvironment) {
        auto mdl = submodel->model.lock();
        if (!mdl) {
            THROW exception::ExpiredWeakPtr();
        }
        submodel->subenvironment = std::shared_ptr<SubEnvironmentData>(new SubEnvironmentData(mdl, submodel->shared_from_this(), submodel->submodel->environment));
    }
    if (auto_map) {
        submodel->subenvironment->description->autoMapProperties();
        submodel->subenvironment->description->autoMapMacroProperties();
    }
    return *submodel->subenvironment->description;
}

void SubModelDescription::setMaxSteps(const unsigned int max_steps) {
    submodel->max_steps = max_steps;
}

}  // namespace flamegpu
