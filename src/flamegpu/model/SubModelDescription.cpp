#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/SubModelData.h"

SubModelDescription::SubModelDescription(const ModelData *const _model, SubModelData *const _data)
    : model(_model)
    , data(_data) { }

SubAgentDescription &SubModelDescription::bindAgent(const std::string &sub_agent_name, const std::string &master_agent_name, bool auto_map_vars, bool auto_map_states) {
    // Sub agent exists
    const auto subagent = data->submodel->agents.find(sub_agent_name);
    if (subagent == data->submodel->agents.end()) {
        THROW InvalidSubAgentName("SubModel '%s' does not contain Agent '%s', "
            "in SubModelDescription::bindAgent()\n", data->submodel->name.c_str(), sub_agent_name.c_str());
    }
    // Master agent exists
    const auto masteragent = model->agents.find(master_agent_name);
    if (masteragent == model->agents.end()) {
        THROW InvalidAgentName("Master Model '%s' does not contain Agent '%s', "
            "in SubModelDescription::bindAgent()\n", model->name.c_str(), master_agent_name.c_str());
    }
    // Sub agent has not been bound yet
    {
        const auto subagent_bind = data->subagents.find(sub_agent_name);
        if (subagent_bind != data->subagents.end()) {
            auto master_agent_ptr = subagent_bind->second->masterAgent.lock();
            THROW InvalidSubAgentName("SubModel '%s's Agent '%s' has already been bound to Master agent '%s', "
                "in SubModelDescription::bindAgent()\n", data->submodel->name.c_str(), sub_agent_name.c_str(), master_agent_ptr ? master_agent_ptr->name.c_str() : "?");
        }
    }
    // Master agent has not been bound yet
    for (auto &a : data->subagents) {
        const auto master_agent_ptr = a.second->masterAgent.lock();
        if (master_agent_ptr && master_agent_ptr->name == master_agent_name) {
            THROW InvalidAgentName("Master Agent '%s' has already been bound to Sub agent '%s', "
                "in SubModelDescription::bindAgent()\n", master_agent_name.c_str(), a.first.c_str());
        }
    }
    // Create SubAgent
    auto rtn = std::shared_ptr<SubAgentData>(new SubAgentData(model, data->shared_from_this(), subagent->second, masteragent->second));
    data->subagents.emplace(sub_agent_name, rtn);
    // If auto_map, map any matching vars
    if (auto_map_vars) {
        for (auto &sub_var : subagent->second->variables) {
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
    const auto rtn = data->subagents.find(sub_agent_name);
    if (rtn != data->subagents.end())
        return *rtn->second->description;
    THROW InvalidSubAgentName("SubAgent ('%s') either does not exist, or has not been bound yet, "
        "in SubModelDescription::SubAgent().",
        sub_agent_name.c_str());
}
const SubAgentDescription &SubModelDescription::getSubAgent(const std::string &sub_agent_name) const {
    const auto rtn = data->subagents.find(sub_agent_name);
    if (rtn != data->subagents.end())
        return *rtn->second->description;
    THROW InvalidSubAgentName("SubAgent ('%s')  either does not exist, or has not been bound yet, "
        "in SubModelDescription::getSubAgent().",
        sub_agent_name.c_str());
}
