#include "flamegpu/model/SubAgentDescription.h"
#include "flamegpu/model/ModelData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/AgentData.h"

SubAgentDescription::SubAgentDescription(const ModelData *const _model, SubAgentData *const _data)
    : model(_model)
    , data(_data) { }

void SubAgentDescription::mapVariable(const std::string &sub_variable_name, const std::string &master_variable_name) {
    // Sub variable exists
    auto subAgent = data->subAgent.lock();
    if (!subAgent) {
        THROW InvalidParent("SubAgent pointer has expired, "
            "in SubAgentDescription::mapVariable()\n");
    }
    const auto subVar = subAgent->variables.find(sub_variable_name);
    if (subVar == subAgent->variables.end()) {
        THROW InvalidAgentVar("Sub Agent '%s' does not contain Variable '%s', "
            "in SubAgentDescription::mapVariable()\n", subAgent->name.c_str(), sub_variable_name.c_str());
    }
    // Master variable exists
    auto masterAgent = data->masterAgent.lock();
    if (!masterAgent) {
        THROW InvalidParent("MasterAgent pointer has expired, "
            "in SubAgentDescription::mapVariable()\n");
    }
    const auto masterVar = masterAgent->variables.find(master_variable_name);
    if (masterVar == masterAgent->variables.end()) {
        THROW InvalidAgentVar("Master Agent '%s' does not contain Variable '%s', "
            "in SubAgentDescription::mapVariable()\n", masterAgent->name.c_str(), master_variable_name.c_str());
    }
    // Sub variable has not been bound yet
    if (data->variables.find(sub_variable_name) != data->variables.end()) {
        const auto parent = data->parent.lock();
        THROW InvalidAgentVar("SubModel '%s's Agent '%s' variable '%s' has already been mapped, "
            "in SubAgentDescription::mapVariable()\n", parent ? parent->submodel->name.c_str() : "?", subAgent->name.c_str(), sub_variable_name.c_str());
    }
    // Master variable has already been bound
    for (auto &v : data->variables) {
        if (v.second == master_variable_name) {
            THROW InvalidAgentVar("Master Agent '%s' variable '%s' has already been mapped, "
                "in SubAgentDescription::mapVariable()\n", masterAgent->name.c_str(), master_variable_name.c_str());
        }
    }
    // Check type and length (is it an array var)
    if (subVar->second.type != masterVar->second.type
        || subVar->second.elements != masterVar->second.elements) {
        THROW InvalidAgentVar("Variable types ('%s', '%s') and/or lengths (%u, %u) do not match, "
            "in SubAgentDescription::mapVariable()\n", subVar->second.type.name(), masterVar->second.type.name(), subVar->second.elements, masterVar->second.elements);
    }
    // Variables match, create mapping
    data->variables.emplace(subVar->first, masterVar->first);
}
std::string SubAgentDescription::getVariableMapping(const std::string &sub_variable_name) {
    const auto v = data->variables.find(sub_variable_name);
    if (v != data->variables.end())
        return v->second;
    THROW InvalidAgentVar("Sub agent var '%s', either does not exist or has not been mapped yet, "
        "in SubAgentDescription::getVariableMapping()\n", sub_variable_name.c_str());
}
