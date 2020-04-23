#include <iostream>

#include "flamegpu/model/ModelData.h"

#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/AgentFunctionData.h"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/model/SubModelData.h"

const char *ModelData::DEFAULT_STATE = "default";

/**
 * Constructors
 */
ModelData::ModelData(const std::string &model_name)
    : environment(new EnvironmentDescription())
    , name(model_name) { }

ModelData::~ModelData() { }

std::shared_ptr<ModelData> ModelData::clone() const {
    // Awkwardly cant use shared from this inside constructor, so use raw pts instead
    return std::shared_ptr<ModelData>(new ModelData(*this));
}

ModelData::ModelData(const ModelData &other)
    : initFunctions(other.initFunctions)
    , stepFunctions(other.stepFunctions)
    , exitFunctions(other.exitFunctions)
    , exitConditions(other.exitConditions)
    , environment(new EnvironmentDescription(*other.environment))
    , name(other.name) {
    // Manually copy construct maps of shared ptr
    for (const auto &m : other.messages) {
        messages.emplace(m.first, std::shared_ptr<MsgBruteForce::Data>(m.second->clone(this)));  // Need to convert this to shared_ptr, how to force shared copy construct?
    }
    // Copy all agents first
    for (const auto &a : other.agents) {
        auto b = std::shared_ptr<AgentData>(new AgentData(this, *a.second));
        agents.emplace(a.first, b);
    }
    // Copy agent functions per agent, after all agents have been implemented.
    for (const auto &a : other.agents) {
        auto b = agents.find(a.first)->second;
        // Manually copy construct maps of shared ptr
        for (const auto &f : a.second->functions) {
            b->functions.emplace(f.first, std::shared_ptr<AgentFunctionData>(new AgentFunctionData(this, b, *f.second)));
        }
    }
    // Copy submodels
    for (const auto &a : other.submodels) {
        auto b = std::shared_ptr<SubModelData>(new SubModelData(this, *a.second));
        // Manually copy construct maps of shared ptr
        for (const auto &f : a.second->subagents) {
            b->subagents.emplace(f.first, std::shared_ptr<SubAgentData>(new SubAgentData(this, b, *f.second)));
        }
        submodels.emplace(a.first, b);
    }

    for (const auto &m : other.layers) {
        layers.push_back(std::shared_ptr<LayerData>(new LayerData(this, *m)));
    }
}

bool ModelData::operator==(const ModelData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (name == rhs.name
        && agents.size() == rhs.agents.size()
        && messages.size() == rhs.messages.size()
        && submodels.size() == rhs.submodels.size()
        && layers.size() == rhs.layers.size()
        && initFunctions.size() == rhs.initFunctions.size()
        && stepFunctions.size() == rhs.stepFunctions.size()
        && exitFunctions.size() == rhs.exitFunctions.size()
        && exitConditions.size() == rhs.exitConditions.size()
        && *environment == *rhs.environment) {
            {  // Compare agents (map)
                for (auto &v : agents) {
                    auto _v = rhs.agents.find(v.first);
                    if (_v == rhs.agents.end())
                        return false;
                    if (*v.second != *_v->second)
                        return false;
                }
            }
            {  // Compare messages (map)
                for (auto &v : messages) {
                    auto _v = rhs.messages.find(v.first);
                    if (_v == rhs.messages.end())
                        return false;
                    if (*v.second != *_v->second)
                        return false;
                }
            }
            {  // Compare submodels (map)
                for (auto &v : submodels) {
                    auto _v = rhs.submodels.find(v.first);
                    if (_v == rhs.submodels.end())
                        return false;
                    if (*v.second != *_v->second)
                        return false;
                }
            }
            {  // Compare layers (ordered list)
                auto it1 = layers.begin();
                auto it2 = rhs.layers.begin();
                while (it1 != layers.end() && it2 != rhs.layers.end()) {
                    if (*it1 != *it2)
                        return false;
                    ++it1;
                    ++it2;
                }
            }
            {  // Init fns (set)
                if (initFunctions != rhs.initFunctions)
                    return false;
            }
            {  // Step fns (set)
                if (stepFunctions != rhs.stepFunctions)
                    return false;
            }
            {  // Exit fns (set)
                if (exitFunctions != rhs.exitFunctions)
                    return false;
            }
            {  // Exit cdns (set)
                if (exitConditions != rhs.exitConditions)
                    return false;
            }
            return true;
    }
    return false;
}

bool ModelData::operator!=(const ModelData& rhs) const {
    return !operator==(rhs);
}
