#include <iostream>
#include <algorithm>

#include "flamegpu/model/ModelData.h"

#include "flamegpu/model/AgentData.h"
#include "flamegpu/model/AgentDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/AgentFunctionData.cuh"
#include "flamegpu/model/LayerData.h"
#include "flamegpu/model/SubModelData.h"
#include "flamegpu/model/SubAgentData.h"
#include "flamegpu/model/SubEnvironmentData.h"
#include "flamegpu/model/DependencyGraph.h"
#include "flamegpu/runtime/HostFunctionCallback.h"

namespace flamegpu {

const char *ModelData::DEFAULT_STATE = "default";

/**
 * Constructors
 */
ModelData::ModelData(const std::string &model_name)
    : environment(new EnvironmentDescription())
    , name(model_name)
    , dependencyGraph(new DependencyGraph(this)) { }

ModelData::~ModelData() { }

std::shared_ptr<ModelData> ModelData::clone() const {
    // Awkwardly cant use shared from this inside constructor, so use raw pts instead
    auto rtn = std::shared_ptr<ModelData>(new ModelData(*this));

    // Manually copy construct maps of shared ptr
    for (const auto &m : this->messages) {
        rtn->messages.emplace(m.first, std::shared_ptr<MessageBruteForce::Data>(m.second->clone(rtn)));  // Need to convert this to shared_ptr, how to force shared copy construct?
    }
    // Copy all agents first
    for (const auto &a : this->agents) {
        auto b = std::shared_ptr<AgentData>(new AgentData(rtn, *a.second));
        rtn->agents.emplace(a.first, b);
    }
    // Copy agent functions per agent, after all agents have been implemented.
    for (const auto &a : this->agents) {
        auto b = rtn->agents.find(a.first)->second;
        // Manually copy construct maps of shared ptr
        for (const auto &f : a.second->functions) {
            b->functions.emplace(f.first, std::shared_ptr<AgentFunctionData>(new AgentFunctionData(rtn, b, *f.second)));
        }
    }
    // Copy submodels
    for (const auto &a : this->submodels) {
        auto b = std::shared_ptr<SubModelData>(new SubModelData(rtn, *a.second));
        // Manually copy construct maps of shared ptr
        for (const auto &f : a.second->subagents) {
            b->subagents.emplace(f.first, std::shared_ptr<SubAgentData>(new SubAgentData(rtn, b, *f.second)));
        }
        // Manually copy construct environment
        b->subenvironment = std::unique_ptr<SubEnvironmentData>(new SubEnvironmentData(rtn, b, *a.second->subenvironment));
        rtn->submodels.emplace(a.first, b);
    }

    for (const auto &m : this->layers) {
        rtn->layers.push_back(std::shared_ptr<LayerData>(new LayerData(rtn, *m)));
    }
    return rtn;
}

ModelData::ModelData(const ModelData &other)
    : initFunctions(other.initFunctions)
    , initFunctionCallbacks(other.initFunctionCallbacks)
    , stepFunctions(other.stepFunctions)
    , stepFunctionCallbacks(other.stepFunctionCallbacks)
    , exitFunctions(other.exitFunctions)
    , exitFunctionCallbacks(other.exitFunctionCallbacks)
    , exitConditions(other.exitConditions)
    , exitConditionCallbacks(other.exitConditionCallbacks)
    , environment(new EnvironmentDescription(*other.environment))
    , name(other.name)
    , dependencyGraph(new DependencyGraph(*other.dependencyGraph)) {
    // Must be called from clone() so that items are all init
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
        && initFunctionCallbacks.size() == rhs.initFunctionCallbacks.size()
        && stepFunctionCallbacks.size() == rhs.stepFunctionCallbacks.size()
        && exitFunctionCallbacks.size() == rhs.exitFunctionCallbacks.size()
        && exitConditionCallbacks.size() == rhs.exitConditionCallbacks.size()
        && exitConditions.size() == rhs.exitConditions.size()
        && *environment == *rhs.environment
        && *dependencyGraph == *rhs.dependencyGraph) {
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
                    if (*(*it1) != *(*it2))
                        return false;
                    ++it1;
                    ++it2;
                }
            }
            {  // Init fns (set)
                if (initFunctions != rhs.initFunctions)
                    return false;
                if (initFunctionCallbacks != rhs.initFunctionCallbacks)
                    return false;
            }
            {  // Step fns (set)
                if (stepFunctions != rhs.stepFunctions)
                    return false;
                if (stepFunctionCallbacks != rhs.stepFunctionCallbacks)
                    return false;
            }
            {  // Exit fns (set)
                if (exitFunctions != rhs.exitFunctions)
                    return false;
                if (exitFunctionCallbacks != rhs.exitFunctionCallbacks)
                    return false;
            }
            {  // Exit cdns (set)
                if (exitConditions != rhs.exitConditions)
                    return false;
                if (exitConditionCallbacks != rhs.exitConditionCallbacks)
                    return false;
            }
            return true;
    }
    return false;
}

bool ModelData::operator!=(const ModelData& rhs) const {
    return !operator==(rhs);
}
bool ModelData::hasSubModelRecursive(const std::shared_ptr<const ModelData> &submodel_data) const {
    for (auto &m : submodels) {
        if (m.second->submodel.get() == submodel_data.get())
            return true;
        if (m.second->submodel->hasSubModelRecursive(submodel_data))
            return true;
    }
    return false;
}

ModelData::size_type ModelData::getMaxLayerWidth() const {
    unsigned int maxWidth = 0u;
    for (auto &layer : layers) {
        maxWidth = (std::max)(maxWidth, static_cast<ModelData::size_type>(layer->agent_functions.size()));
    }
    return maxWidth;
}

}  // namespace flamegpu
