#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubModelData.h"

LayerDescription::LayerDescription(const std::shared_ptr<const ModelData> &_model, LayerData *const data)
    : model(_model)
    , layer(data) { }

bool LayerDescription::operator==(const LayerDescription& rhs) const {
    return *this->layer == *rhs.layer;  // Compare content is functionally the same
}
bool LayerDescription::operator!=(const LayerDescription& rhs) const {
    return !(*this == rhs);
}

void LayerDescription::addAgentFunction(const AgentFunctionDescription &afd) {
    if (afd.model.lock() == model.lock()) {
        auto m = model.lock();
        // Find the same afd in the model hierarchy
        for (auto &agt : m->agents) {
            for (auto &fn : agt.second->functions) {
                if (fn.second->description.get() == &afd) {
                    addAgentFunction(agt.first, fn.first);
                    return;
                }
            }
        }
    }
    THROW DifferentModel("Attempted to add agent function description which is from a different model, "
        "in LayerDescription::addAgentFunction().");
}
void LayerDescription::addAgentFunction(const std::string &agentName, const std::string &functionName) {
    if (layer->sub_model) {
        THROW InvalidLayerMember("A layer containing agent functions and/or host functions, may not also contain a submodel, "
        "in LayerDescription::addSubModel()\n");
    }
    // Locate the matching agent function in the model hierarchy
    auto mdl = model.lock();
    if (!mdl) {
        THROW ExpiredWeakPtr();
    }
    auto a = mdl->agents.find(agentName);
    if (a != mdl->agents.end()) {
        auto f = a->second->functions.find(functionName);
        if (f != a->second->functions.end()) {
            // Check it's not a duplicate agent fn
            if (layer->agent_functions.find(f->second) != layer->agent_functions.end()) {
                THROW InvalidAgentFunc("Attempted to add agent function '%s' owned by agent '%s' to same layer twice, "
                    "in LayerDescription::addAgentFunction().",
                    functionName.c_str(), agentName.c_str());
            }
            // Check that layer does not already contain function with same agent + states
            for (const auto &b : layer->agent_functions) {
                if (auto parent = b->parent.lock()) {
                    // If agent matches
                    if (parent->name == a->second->name) {
                        // If they share a state
                        if (b->initial_state == f->second->initial_state ||
                            b->initial_state == f->second->end_state ||
                            b->end_state == f->second->initial_state ||
                            b->end_state == f->second->end_state) {
                            THROW InvalidAgentFunc("Agent function '%s' owned by agent '%s' cannot be added to this layer as agent function '%s' "
                                "within the layer shares an input or output state, this is not permitted, "
                                "in LayerDescription::addAgentFunction().",
                                a->second->name.c_str(), agentName.c_str(), b->name.c_str());
                        }
                    }
                }
            }
            layer->agent_functions.insert(f->second);
            return;
        }
    }
    THROW InvalidAgentFunc("Agent function '%s' owned by agent '%s' was not found, "
        "in LayerDescription::addAgentFunction()\n",
        functionName.c_str(), agentName.c_str());
}
/**
 * Template magic means that the implicit cast doesn't occur as normal
 */
void LayerDescription::addAgentFunction(const char *an, const char *fn) {
    addAgentFunction(std::string(an), std::string(fn));
}
void LayerDescription::addHostFunction(FLAMEGPU_HOST_FUNCTION_POINTER func_p) {
    if (layer->sub_model) {
        THROW InvalidLayerMember("A layer containing agent functions and/or host functions, may not also contain a submodel, "
        "in LayerDescription::addSubModel()\n");
    }
    if (!layer->host_functions.insert(func_p).second) {
        THROW InvalidHostFunc("HostFunction has already been added to LayerDescription,"
            "in LayerDescription::addHostFunction().");
    }
}
void LayerDescription::addSubModel(const std::string &name) {
    if (!layer->host_functions.empty() || !layer->agent_functions.empty() || !layer->host_functions_callbacks.empty()) {
        THROW InvalidLayerMember("A layer containing agent functions and/or host functions, may not also contain a submodel, "
        "in LayerDescription::addSubModel()\n");
    }
    if (layer->sub_model) {
        THROW InvalidSubModel("Layer has already been assigned a submodel, "
            "in LayerDescription::addSubModel()\n");
    }
    // Find the correct submodel shared ptr
    auto mdl = model.lock();
    if (!mdl) {
        THROW ExpiredWeakPtr();
    }
    for (auto &sm : mdl->submodels) {
        if (sm.first == name) {
            layer->sub_model = sm.second;
            return;
        }
    }
    THROW InvalidSubModel("SubModel '%s' does not belong to Model '%s', "
        "in LayerDescription::addSubModel()\n",
        name.c_str(), mdl->name.c_str());
}
void LayerDescription::addSubModel(const SubModelDescription &submodel) {
    auto mdl = model.lock();
    if (!mdl) {
        THROW ExpiredWeakPtr();
    }
    if (submodel.model.lock() == mdl) {
        // Find the correct submodel shared ptr
        for (auto &sm : mdl->submodels) {
            if (sm.second.get() == submodel.data) {
                addSubModel(sm.first);
                return;
            }
        }
    }
    THROW InvalidSubModel("SubModel '%s' does not belong to Model '%s', "
        "in LayerDescription::addSubModel()\n",
        submodel.data->submodel->name.c_str(), mdl->name.c_str());
}

std::string LayerDescription::getName() const {
    return layer->name;
}

ModelData::size_type LayerDescription::getIndex() const {
    return layer->index;
}


ModelData::size_type LayerDescription::getAgentFunctionsCount() const {
    // Safe down-cast
    return static_cast<ModelData::size_type>(layer->agent_functions.size());
}
ModelData::size_type LayerDescription::getHostFunctionsCount() const {
    // Safe down-cast
    return static_cast<ModelData::size_type>(layer->host_functions.size());
}

const AgentFunctionDescription &LayerDescription::getAgentFunction(unsigned int index) const {
    if (index < layer->agent_functions.size()) {
        auto it = layer->agent_functions.begin();
        for (unsigned int i = 0; i < index; ++i)
            ++it;
        return *((*it)->description);
    }
    THROW OutOfBoundsException("Index %d is out of bounds (only %d items exist) "
        "in LayerDescription.getAgentFunction().",
    index, layer->agent_functions.size());
}
FLAMEGPU_HOST_FUNCTION_POINTER LayerDescription::getHostFunction(unsigned int index) const {
    if (index < layer->host_functions.size()) {
        auto it = layer->host_functions.begin();
        for (unsigned int i = 0; i < index; ++i)
            ++it;
        return *it;
    }
    THROW OutOfBoundsException("Index %d is out of bounds (only %d items exist) "
        "in LayerDescription.getHostFunction().",
        index, layer->host_functions.size());
}
