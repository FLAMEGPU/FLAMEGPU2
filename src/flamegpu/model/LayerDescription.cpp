#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"
#include "flamegpu/model/SubModelDescription.h"
#include "flamegpu/model/SubModelData.h"

namespace flamegpu {

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
    THROW exception::DifferentModel("Attempted to add agent function description which is from a different model, "
        "in LayerDescription::addAgentFunction().");
}
void LayerDescription::addAgentFunction(const std::string &agentName, const std::string &functionName) {
    if (layer->sub_model) {
        THROW exception::InvalidLayerMember("A layer containing agent functions and/or host functions, may not also contain a submodel, "
        "in LayerDescription::addAgentFunction()\n");
    }
    if (layer->host_functions.size() || layer->host_functions_callbacks.size()) {
        THROW exception::InvalidLayerMember("A layer containing host functions, may not also contain agent functions, "
            "in LayerDescription::addAgentFunction()\n");
    }
    if (!layer->host_functions.empty() || !layer->host_functions_callbacks.empty()) {
        THROW exception::InvalidLayerMember("A layer containing a host function may not also contain an agent function"
            "in LayerDescription::addAgentFunction()\n");
    }
    // Locate the matching agent function in the model hierarchy
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    auto a = mdl->agents.find(agentName);
    if (a != mdl->agents.end()) {
        auto f = a->second->functions.find(functionName);
        if (f != a->second->functions.end()) {
            // Check it's not a duplicate agent fn
            if (layer->agent_functions.find(f->second) != layer->agent_functions.end()) {
                THROW exception::InvalidAgentFunc("Attempted to add agent function '%s' owned by agent '%s' to same layer twice, "
                    "in LayerDescription::addAgentFunction().",
                    functionName.c_str(), agentName.c_str());
            }
            auto a_agent_out = f->second->agent_output.lock();
            auto a_msg_out = f->second->message_output.lock();
            auto a_msg_in = f->second->message_input.lock();
            for (const auto &b : layer->agent_functions) {
                if (auto parent = b->parent.lock()) {
                    // Check that layer does not already contain function with same agent + states
                    // If agent matches
                    if (parent->name == a->second->name) {
                        // If they share a state
                        if (b->initial_state == f->second->initial_state ||
                            b->initial_state == f->second->end_state ||
                            b->end_state == f->second->initial_state ||
                            b->end_state == f->second->end_state) {
                            THROW exception::InvalidAgentFunc("Agent function '%s' owned by agent '%s' cannot be added to this layer as agent function '%s' "
                                "within the layer shares an input or output state, this is not permitted, "
                                "in LayerDescription::addAgentFunction().",
                                a->second->name.c_str(), agentName.c_str(), b->name.c_str());
                        }
                    }
                    // Check that the layer does not already contain function for the agent + state being output to
                    if (a_agent_out) {
                        // If agent matches
                        if (parent->name == a_agent_out->name) {
                            // If state matches
                            if (b->initial_state == f->second->agent_output_state) {
                                THROW exception::InvalidLayerMember("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                                    "within the layer requires the same agent state as an input, as this agent function births, "
                                    "in LayerDescription::addAgentFunction().",
                                    f->second->name.c_str(), b->name.c_str());
                            }
                        }
                    }
                    // Also check the inverse
                    auto b_agent_out = b->agent_output.lock();
                    if (b_agent_out) {
                        // If agent matches
                        if (a->second->name == b_agent_out->name) {
                            // If state matches
                            if (f->second->initial_state == b->agent_output_state) {
                                THROW exception::InvalidLayerMember("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                                    "within the layer agent births to the same agent state as this agent function requires as an input, "
                                    "in LayerDescription::addAgentFunction().",
                                    f->second->name.c_str(), b->name.c_str());
                            }
                        }
                    }
                }
                // Check the layer does not already contain function which outputs to same message list
                auto b_msg_out = b->message_output.lock();
                auto b_msg_in = b->message_input.lock();
                if ((a_msg_out && b_msg_out && a_msg_out == b_msg_out) ||
                    (a_msg_out && b_msg_in && a_msg_out == b_msg_in) ||
                    (a_msg_in && b_msg_out && a_msg_in == b_msg_out)) {  // Pointer comparison should be fine here
                    THROW exception::InvalidLayerMember("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                        "within the layer also inputs or outputs to the same messagelist, this is not permitted, "
                        "in LayerDescription::addAgentFunction().",
                        f->second->name.c_str(), b->name.c_str());
                }
            }
            layer->agent_functions.insert(f->second);
            return;
        }
    }
    THROW exception::InvalidAgentFunc("Agent function '%s' owned by agent '%s' was not found, "
        "in LayerDescription::addAgentFunction()\n",
        functionName.c_str(), agentName.c_str());
}

void LayerDescription::addAgentFunction(const char * agentName, const char * functionName) {
    addAgentFunction(std::string(agentName), std::string(functionName));
}
void LayerDescription::addHostFunction(FLAMEGPU_HOST_FUNCTION_POINTER func_p) {
    if (layer->sub_model) {
        THROW exception::InvalidLayerMember("A layer containing a submodel may not also contain a host function, "
        "in LayerDescription::addHostFunction()\n");
    }
    if (!layer->host_functions.empty() || !layer->agent_functions.empty() || !layer->host_functions_callbacks.empty()) {
        THROW exception::InvalidLayerMember("A layer containing agent functions or a host function may not also contain a host function, "
        "in LayerDescription::addHostFunction()\n");
    }
    if (!layer->host_functions.insert(func_p).second) {
        THROW exception::InvalidHostFunc("HostFunction has already been added to LayerDescription,"
            "in LayerDescription::addHostFunction().");
    }
}
void LayerDescription::addSubModel(const std::string &name) {
    if (!layer->host_functions.empty() || !layer->agent_functions.empty() || !layer->host_functions_callbacks.empty()) {
        THROW exception::InvalidLayerMember("A layer containing agent functions and/or host functions, may not also contain a submodel, "
        "in LayerDescription::addSubModel()\n");
    }
    if (layer->sub_model) {
        THROW exception::InvalidSubModel("Layer has already been assigned a submodel, "
            "in LayerDescription::addSubModel()\n");
    }
    // Find the correct submodel shared ptr
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
    }
    for (auto &sm : mdl->submodels) {
        if (sm.first == name) {
            layer->sub_model = sm.second;
            return;
        }
    }
    THROW exception::InvalidSubModel("SubModel '%s' does not belong to Model '%s', "
        "in LayerDescription::addSubModel()\n",
        name.c_str(), mdl->name.c_str());
}
void LayerDescription::addSubModel(const SubModelDescription &submodel) {
    auto mdl = model.lock();
    if (!mdl) {
        THROW exception::ExpiredWeakPtr();
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
    THROW exception::InvalidSubModel("SubModel '%s' does not belong to Model '%s', "
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
    THROW exception::OutOfBoundsException("Index %d is out of bounds (only %d items exist) "
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
    THROW exception::OutOfBoundsException("Index %d is out of bounds (only %d items exist) "
        "in LayerDescription.getHostFunction().",
        index, layer->host_functions.size());
}

void LayerDescription::_addHostFunctionCallback(HostFunctionCallback* func_callback) {
    if (layer->sub_model) {
        THROW exception::InvalidLayerMember("A layer containing a submodel may not also contain a host function, "
        "in LayerDescription::addHostFunctionCallback()\n");
    }
    if (!layer->host_functions.empty() || !layer->agent_functions.empty() || !layer->host_functions_callbacks.empty()) {
        THROW exception::InvalidLayerMember("A layer containing agent functions or a host function may not also contain a host function, "
        "in LayerDescription::addHostFunctionCallback()\n");
    }
    if (!layer->host_functions_callbacks.insert(func_callback).second) {
            THROW exception::InvalidHostFunc("Attempted to add same host function callback twice,"
                "in LayerDescription::addHostFunctionCallback()");
        }
}

}  // namespace flamegpu
