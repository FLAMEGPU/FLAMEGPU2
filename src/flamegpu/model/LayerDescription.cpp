#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

LayerDescription::LayerDescription(ModelData *const _model, LayerData *const data)
    : model(_model)
    , layer(data) { }

bool LayerDescription::operator==(const LayerDescription& rhs) const {
    return *this->layer == *rhs.layer;  // Compare content is functionally the same
}
bool LayerDescription::operator!=(const LayerDescription& rhs) const {
    return !(*this == rhs);
}

void LayerDescription::addAgentFunction(const AgentFunctionDescription &afd) {
    if (afd.model == layer->description->model) {
        addAgentFunction(afd.getName());
        return;
    }
    THROW DifferentModel("Attempted to add agent function description which is from a different model, "
        "in LayerDescription::addAgentFunction().");
}
void LayerDescription::addAgentFunction(const std::string &name) {
    // Locate the matching agent function in the model hierarchy
    for (auto a : model->agents) {
        for (auto f : a.second->functions) {
            if (f.second->name == name) {
                // Check that layer does not already contain function with same agent + states
                for (const auto &b : layer->agent_functions) {
                    if (auto parent = b->parent.lock()) {
                        // If agent matches
                        if (parent->name == a.second->name) {
                            // If they share a state
                            if (b->initial_state == f.second->initial_state ||
                                b->initial_state == f.second->end_state ||
                                b->end_state == f.second->initial_state ||
                                b->end_state == f.second->end_state) {
                                THROW InvalidAgentFunc("Agent functions '%s' cannot be added to this layer as agent function '%s' "
                                    "within the layer shares an input or output state, this is not permitted, "
                                    "in LayerDescription::addAgentFunction().",
                                    f.second->name.c_str(), b->name.c_str());
                            }
                        }
                    }
                }
                if (layer->agent_functions.insert(f.second).second)
                    return;
                THROW InvalidAgentFunc("Attempted to add agent function '%s' to same layer twice, "
                    "in LayerDescription::addAgentFunction().",
                    name.c_str());
            }
        }
    }
    THROW InvalidAgentFunc("Agent function '%s' was not found, "
        "in LayerDescription::addAgentFunction()\n",
        name.c_str());
}
/**
 * Template magic means that the implicit cast doesn't occur as normal
 */
void LayerDescription::addAgentFunction(const char *af) {
    addAgentFunction(std::string(af));
}
void LayerDescription::addHostFunction(FLAMEGPU_HOST_FUNCTION_POINTER func_p) {
    if (!layer->host_functions.insert(func_p).second) {
        THROW InvalidHostFunc("HostFunction has already been added to LayerDescription,"
            "in LayerDescription::addHostFunction().");
    }
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
