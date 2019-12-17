#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

LayerDescription::LayerDescription(ModelData *const _model, LayerData *const data)
    : model(_model)
    , layer(data) { }
// Copy Construct
LayerDescription::LayerDescription(const LayerDescription &other_layer)
    : model(other_layer.model)
    , layer(other_layer.layer) {
    // TODO
}
// Move Construct
LayerDescription::LayerDescription(LayerDescription &&other_layer) noexcept
    : model(move(other_layer.model))
    , layer(other_layer.layer) {
    // TODO
}

bool LayerDescription::operator==(const LayerDescription& rhs) const {
    return *this->layer == *rhs.layer;  // Compare content is functionally the same
}
bool LayerDescription::operator!=(const LayerDescription& rhs) const {
    return !(*this == rhs);
}

void LayerDescription::addAgentFunction(const AgentFunctionDescription &afd) {
    addAgentFunction(afd.getName());
}
void LayerDescription::addAgentFunction(const std::string &name) {
    for (auto a : model->agents) {
        for (auto f : a.second->functions) {
            if (f.second->name == name) {
                if (layer->agent_functions.insert(f.second).second)
                    return;
                THROW InvalidAgentFunc("Attempted to add agent function '%s' to same layer twice, "
                    "in LayerDescription::addAgentFunction()\n",
                    name.c_str());
            }
        }
    }
    THROW InvalidAgentFunc("Agent function '%s' was not found, "
        "in AgentFunctionDescription::addAgentFunction()\n",
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
        "in LayerDescription.getAgentFunction()\n",
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
        "in LayerDescription.getHostFunction()\n",
        index, layer->host_functions.size());
}
