#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/AgentFunctionDescription.h"

LayerDescription::LayerDescription(std::weak_ptr<ModelData> _model, LayerData *const data)
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
// Copy Assign
LayerDescription& LayerDescription::operator=(const LayerDescription &other_layer) {
    // TODO
    return *this;
}
// Move Assign
LayerDescription& LayerDescription::operator=(LayerDescription &&other_layer) noexcept {
    // TODO
    return *this;
}


void LayerDescription::addAgentFunction(const AgentFunctionDescription &afd) {
    addAgentFunction(afd.getName());
}
void LayerDescription::addAgentFunction(const std::string &name) {
    if (auto m = model.lock()) {
        for (auto a : m->agents) {
            for (auto f : a.second->functions) {
                if (f.second->name == name) {
                    if(layer->agent_functions.insert(f.second).second)
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
    THROW InvalidParent("Agent parent has expired, "
        "in AgentFunctionDescription::addAgentFunction()\n");
}
void LayerDescription::addHostFunction(FLAMEGPU_HOST_FUNCTION_POINTER func_p) {

    layer->host_functions.insert(func_p);
}

ModelData::size_type LayerDescription::getIndex() const {
    return layer->index;
}