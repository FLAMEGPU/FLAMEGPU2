#include "flamegpu/model/LayerData.h"

#include "flamegpu/model/LayerDescription.h"
#include "flamegpu/model/SubModelData.h"

LayerData::LayerData(const std::shared_ptr<const ModelData> &model, const std::string &layer_name, const ModelData::size_type &layer_index)
    : description(new LayerDescription(model, this))
    , name(layer_name)
    , index(layer_index) { }

LayerData::LayerData(const std::shared_ptr<const ModelData> &model, const LayerData &other)
    : host_functions(other.host_functions)
    , host_functions_callbacks(other.host_functions_callbacks)
    , description(model ? new LayerDescription(model, this) : nullptr)
    , name(other.name)
    , index(other.index) {
    // Manually perform lookup copies
    for (auto &_f : other.agent_functions) {
        for (auto &a : model->agents) {
            for (auto &f : a.second->functions) {
                if (f.second->func == _f->func) {
                    if (f.second->name == _f->name) {
                        agent_functions.emplace(f.second);
                        goto next_agent_fn;
                    }
                }
            }
        }
    next_agent_fn : {}
    }
    if (other.sub_model) {
        for (auto &a : model->submodels) {
            if (other.sub_model->name == a.second->name) {
                sub_model = a.second;
            }
        }
    }
}

bool LayerData::operator==(const LayerData &rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    if (name == rhs.name
    && index == rhs.index
    && agent_functions.size() == rhs.agent_functions.size()
    && host_functions.size() == rhs.host_functions.size()
    && host_functions_callbacks.size() == rhs.host_functions_callbacks.size()
    && agent_functions == rhs.agent_functions
    && host_functions == rhs.host_functions) {
        return true;
    }
    return false;
}

bool LayerData::operator!=(const LayerData &rhs) const {
    return !operator==(rhs);
}
