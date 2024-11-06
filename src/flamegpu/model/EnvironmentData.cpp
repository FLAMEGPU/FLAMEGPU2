#include "flamegpu/model/EnvironmentData.h"

#include <memory>

#include "flamegpu/model/EnvironmentDirectedGraphData.cuh"

namespace flamegpu {

EnvironmentData::EnvironmentData(std::shared_ptr<const ModelData> _model)
    : model(_model) {
    // Add CUDASimulation specific environment members
    // We do this here, to not break comparing different model description hierarchies before/after CUDASimulation creation
    properties.emplace("_stepCount", PropData(false, detail::Any(0u)));
}

EnvironmentData::EnvironmentData(std::shared_ptr<const ModelData> _model, const EnvironmentData& other)
    : model(_model)
    , properties(other.properties)
    , macro_properties(other.macro_properties) {
    directed_graphs.clear();
    for (const auto& g : other.directed_graphs) {
        auto t = std::shared_ptr<EnvironmentDirectedGraphData>(new EnvironmentDirectedGraphData(_model, *g.second));
        directed_graphs.emplace(g.first, t);
    }
}

bool EnvironmentData::operator==(const EnvironmentData& rhs) const {
    if (this == &rhs)  // They point to same object
        return true;
    //  if(model.lock() != rhs.model.lock()) return false;  // Don't check weak pointers
    if (properties.size() == rhs.properties.size()) {
        for (auto& v : properties) {
            auto _v = rhs.properties.find(v.first);
            if (_v == rhs.properties.end())
                return false;
            if (v.second != _v->second)
                return false;
        }
        return true;
    }
    if (macro_properties.size() == rhs.macro_properties.size()) {
        for (auto& v : macro_properties) {
            auto _v = rhs.macro_properties.find(v.first);
            if (_v == rhs.macro_properties.end())
                return false;
            if (v.second != _v->second)
                return false;
        }
        return true;
    }
    if (macro_properties.size() == rhs.macro_properties.size()) {
        for (auto& v : macro_properties) {
            auto _v = rhs.macro_properties.find(v.first);
            if (_v == rhs.macro_properties.end())
                return false;
            if (v.second != _v->second)
                return false;
        }
        return true;
    }
    if (directed_graphs.size() == rhs.directed_graphs.size()) {
        for (auto& v : directed_graphs) {
            auto _v = rhs.directed_graphs.find(v.first);
            if (_v == rhs.directed_graphs.end())
                return false;
            if (v.second != _v->second)
                return false;
        }
        return true;
    }
    return false;
}
bool EnvironmentData::operator!=(const EnvironmentData& rhs) const {
    return !(*this == rhs);
}

}  // namespace flamegpu
