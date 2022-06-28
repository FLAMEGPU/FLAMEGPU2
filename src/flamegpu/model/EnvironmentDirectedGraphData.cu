#include "flamegpu/model/EnvironmentDirectedGraphData.cuh"

#include "flamegpu/model/EnvironmentDirectedGraphDescription.cuh"
#include "flamegpu/defines.h"

namespace flamegpu {

EnvironmentDirectedGraphData::EnvironmentDirectedGraphData(const std::shared_ptr<const EnvironmentData> &_parent, const std::string& agent_name)
    : model(_parent->model)
    , name(agent_name) {
    vertexProperties.emplace(ID_VARIABLE_NAME, Variable(std::array<id_t, 1>{ ID_NOT_SET }));
    edgeProperties.emplace(GRAPH_SOURCE_DEST_VARIABLE_NAME, Variable(std::array<id_t, 2>{ ID_NOT_SET, ID_NOT_SET }));
}
EnvironmentDirectedGraphData::EnvironmentDirectedGraphData(const std::shared_ptr<const ModelData> &_model, const EnvironmentDirectedGraphData& other)
    : model(_model)
    , vertexProperties(other.vertexProperties)
    , edgeProperties(other.edgeProperties)
    , name(other.name) { }
bool EnvironmentDirectedGraphData::operator==(const EnvironmentDirectedGraphData& rhs) const {
    if (name == rhs.name
        && vertexProperties.size() == rhs.vertexProperties.size()
        && edgeProperties.size() == rhs.edgeProperties.size()) {
        {  // Compare vertex properties
            for (auto& v : vertexProperties) {
                auto _v = rhs.vertexProperties.find(v.first);
                if (_v == rhs.vertexProperties.end())
                    return false;
                if (v.second.type_size != _v->second.type_size || v.second.type != _v->second.type || v.second.elements != _v->second.elements)
                    return false;
            }
        }
        {  // Compare edge properties
            for (auto& v : edgeProperties) {
                auto _v = rhs.edgeProperties.find(v.first);
                if (_v == rhs.edgeProperties.end())
                    return false;
                if (v.second.type_size != _v->second.type_size || v.second.type != _v->second.type || v.second.elements != _v->second.elements)
                    return false;
            }
        }
        return true;
    }
    return false;
}
bool EnvironmentDirectedGraphData::operator!=(const EnvironmentDirectedGraphData& rhs) const {
    return !operator==(rhs);
}

}  // namespace flamegpu
