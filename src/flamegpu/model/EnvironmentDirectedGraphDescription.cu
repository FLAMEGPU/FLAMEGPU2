#include "flamegpu/model/EnvironmentDirectedGraphDescription.cuh"
namespace flamegpu {

CEnvironmentDirectedGraphDescription::CEnvironmentDirectedGraphDescription(std::shared_ptr<EnvironmentDirectedGraphData> data)
    : graph(std::move(data)) { }
CEnvironmentDirectedGraphDescription::CEnvironmentDirectedGraphDescription(std::shared_ptr<const EnvironmentDirectedGraphData> data)
    : graph(std::const_pointer_cast<EnvironmentDirectedGraphData>(data)) { }

bool CEnvironmentDirectedGraphDescription::operator==(const CEnvironmentDirectedGraphDescription& rhs) const {
    return *this->graph == *rhs.graph;  // Compare content is functionally the same
}
bool CEnvironmentDirectedGraphDescription::operator!=(const CEnvironmentDirectedGraphDescription& rhs) const {
    return !(*this == rhs);
}
/**
 * Const Accessors
 */
std::string CEnvironmentDirectedGraphDescription::getName() const {
    return graph->name;
}
const std::type_index& CEnvironmentDirectedGraphDescription::getVertexPropertyType(const std::string& property_name) const {
    const auto f = graph->vertexProperties.find(property_name);
    if (f != graph->vertexProperties.end()) {
        return f->second.type;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') does not contain vertex property '%s', "
        "in EnvironmentDirectedGraphDescription::getVertexPropertyType().",
        graph->name.c_str(), property_name.c_str());
}
const std::type_index& CEnvironmentDirectedGraphDescription::getEdgePropertyType(const std::string& property_name) const {
    const auto f = graph->edgeProperties.find(property_name);
    if (f != graph->edgeProperties.end()) {
        return f->second.type;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') does not contain edge property '%s', "
        "in EnvironmentDirectedGraphDescription::getEdgePropertyType().",
        graph->name.c_str(), property_name.c_str());
}

size_t CEnvironmentDirectedGraphDescription::getVertexPropertySize(const std::string& property_name) const {
    const auto f = graph->vertexProperties.find(property_name);
    if (f != graph->vertexProperties.end()) {
        return f->second.type_size;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') does not contain vertex property '%s', "
        "in EnvironmentDirectedGraphDescription::getVertexPropertySize().",
        graph->name.c_str(), property_name.c_str());
}
size_t CEnvironmentDirectedGraphDescription::getEdgePropertySize(const std::string& property_name) const {
    const auto f = graph->edgeProperties.find(property_name);
    if (f != graph->edgeProperties.end()) {
        return f->second.type_size;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') does not contain edge property '%s', "
        "in EnvironmentDirectedGraphDescription::getEdgePropertySize().",
        graph->name.c_str(), property_name.c_str());
}

flamegpu::size_type CEnvironmentDirectedGraphDescription::getVertexPropertyLength(const std::string& property_name) const {
    const auto f = graph->vertexProperties.find(property_name);
    if (f != graph->vertexProperties.end()) {
        return f->second.elements;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') does not contain vertex property '%s', "
        "in EnvironmentDirectedGraphDescription::getVertexPropertyLength().",
        graph->name.c_str(), property_name.c_str());
}
flamegpu::size_type CEnvironmentDirectedGraphDescription::getEdgePropertyLength(const std::string& property_name) const {
    const auto f = graph->edgeProperties.find(property_name);
    if (f != graph->edgeProperties.end()) {
        return f->second.elements;
    }
    THROW exception::InvalidGraphProperty("Graph ('%s') does not contain edge property '%s', "
        "in EnvironmentDirectedGraphDescription::getEdgePropertyLength().",
        graph->name.c_str(), property_name.c_str());
}

flamegpu::size_type CEnvironmentDirectedGraphDescription::geVertexPropertiesCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<flamegpu::size_type>(graph->vertexProperties.size());
}
flamegpu::size_type CEnvironmentDirectedGraphDescription::getEdgePropertiesCount() const {
    // Downcast, will never have more than UINT_MAX VARS
    return static_cast<flamegpu::size_type>(graph->edgeProperties.size());
}

bool CEnvironmentDirectedGraphDescription::hasVertexProperty(const std::string& property_name) const {
    return graph->vertexProperties.find(property_name) != graph->vertexProperties.end();
}
bool CEnvironmentDirectedGraphDescription::hasEdgeProperty(const std::string& property_name) const {
    return graph->edgeProperties.find(property_name) != graph->edgeProperties.end();
}

/**
 * Constructors
 */
EnvironmentDirectedGraphDescription::EnvironmentDirectedGraphDescription(std::shared_ptr<EnvironmentDirectedGraphData> data)
    : CEnvironmentDirectedGraphDescription(std::move(data)) { }


}  // namespace flamegpu
