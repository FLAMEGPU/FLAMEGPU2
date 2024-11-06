#include <utility>
#include <memory>
#include <string>

#include "flamegpu/model/EnvironmentDescription.h"
#include "flamegpu/model/EnvironmentDirectedGraphDescription.cuh"

namespace flamegpu {

CEnvironmentDescription::CEnvironmentDescription(std::shared_ptr<EnvironmentData> data)
    : environment(std::move(data)) { }
CEnvironmentDescription::CEnvironmentDescription(std::shared_ptr<const EnvironmentData> data)
    : environment(std::const_pointer_cast<EnvironmentData>(data)) { }

bool CEnvironmentDescription::operator==(const CEnvironmentDescription& rhs) const {
    return *this->environment == *rhs.environment;  // Compare content is functionally the same
}
bool CEnvironmentDescription::operator!=(const CEnvironmentDescription& rhs) const {
    return !(*this == rhs);
}

/**
 * Const Accessors
 */
bool CEnvironmentDescription::getConst(const std::string& name) const {
    for (auto& i : environment->properties) {
        if (i.first == name) {
            return i.second.isConst;
        }
    }
    THROW exception::InvalidEnvProperty("Environmental property with name '%s' does not exist, "
        "in EnvironmentDescription::getConst().",
        name.c_str());
}
CEnvironmentDirectedGraphDescription CEnvironmentDescription::getDirectedGraph(const std::string& graph_name) const {
    auto r = this->environment->directed_graphs.find(graph_name);
    if (r == this->environment->directed_graphs.end()) {
        THROW exception::InvalidGraphName("Directed graph with name '%s' already exists, "
            "in EnvironmentDescription::getDirectedGraph().",
            graph_name.c_str());
    }
    return CEnvironmentDirectedGraphDescription(r->second);
}

/**
 * Constructors
 */
EnvironmentDescription::EnvironmentDescription(std::shared_ptr<EnvironmentData> data)
    : CEnvironmentDescription(std::move(data)) { }

/**
 * Accessors
 */
void EnvironmentDescription::newProperty(const std::string &name, const char *ptr, size_t length, bool isConst, flamegpu::size_type elements, const std::type_index &type) {
    environment->properties.emplace(name, EnvironmentData::PropData(isConst, detail::Any(ptr, length, type, elements)));
}
EnvironmentDirectedGraphDescription EnvironmentDescription::newDirectedGraph(const std::string& graph_name) {
    if (this->environment->directed_graphs.find(graph_name) == this->environment->directed_graphs.end()) {
        auto t = std::shared_ptr<EnvironmentDirectedGraphData>(new EnvironmentDirectedGraphData(this->environment, graph_name));
        this->environment->directed_graphs.emplace(graph_name, t);
        return EnvironmentDirectedGraphDescription(t);
    }
    THROW exception::InvalidGraphName("Directed graph with name '%s' already exists, "
        "in EnvironmentDescription::newDirectedGraph().",
        graph_name.c_str());
}
EnvironmentDirectedGraphDescription EnvironmentDescription::getDirectedGraph(const std::string& graph_name) {
    auto r = this->environment->directed_graphs.find(graph_name);
    if (r == this->environment->directed_graphs.end()) {
        THROW exception::InvalidGraphName("Directed graph with name '%s' already exists, "
            "in EnvironmentDescription::getDirectedGraph().",
            graph_name.c_str());
    }
    return EnvironmentDirectedGraphDescription(r->second);
}

}  // namespace flamegpu
