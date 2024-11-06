#include "flamegpu/visualiser/EnvironmentGraphVis.h"

#include <utility>
#include <string>
#include <memory>

#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"
#include "flamegpu/visualiser/LineVis.h"
#include "flamegpu/visualiser/config/LineConfig.h"

namespace flamegpu {
namespace visualiser {

EnvironmentGraphVisData::EnvironmentGraphVisData(std::shared_ptr <EnvironmentDirectedGraphData> _graphData, std::shared_ptr<LineConfig>_lines)
    : color(Stock::Colors::WHITE)
    , graphData(std::move(_graphData))
    , lines(std::move(_lines)) {
    const auto &x_it = graphData->vertexProperties.find("x");
    if (x_it != graphData->vertexProperties.end() &&
        x_it->second.type == std::type_index(typeid(float)) &&
        x_it->second.elements == 1) {
        x_varName = "x";
    }
    const auto &y_it = graphData->vertexProperties.find("y");
    if (y_it != graphData->vertexProperties.end() &&
        y_it->second.type == std::type_index(typeid(float)) &&
        y_it->second.elements == 1) {
        y_varName = "y";
    }
    const auto &z_it = graphData->vertexProperties.find("z");
    if (z_it != graphData->vertexProperties.end() &&
        z_it->second.type == std::type_index(typeid(float)) &&
        z_it->second.elements == 1) {
        z_varName = "z";
    }
}
void EnvironmentGraphVisData::constructGraph(const std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> &graph) {
    // Can't construct prior to initialisation
    if (!graph->getVertexCount() && !graph->getEdgeCount())
        return;
    // Retrieve buffer data
    const float *x = nullptr;
    const float *y = nullptr;
    const float *z = nullptr;
    int x_stride = 1;
    int y_stride = 1;
    int z_stride = 1;
    if (!x_varName.empty() && !y_varName.empty()) {
        size_type ONE = 1;
        x = graph->getVertexPropertyBuffer<float>(x_varName, ONE, nullptr);
        y = graph->getVertexPropertyBuffer<float>(y_varName, ONE, nullptr);
        if (!z_varName.empty())
            z = graph->getVertexPropertyBuffer<float>(z_varName, ONE, nullptr);
    } else if (!xy_varName.empty()) {
        size_type TWO = 1;
        const float* xy = graph->getVertexPropertyBuffer<float>(xy_varName, TWO, nullptr);
        x = xy;
        y = x+1;
        x_stride = 2;
        y_stride = 2;
    } else if (!xyz_varName.empty()) {
        size_type THREE = 1;
        const float* xyz = graph->getVertexPropertyBuffer<float>(xyz_varName, THREE, nullptr);
        x = xyz;
        y = x + 1;
        z = y + 1;
        x_stride = 3;
        y_stride = 3;
        z_stride = 3;
    } else {
        throw exception::InvalidOperation("Unable to construct graph visualisation, appropriate vertex variables not set, in EnvironmentGraphVisData::constructGraph()");
    }
    // Update line sketch
    auto ln = LineVis(lines, color.r, color.g, color.b, color.a);
    ln.clear();
    // Iterate edges
    size_type TWO = 2;
    const id_t *edges = graph->getEdgePropertyBuffer<id_t>(GRAPH_SOURCE_DEST_VARIABLE_NAME, TWO, nullptr);
    for (unsigned int i = 0; i < graph->getEdgeCount(); ++i) {
        // Sketch destination vertex
        const size_type dest_i = graph->getVertexIndex(edges[i * 2]);
        if (z) {
            ln.addVertex(x[dest_i * x_stride], y[dest_i * y_stride], z[dest_i * z_stride]);
        } else {
            ln.addVertex(x[dest_i * x_stride], y[dest_i * y_stride]);
        }
        // Sketch source vertex
        const size_type src_i = graph->getVertexIndex(edges[(i * 2) + 1]);
        if (z) {
            ln.addVertex(x[src_i * x_stride], y[src_i * y_stride], z[src_i * z_stride]);
        } else {
            ln.addVertex(x[src_i * x_stride], y[src_i * y_stride]);
        }
    }
}

EnvironmentGraphVis::EnvironmentGraphVis(std::shared_ptr<EnvironmentGraphVisData> _data)
    : data(std::move(_data)) { }

void EnvironmentGraphVis::setXVertexProperty(const std::string &var_name) {
    auto it = data->graphData->vertexProperties.find(var_name);
    if (it == data->graphData->vertexProperties.end()) {
        THROW exception::InvalidEnvProperty("Property '%s' was not found within graph '%s', "
            "in EnvironmentGraphVis::setXProperty()\n",
            var_name.c_str(), data->graphData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidEnvProperty("Visualisation position x property must be type float[1], graph '%s' property '%s' is type %s[%u], "
            "in EnvironmentGraphVis::setXProperty()\n",
            data->graphData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->xy_varName.clear();
    data->xyz_varName.clear();
    data->x_varName = var_name;
}
void EnvironmentGraphVis::setYVertexProperty(const std::string &var_name) {
    auto it = data->graphData->vertexProperties.find(var_name);
    if (it == data->graphData->vertexProperties.end()) {
        THROW exception::InvalidEnvProperty("Property '%s' was not found within graph '%s', "
            "in EnvironmentGraphVis::setYProperty()\n",
            var_name.c_str(), data->graphData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidEnvProperty("Visualisation position y property must be type float[1], graph '%s' property '%s' is type %s[%u], "
            "in EnvironmentGraphVis::setYProperty()\n",
            data->graphData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->xy_varName.clear();
    data->xyz_varName.clear();
    data->y_varName = var_name;
}
void EnvironmentGraphVis::setZVertexProperty(const std::string &var_name) {
    auto it = data->graphData->vertexProperties.find(var_name);
    if (it == data->graphData->vertexProperties.end()) {
        THROW exception::InvalidEnvProperty("Property '%s' was not found within graph '%s', "
            "in EnvironmentGraphVis::setZProperty()\n",
            var_name.c_str(), data->graphData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 1) {
        THROW exception::InvalidEnvProperty("Visualisation position z property must be type float[1], graph '%s' property '%s' is type %s[%u], "
            "in EnvironmentGraphVis::setZProperty()\n",
            data->graphData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->xy_varName.clear();
    data->xyz_varName.clear();
    data->z_varName = var_name;
}
void EnvironmentGraphVis::setXYVertexProperty(const std::string& var_name) {
    auto it = data->graphData->vertexProperties.find(var_name);
    if (it == data->graphData->vertexProperties.end()) {
        THROW exception::InvalidEnvProperty("Property '%s' was not found within graph '%s', "
            "in EnvironmentGraphVis::setXYProperty()\n",
            var_name.c_str(), data->graphData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 2) {
        THROW exception::InvalidEnvProperty("Visualisation position x property must be type float[2], graph '%s' property '%s' is type %s[%u], "
            "in EnvironmentGraphVis::setXYProperty()\n",
            data->graphData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->x_varName.clear();
    data->y_varName.clear();
    data->z_varName.clear();
    data->xyz_varName.clear();
    data->xy_varName = var_name;
}
void EnvironmentGraphVis::setXYZVertexProperty(const std::string& var_name) {
    auto it = data->graphData->vertexProperties.find(var_name);
    if (it == data->graphData->vertexProperties.end()) {
        THROW exception::InvalidEnvProperty("Property '%s' was not found within graph '%s', "
            "in EnvironmentGraphVis::setXYZProperty()\n",
            var_name.c_str(), data->graphData->name.c_str());
    } else if (it->second.type != std::type_index(typeid(float)) || it->second.elements != 3) {
        THROW exception::InvalidEnvProperty("Visualisation position x property must be type float[3], graph '%s' property '%s' is type %s[%u], "
            "in EnvironmentGraphVis::setXYZProperty()\n",
            data->graphData->name.c_str(), var_name.c_str(), it->second.type.name(), it->second.elements);
    }
    data->x_varName.clear();
    data->y_varName.clear();
    data->y_varName.clear();
    data->xy_varName.clear();
    data->xyz_varName = var_name;
}
void EnvironmentGraphVis::setColor(const Color& color) {
    data->color = color;
}
}  // namespace visualiser
}  // namespace flamegpu
