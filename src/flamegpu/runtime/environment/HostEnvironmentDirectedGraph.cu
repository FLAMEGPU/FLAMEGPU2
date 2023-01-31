#include "flamegpu/runtime/environment/HostEnvironmentDirectedGraph.cuh"

#include <cstring>

#include "flamegpu/io/JSONGraphReader.h"
#include "flamegpu/io/JSONGraphWriter.h"

namespace flamegpu {
HostEnvironmentDirectedGraph::HostEnvironmentDirectedGraph(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>& _directed_graph, detail::CUDAScatter& _scatter, const unsigned int _streamID, const cudaStream_t _stream)
    : directed_graph(_directed_graph)
    , scatter(_scatter)
    , streamID(_streamID)
    , stream(_stream)
{ }
HostEnvironmentDirectedGraph::~HostEnvironmentDirectedGraph() {
    if (const auto dg = directed_graph.lock())
        dg->syncDevice_async(scatter, streamID, stream);
}
void HostEnvironmentDirectedGraph::importGraph(const std::string& in_file) {
    // Case insensitive ends_with()
    const std::string json_type = ".json";
    bool is_json = true;
    for (unsigned int i = 1; i <= in_file.size() && i <= json_type.size(); ++i) {
        if (in_file[in_file.size() - i] != json_type[json_type.size() - i])
            is_json =  false;
    }
    if (in_file.size() < json_type.size() || !is_json) {
        THROW exception::UnsupportedFileType("Input file '%s' does not correspond to a supported format (e.g. .json)\n", in_file.c_str());
    }
    if (const auto dg = directed_graph.lock()) {
        io::JSONGraphReader::loadAdjacencyLike(in_file, dg, stream);
        dg->markForRebuild();
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::importGraph()\n");
    }
}
void HostEnvironmentDirectedGraph::exportGraph(const std::string& out_file) {
    // Case insensitive ends_with()
    const std::string json_type = ".json";
    bool is_json = true;
    for (unsigned int i = 1; i <= out_file.size() && i <= json_type.size(); ++i) {
        if (out_file[out_file.size() - i] != json_type[json_type.size() - i])
            is_json =  false;
    }
    if (out_file.size() < json_type.size() || !is_json) {
        THROW exception::UnsupportedFileType("Output file '%s' does not correspond to a supported format (e.g. .json)\n", out_file.c_str());
    }
    if (const auto dg = directed_graph.lock()) {
        std::shared_ptr<const detail::CUDAEnvironmentDirectedGraphBuffers> const_dg = std::const_pointer_cast<detail::CUDAEnvironmentDirectedGraphBuffers>(dg);
        io::JSONGraphWriter::saveAdjacencyLike(out_file, const_dg, stream, true);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::exportGraph()\n");
    }
}
void HostEnvironmentDirectedGraph::setVertexCount(const size_type count) {
    if (const auto dg = directed_graph.lock()) {
        dg->setVertexCount(count);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setVertexCount()\n");
    }
}
flamegpu::size_type HostEnvironmentDirectedGraph::getVertexCount() {
    if (const auto dg = directed_graph.lock()) {
        return dg->getVertexCount();
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getVertexCount()\n");
    }
}
void HostEnvironmentDirectedGraph::setEdgeCount(const size_type count) {
    if (const auto dg = directed_graph.lock()) {
        dg->setEdgeCount(count);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeCount()\n");
    }
}
flamegpu::size_type HostEnvironmentDirectedGraph::getEdgeCount() {
    if (const auto dg = directed_graph.lock()) {
        return dg->getEdgeCount();
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getEdgeCount()\n");
    }
}
void HostEnvironmentDirectedGraph::setEdgeSource(unsigned int edge_index, const id_t vertex_source_id) {
    setEdgeProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, edge_index, 1, vertex_source_id);
    if (const auto dg = directed_graph.lock()) {
        dg->markForRebuild();
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeSource()\n");
    }
}
void HostEnvironmentDirectedGraph::setEdgeDestination(unsigned int edge_index, const id_t vertex_dest_id) {
    setEdgeProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, edge_index, 0, vertex_dest_id);
    if (const auto dg = directed_graph.lock()) {
        dg->markForRebuild();
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeDestination()\n");
    }
}
void HostEnvironmentDirectedGraph::setEdgeSourceDestination(unsigned int edge_index, const id_t vertex_source_id, const id_t vertex_dest_id) {
    setEdgeProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, edge_index, {  vertex_dest_id, vertex_source_id });
    if (const auto dg = directed_graph.lock()) {
        dg->markForRebuild();
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeSourceDestination()\n");
    }
}
void HostEnvironmentDirectedGraph::setVertexID(const unsigned int vertex_index, const id_t vertex_identifier) {
    setVertexProperty<id_t>(ID_VARIABLE_NAME, vertex_index, vertex_identifier);
}

id_t HostEnvironmentDirectedGraph::getVertexID(const unsigned int vertex_index) const {
    return getVertexProperty<id_t>(ID_VARIABLE_NAME, vertex_index);
}
id_t HostEnvironmentDirectedGraph::getEdgeSource(const unsigned int edge_index) const {
    return getEdgeProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, edge_index, 1);
}
id_t HostEnvironmentDirectedGraph::getEdgeDestination(const unsigned int edge_index) const {
    return getEdgeProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, edge_index, 0);
}
std::pair<id_t, id_t> HostEnvironmentDirectedGraph::getEdgeSourceDestination(const unsigned int edge_index) const {
    const std::array<id_t, 2> t = getEdgeProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, edge_index);
    return std::pair<id_t, id_t>{t[1], t[0]};
}
#ifdef FLAMEGPU_ADVANCED_API
std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> HostEnvironmentDirectedGraph::getCUDABuffers() {
    if (const auto dg = directed_graph.lock()) {
        return dg;
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeSourceDestination()\n");
    }
}
#endif
}  // namespace flamegpu
