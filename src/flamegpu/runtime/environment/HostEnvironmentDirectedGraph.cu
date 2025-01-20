#include "flamegpu/runtime/environment/HostEnvironmentDirectedGraph.cuh"

#include <cstring>
#include <stdexcept>
#include <utility>
#include <string>
#include <memory>

#include "flamegpu/io/JSONGraphReader.h"
#include "flamegpu/io/JSONGraphWriter.h"

namespace flamegpu {
HostEnvironmentDirectedGraph::HostEnvironmentDirectedGraph(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>& _directed_graph, const cudaStream_t _stream,
    detail::CUDAScatter& _scatter, const unsigned int _streamID)
    : directed_graph(_directed_graph)
    , stream(_stream)
#ifdef FLAMEGPU_ADVANCED_API
    , scatter(_scatter)
    , streamID(_streamID)
#endif
{ }
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
        dg->setVertexCount(count, stream);
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

#ifdef FLAMEGPU_ADVANCED_API
std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> HostEnvironmentDirectedGraph::getCUDABuffers() {
    if (const auto dg = directed_graph.lock()) {
        return dg;
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeSourceDestination()\n");
    }
}
#endif


typedef HostEnvironmentDirectedGraph::VertexMap VertexMap;
typedef HostEnvironmentDirectedGraph::VertexMap::Vertex Vertex;

VertexMap HostEnvironmentDirectedGraph::vertices() {
    if (const auto dg = directed_graph.lock()) {
        return VertexMap(dg, stream);
    }
    THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::vertices()\n");
}
VertexMap::VertexMap(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, const cudaStream_t _stream)
    : directed_graph(std::move(_directed_graph))
    , stream(_stream) { }
size_type VertexMap::size() const {
    return directed_graph->getReadyVertexCount();
}
size_type VertexMap::allocated_size() const {
    return directed_graph->getVertexCount();
}
Vertex VertexMap::atIndex(unsigned int index) {
    return Vertex{ directed_graph, stream, index, true };
}
Vertex VertexMap::operator[](id_t vertex_id) {
    // Attempt to create vertex in id_map if it doesn't already exist
    directed_graph->createIfNotExistVertex(vertex_id, stream);
    // Return
    return Vertex{directed_graph, stream, vertex_id};
}

Vertex::Vertex(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, const cudaStream_t _stream, id_t _vertex_id, bool is_index)
    : directed_graph(std::move(_directed_graph))
    , stream(_stream)
    , vertex_index(is_index ? _vertex_id : directed_graph->getVertexIndex(_vertex_id)) { }
void Vertex::setID(id_t vertex_identifier) {
    // Update ID
    directed_graph->setVertexID(vertex_index, vertex_identifier, stream);
}
id_t Vertex::getID() const {
    return getProperty<id_t>(ID_VARIABLE_NAME);
}

typedef HostEnvironmentDirectedGraph::EdgeMap EdgeMap;
typedef HostEnvironmentDirectedGraph::EdgeMap::Edge Edge;

EdgeMap HostEnvironmentDirectedGraph::edges() {
    if (const auto dg = directed_graph.lock()) {
        return EdgeMap(dg, stream);
    }
    THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::edges()\n");
}
EdgeMap::EdgeMap(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, const cudaStream_t _stream)
    : directed_graph(std::move(_directed_graph))
    , stream(_stream) { }

size_type EdgeMap::size() const {
    return directed_graph->getReadyEdgeCount();
}
size_type EdgeMap::allocated_size() const {
    return directed_graph->getEdgeCount();
}
Edge EdgeMap::atIndex(unsigned int index) {
    return Edge{ directed_graph, stream, index };
}
Edge EdgeMap::operator[](SrcDestPair source_dest_vertex_ids) {
    // Attempt to create edge in id_map if it doesn't already exist
    directed_graph->createIfNotExistEdge(source_dest_vertex_ids.first, source_dest_vertex_ids.second, stream);
    // Return
    return Edge{directed_graph, stream, source_dest_vertex_ids.first, source_dest_vertex_ids.second};
}

Edge::Edge(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, const cudaStream_t _stream, id_t _source_vertex_id, id_t _dest_vertex_id)
    : directed_graph(std::move(_directed_graph))
    , stream(_stream)
    , edge_index(directed_graph->getEdgeIndex(_source_vertex_id, _dest_vertex_id)) { }

Edge::Edge(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, const cudaStream_t _stream, unsigned int _edge_index)
    : directed_graph(std::move(_directed_graph))
    , stream(_stream)
    , edge_index(_edge_index)
    { }
void Edge::setSourceVertexID(id_t _source_vertex_id) {
    // Update ID
    directed_graph->setEdgeSource(edge_index, _source_vertex_id);
}
void Edge::setDestinationVertexID(id_t _dest_vertex_id) {
    // Update ID
    directed_graph->setEdgeDestination(edge_index, _dest_vertex_id);
}
void Edge::setSourceDestinationVertexID(id_t _source_vertex_id, id_t _dest_vertex_id) {
    // Update ID
    directed_graph->setEdgeSourceDestination(edge_index, _source_vertex_id, _dest_vertex_id);
}
id_t Edge::getSourceVertexID() const {
    return getProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, 1);
}
id_t Edge::getDestinationVertexID() const {
    return getProperty<id_t, 2>(GRAPH_SOURCE_DEST_VARIABLE_NAME, 0);
}
}  // namespace flamegpu
