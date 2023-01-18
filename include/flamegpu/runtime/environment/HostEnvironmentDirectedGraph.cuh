#ifndef INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_HOSTENVIRONMENTDIRECTEDGRAPH_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_HOSTENVIRONMENTDIRECTEDGRAPH_CUH_

#include <utility>
#include <string>
#include <vector>
#include <memory>
#include <array>

#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"
#include "flamegpu/defines.h"

namespace flamegpu {
/**
 * @brief HostAPI handle to a specific directed graph stored within the environment
 *
 * Allows a specific directed graph to be updated by creating new vertices/edges, or updating properties assigned to them
 */
class HostEnvironmentDirectedGraph {
    // Windows/Python edge case, if an exception is thrown this object is destructed super late
    // So be extra safe with weak ptr
    std::weak_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph;
    detail::CUDAScatter &scatter;
    const unsigned int streamID;
    const cudaStream_t stream;

 public:
    HostEnvironmentDirectedGraph(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>& _directed_graph, detail::CUDAScatter& _scatter, unsigned int _streamID, cudaStream_t _stream);
    ~HostEnvironmentDirectedGraph();
    // Graph Structure Modifiers
    /**
     * Attempts to import edge and vertex data from the specified file
     *
     * This file must be in the appropriate format as documented in the FLAMEGPU documentation
     *
     * @param in_file Path to the file on disk containing the graph
     *
     * @throws exception::InvalidFilePath If the file does not exist, or cannot be opened
     * @throws exception::UnsupportedFileType If the specified file type does not correspond to a known format
     * @throws exception::InvalidInputFile If the loaded file cannot be parsed as a valid graph
     *
     * @see exportGraph()
     */
    void importGraph(const std::string &in_file);
    /**
     * Attempts to export the edge and vertex data to the specified file
     *
     * @param out_file Path to the file on disk containing to store the graph
     *
     * @throws exception::InvalidFilePath If the file cannot be opened for writing
     * @throws exception::UnsupportedFileType If the specified file type does not correspond to a known format
     * @throws exception::RapidJSONError If conversion to JSON fails for any reason
     *
     * @see importGraph()
     */
    void exportGraph(const std::string &out_file);
    /**
     * Set the number of vertices present in the graph
     * This causes the internal data structure to be (re)allocated, and existing vertex data is not retained.
     * Calling this regularly may harm performance
     *
     * @param count The number of vertices
     * @note Calling this will also invalidate any existing edges (as the vertices will nolonger exist)
     */
    void setVertexCount(flamegpu::size_type count);
    /**
     * Returns the current number of vertices present in the graph
     */
    size_type getVertexCount();
    /**
     * Set the number of edges present in the graph
     * This causes the internal data structure to be (re)allocated, and existing edge data is not retained.
     * Calling this regularly may harm performance
     *
     * @param count The number of edges
     */
    void setEdgeCount(flamegpu::size_type count);
    /**
     * Returns the current number of edges present in the graph
     */
    size_type getEdgeCount();
    // Individual Property Accessors
    /**
     * Set the identifier for the vertex at the specific index
     * @param vertex_index Index of the vertex to update
     * @param vertex_identifier The value to set the specified vertex's identifier
     *
     * @throw exception::InvalidArgument If the vertex_identifier is not unique among other vertex IDs, or 0
     * @throw exception::OutOfBoundsException If vertex_index exceeds the number of vertices
     */
    void setVertexID(unsigned int vertex_index, id_t vertex_identifier);
    /**
     * Set the value of the specified property of the vertex at the specific index
     * @param property_name The name of the property to set
     * @param vertex_index Index of the vertex to update
     * @param property_value The value to set the specified vertex's property
     * @tparam T The type of the property
     *
     * @throw exception::InvalidArgument If property_name does not refer to a valid vertex property
     * @throw exception::InvalidArgument If a vertex property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If vertex_index exceeds the number of vertices
     */
    template<typename T>
    void setVertexProperty(const std::string& property_name, unsigned int vertex_index, T property_value);
    template<typename T, flamegpu::size_type N = 0>
    void setVertexProperty(const std::string& property_name, unsigned int vertex_index, flamegpu::size_type element_index, T property_value);
    template<typename T, flamegpu::size_type N>
    void setVertexProperty(const std::string& property_name, unsigned int vertex_index, const std::array<T, N>& property_value);
    /**
     * Sets the source vertex ID of the specified edge
     * @param edge_index Index of the edge to update
     * @param vertex_source_id The source vertex ID to assign to the edge
     *
     * @throw exception::InvalidArgument If vertex_source_id is not a valid vertex ID, or 0
     * @throw exception::OutOfBoundsException If edge_index exceeds the number of edges
     * @see setEdgeDestination()
     */
    void setEdgeSource(unsigned int edge_index, id_t vertex_source_id);
    /**
     * Sets the destination vertex ID of the specified edge
     * @param edge_index Index of the edge to update
     * @param vertex_dest_id The destination vertex ID to assign to the edge
     *
     * @throw exception::InvalidArgument If vertex_dest_id is not a valid vertex ID, or 0
     * @throw exception::OutOfBoundsException If edge_index exceeds the number of edges
     * @see setEdgeSource()
     */
    void setEdgeDestination(unsigned int edge_index, id_t vertex_dest_id);
    /**
     * Sets the source and destination vertex IDs of the specified edge
     * @param edge_index Index of the edge to update
     * @param vertex_source_id The source vertex ID to assign to the edge
     * @param vertex_dest_id The destination vertex ID to assign to the edge
     *
     * @throw exception::InvalidArgument If vertex_source_id or vertex_dest_id is not a valid vertex ID, or 0
     * @throw exception::OutOfBoundsException If edge_index exceeds the number of edges
     * @see setEdgeSource()
     * @see setEdgeDestination()
     */
    void setEdgeSourceDestination(unsigned int edge_index, id_t vertex_source_id, id_t vertex_dest_id);
    /**
     * Set the value of the specified property of the edge at the specific index
     * @param property_name The name of the property to set
     * @param edge_index Index of the edge to update
     * @param property_value The value to set the specified edge's property
     * @tparam T The type of the property
     *
     * @throw exception::InvalidArgument If property_name does not refer to a valid edge property
     * @throw exception::InvalidGraphProperty If an edge property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If vertex_index exceeds the number of edges
     * @throw exception::ReservedName If property_name refers to an internal property
     */
    template<typename T>
    void setEdgeProperty(const std::string& property_name, unsigned int edge_index, T property_value);
    template<typename T, flamegpu::size_type N = 0>
    void setEdgeProperty(const std::string& property_name, unsigned int edge_index, flamegpu::size_type element_index, T property_value);
    template<typename T, flamegpu::size_type N>
    void setEdgeProperty(const std::string& property_name, unsigned int edge_index, const std::array<T, N> &property_value);
    /**
     * Returns the identifier for the vertex at the specific index
     * @param vertex_index Index of the vertex to retrieve
     *
     * @throw exception::OutOfBoundsException If vertex_index exceeds the number of vertices
     */
    id_t getVertexID(unsigned int vertex_index) const;
    /**
     * Returns the value of the specified property of the vertex at the specific index
     * @param property_name The name of the property to set
     * @param vertex_index Index of the vertex to retrieve
     * @tparam T The type of the property
     *
     * @throw exception::InvalidArgument If property_name does not refer to a valid vertex property
     * @throw exception::InvalidGraphProperty If a vertex property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If vertex_index exceeds the number of vertices
     */
    template<typename T>
    T getVertexProperty(const std::string& property_name, unsigned int vertex_index) const;
    template<typename T, flamegpu::size_type N = 0>
    T getVertexProperty(const std::string& property_name, unsigned int vertex_index, unsigned int element_index) const;
    template<typename T, flamegpu::size_type N>
    std::array<T, N> getVertexProperty(const std::string& property_name, unsigned int vertex_index) const;
    /**
     * Returns the source vertex ID of the specified edge
     * @param edge_index Index of the edge to retrieve
     *
     * @throw exception::InvalidArgument If vertex_source_id is not a valid vertex ID, or 0
     * @throw exception::OutOfBoundsException If edge_index exceeds the number of edges
     * @see setEdgeDestination()
     */
    id_t getEdgeSource(unsigned int edge_index) const;
    /**
     * Returns the destination vertex ID of the specified edge
     * @param edge_index Index of the edge to retrieve
     *
     * @throw exception::InvalidArgument If vertex_dest_id is not a valid vertex ID, or 0
     * @throw exception::OutOfBoundsException If edge_index exceeds the number of edges
     * @see setEdgeSource()
     */
    id_t getEdgeDestination(unsigned int edge_index) const;
    /**
     * Returns the source and destination vertex IDs of the specified edge
     * @param edge_index Index of the edge to retrieve
     *
     * @throw exception::InvalidArgument If vertex_source_id or vertex_dest_id is not a valid vertex ID, or 0
     * @throw exception::OutOfBoundsException If edge_index exceeds the number of edges
     * @see getEdgeSource()
     * @see getEdgeDestination()
     */
    std::pair<id_t, id_t> getEdgeSourceDestination(unsigned int edge_index) const;
    /**
     * Returns the value of the specified property of the edge at the specific index
     * @param property_name The name of the property to set
     * @param edge_index Index of the edge to retrieve
     * @tparam T The type of the property
     *
     * @throw exception::InvalidArgument If property_name does not refer to a valid edge property
     * @throw exception::InvalidGraphProperty If an edge property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If vertex_index exceeds the number of edges
     */
    template<typename T>
    T getEdgeProperty(const std::string& property_name, unsigned int edge_index) const;
    template<typename T, flamegpu::size_type N = 0>
    T getEdgeProperty(const std::string& property_name, unsigned int edge_index, unsigned int element_index) const;
    template<typename T, flamegpu::size_type N>
    std::array<T, N> getEdgeProperty(const std::string& property_name, unsigned int edge_index) const;
#ifdef SWIG
    template<typename T>
    void setVertexPropertyArray(const std::string& property_name, unsigned int vertex_index, const std::vector<T>& property_value) const;
    template<typename T>
    std::vector<T> getVertexPropertyArray(const std::string& property_name, unsigned int vertex_index) const;
    template<typename T>
    void setEdgePropertyArray(const std::string& property_name, unsigned int vertex_index, const std::vector<T>& property_value) const;
    template<typename T>
    std::vector<T> getEdgePropertyArray(const std::string& property_name, unsigned int edge_index) const;
#endif
#ifdef FLAMEGPU_ADVANCED_API
    /**
     * Returns a shared_ptr to the CUDAEnvironmentDirectedGraphBuffers object which allows direct access to the graph's buffers
     * @note You may need to manually call markForRebuild() if you update edge source/dest pairs, to ensure the CSR/CSC are rebuilt
     */
    std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> getCUDABuffers();
#endif
};

template<typename T>
T HostEnvironmentDirectedGraph::getVertexProperty(const std::string& property_name, const unsigned int vertex_index) const {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = 1;
        const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers *>(dg.get())->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
        const size_type vertex_count = dg->getVertexCount();
        if (vertex_index < vertex_count) {
            return rtn[vertex_index];
        }
        THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in HostEnvironmentDirectedGraph::getVertexProperty()\n",
        vertex_index, vertex_count);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getVertexProperty()\n");
    }
}
template<typename T>
T HostEnvironmentDirectedGraph::getEdgeProperty(const std::string& property_name, const unsigned int edge_index) const {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = 1;
        const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(dg.get())->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
        const size_type edge_count = dg->getEdgeCount();
        if (edge_index < edge_count) {
            return rtn[edge_index];
        }
        THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in HostEnvironmentDirectedGraph::getEdgeProperty()\n",
            edge_index, edge_count);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getEdgeProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
T HostEnvironmentDirectedGraph::getVertexProperty(const std::string& property_name, const unsigned int vertex_index, const size_type element_index) const {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(dg.get())->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
        const size_type vertex_count = dg->getVertexCount();
        if (vertex_index >= vertex_count) {
            THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in HostEnvironmentDirectedGraph::getVertexProperty()\n",
                vertex_index, vertex_count);
        } else if (element_index > element_ct) {
            THROW exception::OutOfBoundsException("Element index %u is out of range %u, in HostEnvironmentDirectedGraph::getVertexProperty()\n",
                element_index, element_ct);
        }
        return rtn[vertex_index * element_ct + element_index];
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getVertexProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
T HostEnvironmentDirectedGraph::getEdgeProperty(const std::string& property_name, const unsigned int edge_index, const size_type element_index) const {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(dg.get())->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
        const size_type edge_count = dg->getEdgeCount();
        if (edge_index >= edge_count) {
            THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in HostEnvironmentDirectedGraph::getEdgeProperty()\n",
                edge_index, edge_count);
        } else if (element_index >= element_ct) {
            THROW exception::OutOfBoundsException("Element index %u is out of range %u, in HostEnvironmentDirectedGraph::getEdgeProperty()\n",
                element_index, element_ct);
        }
        return rtn[edge_index * element_ct + element_index];
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getEdgeProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
std::array<T, N> HostEnvironmentDirectedGraph::getVertexProperty(const std::string& property_name, const unsigned int vertex_index) const {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(dg.get())->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
        const size_type vertex_count = dg->getVertexCount();
        if (vertex_index < vertex_count) {
            const std::array<T, N>* rtn2 = reinterpret_cast<const std::array<T, N>*>(rtn);
            return rtn2[vertex_index];
        }
        THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in HostEnvironmentDirectedGraph::getVertexProperty()\n",
            vertex_index, vertex_count);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getVertexProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
std::array<T, N> HostEnvironmentDirectedGraph::getEdgeProperty(const std::string& property_name, const unsigned int edge_index) const {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        const T * rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(dg.get())->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
        const size_type edge_count = dg->getEdgeCount();
        if (edge_index < edge_count) {
            const std::array<T, N>* rtn2 = reinterpret_cast<const std::array<T, N>*>(rtn);
            return rtn2[edge_index];
        }
        THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in HostEnvironmentDirectedGraph::getEdgeProperty()\n",
            edge_index, edge_count);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getEdgeProperty()\n");
    }
}

template<typename T>
void HostEnvironmentDirectedGraph::setVertexProperty(const std::string& property_name, const unsigned int vertex_index, const T property_value) {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = 1;
        T* val = dg->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
        const size_type vertex_count = dg->getVertexCount();
        if (vertex_index < vertex_count) {
            val[vertex_index] = property_value;
        } else {
            THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in HostEnvironmentDirectedGraph::setVertexProperty()\n",
                vertex_index, vertex_count);
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setVertexProperty()\n");
    }
}
template<typename T>
void HostEnvironmentDirectedGraph::setEdgeProperty(const std::string& property_name, const unsigned int edge_index, const T property_value) {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = 1;
        T* val = dg->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
        const size_type edge_count = dg->getEdgeCount();
        if (edge_index < edge_count) {
            val[edge_index] = property_value;
        } else {
            THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in HostEnvironmentDirectedGraph::setEdgeProperty()\n",
                edge_index, edge_count);
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::setVertexProperty(const std::string& property_name, const  unsigned int vertex_index, const size_type element_index, const T property_value) {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        T* val = dg->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
        const size_type vertex_count = dg->getVertexCount();
        if (vertex_index >= vertex_count) {
            THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in HostEnvironmentDirectedGraph::setVertexProperty()\n",
                vertex_index, vertex_count);
        } else if (element_index >= element_ct) {
            THROW exception::OutOfBoundsException("Element index %u is out of range %u, in HostEnvironmentDirectedGraph::setVertexProperty()\n",
                element_index, element_ct);
        } else {
            val[vertex_index * element_ct + element_index] = property_value;
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setVertexProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::setEdgeProperty(const std::string& property_name, const unsigned int edge_index, const size_type element_index, const T property_value) {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        T* val = dg->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
        const size_type edge_count = dg->getEdgeCount();
        if (edge_index >= edge_count) {
            THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in HostEnvironmentDirectedGraph::setEdgeProperty()\n",
                edge_index, edge_count);
        } else if (element_index >= element_ct) {
            THROW exception::OutOfBoundsException("Element index %u is out of range %u, in HostEnvironmentDirectedGraph::setEdgeProperty()\n",
                element_index, element_ct);
        } else {
            val[edge_index * element_ct + element_index] = property_value;
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::setVertexProperty(const std::string& property_name, const unsigned int vertex_index, const std::array<T, N>& property_value) {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        T* val = dg->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
        const size_type vertex_count = dg->getVertexCount();
        if (vertex_index < vertex_count) {
            std::array<T, N>* val2 = reinterpret_cast<std::array<T, N>*>(val);
            val2[vertex_index] = property_value;
        } else {
            THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in HostEnvironmentDirectedGraph::setVertexProperty()\n",
                vertex_index, vertex_count);
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setVertexProperty()\n");
    }
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::setEdgeProperty(const std::string& property_name, const unsigned int edge_index, const std::array<T, N>& property_value) {
    if (const auto dg = directed_graph.lock()) {
        size_type element_ct = N;
        T* val = dg->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
        const size_type edge_count = dg->getEdgeCount();
        if (edge_index < edge_count) {
            std::array<T, N>* val2 = reinterpret_cast<std::array<T, N>*>(val);
            val2[edge_index] = property_value;
        } else {
            THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in HostEnvironmentDirectedGraph::setEdgeProperty()\n",
                edge_index, edge_count);
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgeProperty()\n");
    }
}
#ifdef SWIG
template<typename T>
void HostEnvironmentDirectedGraph::setVertexPropertyArray(const std::string& property_name, const unsigned int vertex_index, const std::vector<T>& property_value) const {
    if (const auto dg = directed_graph.lock()) {
        size_type n = property_value.size();
        T* val = dg->getVertexPropertyBuffer<T>(property_name, n, stream);
        if (!property_value.size()) {
            THROW exception::InvalidGraphProperty("Vertex property with name '%s' length mismatch '%u' != '%u', in CUDAEnvironmentDirectedGraphBuffers::setVertexPropertyArray()",
                property_name.c_str(), n, static_cast<unsigned int>(property_value.size()));
        }
        const size_type vertex_count = dg->getVertexCount();
        if (vertex_index >= vertex_count) {
            THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in HostEnvironmentDirectedGraph::setEdgePropertyArray()\n",
                vertex_index, vertex_count);
        } else {
            memcpy(val + vertex_index * property_value.size(), property_value.data(), sizeof(T) * property_value.size());
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::setEdgePropertyArray()\n");
    }
}
template<typename T>
std::vector<T> HostEnvironmentDirectedGraph::getVertexPropertyArray(const std::string& property_name, const unsigned int vertex_index) const {
    if (const auto dg = directed_graph.lock()) {
        return dg->getVertexPropertyArray<T>(property_name, vertex_index, stream);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getVertexPropertyArray()\n");
    }
}
template<typename T>
void HostEnvironmentDirectedGraph::setEdgePropertyArray(const std::string& property_name, const unsigned int edge_index, const std::vector<T>& property_value) const {
    if (const auto dg = directed_graph.lock()) {
        size_type n = property_value.size();
        T* val = dg->getEdgePropertyBuffer<T>(property_name, n, stream);
        if (!property_value.size()) {
            THROW exception::InvalidGraphProperty("Edge property with name '%s' length mismatch '%u' != '%u', in CUDAEnvironmentDirectedGraphBuffers::setEdgePropertyArray()",
                property_name.c_str(), n, static_cast<unsigned int>(property_value.size()));
        }
        const size_type edge_count = dg->getEdgeCount();
        if (edge_index >= edge_count) {
            THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in HostEnvironmentDirectedGraph::getEdgePropertyArray()\n",
                edge_index, edge_count);
        } else {
            memcpy(val + edge_index * property_value.size(), property_value.data(), sizeof(T) * property_value.size());
        }
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getEdgePropertyArray()\n");
    }
}
template<typename T>
std::vector<T> HostEnvironmentDirectedGraph::getEdgePropertyArray(const std::string& property_name, const unsigned int edge_index) const {
    if (const auto dg = directed_graph.lock()) {
        return dg->getEdgePropertyArray<T>(property_name, edge_index, stream);
    } else {
        THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::getEdgePropertyArray()\n");
    }
}
#endif
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_HOSTENVIRONMENTDIRECTEDGRAPH_CUH_
