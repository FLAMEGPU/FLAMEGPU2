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
    const cudaStream_t stream;
#ifdef FLAMEGPU_ADVANCED_API
    detail::CUDAScatter &scatter;
    const unsigned int streamID;
#endif

 public:
    class VertexMap;
    class EdgeMap;
    HostEnvironmentDirectedGraph(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers>& _directed_graph, cudaStream_t _stream, detail::CUDAScatter& _scatter, unsigned int _streamID);
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
    /**
     * Returns a map for accessing vertices
     * The vertices are accessed via their ID, and will be created if they don't already exist
     * It is necessary to first call setVertexCount() to initialise the storage for vertices
     */
    VertexMap vertices();
    /**
     * Returns a map for accessing edges
     * The edges are accessed via their source-destination pair, and will be created if they don't already exist
     * It is necessary to first call setEdgeCount() to initialise the storage for edges
     */
    EdgeMap edges();
#ifdef FLAMEGPU_ADVANCED_API
    /**
     * Returns a shared_ptr to the CUDAEnvironmentDirectedGraphBuffers object which allows direct access to the graph's buffers
     * @note You may need to manually call markForRebuild() if you update edge source/dest pairs, to ensure the CSR/CSC are rebuilt
     */
    std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> getCUDABuffers();
    /**
     * Force rebuild the internal graph structure
     * This causes edges to be sorted
     */
    void rebuild() {
        if (const auto dg = directed_graph.lock()) {
            dg->markForRebuild();
            dg->syncDevice_async(scatter, streamID, stream);
        } else {
            THROW exception::ExpiredWeakPtr("Graph nolonger exists, weak pointer could not be locked, in HostEnvironmentDirectedGraph::rebuild()\n");
        }
    }
#endif
    /**
     * A map for accessing vertex storage via vertex ID
     */
    class VertexMap {
        std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph;
        const cudaStream_t stream;
        friend VertexMap HostEnvironmentDirectedGraph::vertices();
        VertexMap(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph, cudaStream_t stream);

     public:
        class Vertex;
        struct Iterator;
        /**
         * @return The number of vertices with a valid ID
         */
        size_type size() const;
        /**
         * @return The number of vertices that memory has been allocated for
         * @see HostEnvironmentDirectedGraph::setVertexCount()
         */
        size_type allocated_size() const;
        /**
         * Return a view into the vertex map of the specified vertex
         * If the vertex has not already been created, it will be created
         * @param vertex_id Identifier of the vertex to access
         * @throw exception::IDOutOfBounds 0 is not a valid value of vertex_id
         * @throw exception::OutOfBoundsException If more vertices have been created, than were allocated via HostEnvironmentDirectedGraph::setVertexCount()
         */
        Vertex operator[](id_t vertex_id);
        /**
         * Return a view into the allocated vertex buffer at the specified index
         * @throw exception::OutOfBoundsException If index exceeds the number of vertices allocated via HostEnvironmentDirectedGraph::setVertexCount()
         */
        Vertex atIndex(unsigned int index);
         class Vertex {
             std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph;
             const cudaStream_t stream;
             unsigned int vertex_index;
             friend Vertex VertexMap::operator[](id_t);
             friend struct Iterator;

          public:
             Vertex(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, cudaStream_t _stream, id_t _vertex_id, bool is_index = false);
             /**
              * Set the id for the current vertex
              * @param vertex_id The value to set the current vertex's id
              */
             void setID(id_t vertex_id);
             /**
              * Set the value of the specified property of the current vertex
              * @param property_name The name of the property to set
              * @param property_value The value to set the current vertex's property
              * @tparam T The type of the property
              *
              * @throw exception::InvalidArgument If property_name does not refer to a valid vertex property
              */
             template<typename T>
             void setProperty(const std::string& property_name, T property_value);
             template<typename T, flamegpu::size_type N = 0>
             void setProperty(const std::string& property_name, flamegpu::size_type element_index, T property_value);
             template<typename T, flamegpu::size_type N>
             void setProperty(const std::string& property_name, const std::array<T, N>& property_value);
             /**
              * Returns the id for the current vertex
              */
             id_t getID() const;
             /**
              * Returns the value of the specified property of the current vertex
              * @param property_name The name of the property to set
              * @tparam T The type of the property
              *
              * @throw exception::InvalidArgument If property_name does not refer to a valid vertex property
              * @throw exception::InvalidGraphProperty If a vertex property with the matching name and type does not exist
              */
             template<typename T>
             T getProperty(const std::string& property_name) const;
             template<typename T, flamegpu::size_type N = 0>
             T getProperty(const std::string& property_name, unsigned int element_index) const;
             template<typename T, flamegpu::size_type N>
             std::array<T, N> getProperty(const std::string& property_name) const;

#ifdef SWIG
             template<typename T>
             void setPropertyArray(const std::string& property_name, const std::vector<T>& property_value) const;
             template<typename T>
             std::vector<T> getPropertyArray(const std::string& property_name) const;
#endif
         };

        /**
         * Iterator for VertexMap
         *
         * Unlike a traditional C++ map iterator, this does not dereference to a key-value pair,
         * as it returns all allocated vertices including those which have not had their key
         * (ID) set, such IDs evaluate as ID_NOT_SET
         *
         * @see HostEnvironmentDirectedGraph::VertexMap::begin()
         * @see HostEnvironmentDirectedGraph::VertexMap::end()
         */
        struct Iterator {
            using iterator_category = std::forward_iterator_tag;
            using difference_type = unsigned int;
            using value_type = Vertex;
            using pointer = Vertex*;
            using reference = Vertex&;

            Iterator(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, const cudaStream_t _stream, difference_type _index)
                : directed_graph(std::move(_directed_graph))
                , stream(_stream)
                , vertex(directed_graph, _stream, _index, true) {}
            reference operator*() const { return vertex; }
            pointer operator->() const { return &vertex; }
            Iterator& operator++() { vertex.vertex_index++; return *this; }
            Iterator operator++(int) { Iterator tmp = *this; ++(*this);  return tmp; }

            bool operator==(const Iterator& other) const { return vertex.vertex_index == other.vertex.vertex_index; }
            bool operator!=(const Iterator& other) const { return vertex.vertex_index != other.vertex.vertex_index; }

            Iterator begin() { return { directed_graph, stream, 0 }; }
            Iterator end() { return { directed_graph, stream, directed_graph->getVertexCount() }; }

         private:
            std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph;
            cudaStream_t stream;
            mutable Vertex vertex;
        };

        /**
         * Returns an iterator to the first vertex in the buffer that backs the vertex map
         *
         * Unlike a traditional C++ map iterator, this does not dereference to a key-value pair,
         * as it returns all allocated vertices including those which have not had their key
         * (ID) set, such IDs evaluate as ID_NOT_SET
         *
         */
        Iterator begin() { return {directed_graph, stream, 0}; }
        /**
         * Returns an iterator to one beyond the last vertex in the buffer that backs the vertex map
         *
         * Unlike a traditional C++ map iterator, this does not dereference to a key-value pair,
         * as it returns all allocated vertices including those which have not had their key
         * (ID) set, such IDs evaluate as ID_NOT_SET
         *
         */
        Iterator end() { return { directed_graph, stream, directed_graph->getVertexCount() }; }
    };
    /**
     * A map for accessing edge storage via vertex ID
     */
    class EdgeMap {
        std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph;
        const cudaStream_t stream;
        friend EdgeMap HostEnvironmentDirectedGraph::edges();
        EdgeMap(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph, cudaStream_t stream);
        typedef std::pair<id_t, id_t> SrcDestPair;

     public:
        struct Iterator;
        class Edge;
        /**
         * @return The number of edges with a valid source/destination vertex pair
         */
        size_type size() const;
        /**
         * @return The number of edges that memory has been allocated for
         * @see HostEnvironmentDirectedGraph::setEdgeCount()
         */
        size_type allocated_size() const;
        /**
         * Return a view into the edge map of the specified edge
         * If the edge has not already been created, it will be created
         * @param source_dest_vertex_ids Identifier of the source vertex of the edge
         * @throw exception::IDOutOfBounds 0 is not a valid value of edge_id
         * @throw exception::OutOfBoundsException If more vertices have been created, than were allocated via HostEnvironmentDirectedGraph::setEdgeCount()
         * @todo C++23 support, make 2 arg version
         */
        Edge operator[](SrcDestPair source_dest_vertex_ids);
        /**
         * Return a view into the allocated edge buffer at the specified index
         * @throw exception::OutOfBoundsException If index exceeds the number of vertices allocated via HostEnvironmentDirectedGraph::setEdgeCount()
         */
        Edge atIndex(unsigned int index);
        class Edge {
            std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph;
            const cudaStream_t stream;
            unsigned int edge_index;
            friend Edge EdgeMap::operator[](SrcDestPair);
            friend struct Iterator;

         public:
            Edge(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, cudaStream_t _stream, id_t _source_vertex_id, id_t _dest_vertex_id);
            Edge(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, cudaStream_t _stream, unsigned int edge_index);
            /**
             * Set the source vertex's id for the current edge
             * @param source_vertex_id The value to set the current edge's source vertex id
             */
            void setSourceVertexID(id_t source_vertex_id);
            /**
             * Set the destination vertex's id for the current edge
             * @param dest_vertex_id The value to set the current edge's destination vertex id
             */
            void setDestinationVertexID(id_t dest_vertex_id);
            /**
             * Set the source and destination vertices' ids for the current edge
             * @param source_vertex_id The value to set the current edge's source vertex id
             * @param dest_vertex_id The value to set the current edge's destination vertex id
             */
            void setSourceDestinationVertexID(id_t source_vertex_id, id_t dest_vertex_id);
            /**
             * Set the value of the specified property of the current edge
             * @param property_name The name of the property to set
             * @param property_value The value to set the current edge's property
             * @tparam T The type of the property
             *
             * @throw exception::InvalidArgument If property_name does not refer to a valid edge property
             */
            template<typename T>
            void setProperty(const std::string& property_name, T property_value);
            template<typename T, flamegpu::size_type N = 0>
            void setProperty(const std::string& property_name, flamegpu::size_type element_index, T property_value);
            template<typename T, flamegpu::size_type N>
            void setProperty(const std::string& property_name, const std::array<T, N>& property_value);
            /**
             * Returns the source vertex's id for the current edge
             */
            id_t getSourceVertexID() const;
            /**
             * Returns the destination vertex's id for the current edge
             */
            id_t getDestinationVertexID() const;
            /**
             * Returns the value of the specified property of the current edge
             * @param property_name The name of the property to set
             * @tparam T The type of the property
             *
             * @throw exception::InvalidArgument If property_name does not refer to a valid edge property
             * @throw exception::InvalidGraphProperty If a edge property with the matching name and type does not exist
             */
            template<typename T>
            T getProperty(const std::string& property_name) const;
            template<typename T, flamegpu::size_type N = 0>
            T getProperty(const std::string& property_name, unsigned int element_index) const;
            template<typename T, flamegpu::size_type N>
            std::array<T, N> getProperty(const std::string& property_name) const;
#ifdef SWIG
            template<typename T>
            void setPropertyArray(const std::string& property_name, const std::vector<T>& property_value) const;
            template<typename T>
            std::vector<T> getPropertyArray(const std::string& property_name) const;
#endif
        };

        /**
         * Iterator for EdgeMap
         *
         * Unlike a traditional C++ map iterator, this does not dereference to a key-value pair,
         * as it returns all allocated edges including those which have not had their key
         * (source/destination vertex) set, such IDs evaluate as ID_NOT_SET
         *
         * @see HostEnvironmentDirectedGraph::EdgeMap::begin()
         * @see HostEnvironmentDirectedGraph::EdgeMap::end()
         */
        struct Iterator {
            using iterator_category = std::forward_iterator_tag;
            using difference_type = unsigned int;
            using value_type = Edge;
            using pointer = Edge*;
            using reference = Edge&;

            Iterator(std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> _directed_graph, const cudaStream_t _stream, difference_type _index)
                : directed_graph(std::move(_directed_graph))
                , stream(_stream)
                , edge(directed_graph, _stream, _index) {}
            reference operator*() const { return edge; }
            pointer operator->() const { return &edge; }
            Iterator& operator++() { edge.edge_index++; return *this; }
            Iterator operator++(int) { Iterator tmp = *this; ++(*this);  return tmp; }

            bool operator==(const Iterator& other) const { return edge.edge_index == other.edge.edge_index; }
            bool operator!=(const Iterator& other) const { return edge.edge_index != other.edge.edge_index; }

            Iterator begin() { return { directed_graph, stream, 0 }; }
            Iterator end() { return { directed_graph, stream, directed_graph->getEdgeCount() }; }

         private:
            std::shared_ptr<detail::CUDAEnvironmentDirectedGraphBuffers> directed_graph;
            cudaStream_t stream;
            mutable Edge edge;
        };

        /**
         * Returns an iterator to the first edge in the buffer that backs the edge map
         *
         * Unlike a traditional C++ map iterator, this does not dereference to a key-value pair,
         * as it returns all allocated edges including those which have not had their key
         * (source/destination vertex) set, such IDs evaluate as ID_NOT_SET
         *
         * @note The order of edges is not stable and is likely to change when the graph is rebuilt
         */
        Iterator begin() { return {directed_graph, stream, 0}; }
        /**
         * Returns an iterator to one beyond the last edge in the buffer that backs the edge map
         *
         * Unlike a traditional C++ map iterator, this does not dereference to a key-value pair,
         * as it returns all allocated edges including those which have not had their key
         * (source/destination vertex) set, such IDs evaluate as ID_NOT_SET
         *
         * @note The order of edges is not stable and is likely to change when the graph is rebuilt
         */
        Iterator end() { return { directed_graph, stream, directed_graph->getEdgeCount() }; }
    };
};

#ifdef SWIG
template<typename T>
void HostEnvironmentDirectedGraph::VertexMap::Vertex::setPropertyArray(const std::string& property_name, const std::vector<T>& property_value) const {
    size_type n = property_value.size();
    T* val = directed_graph->getVertexPropertyBuffer<T>(property_name, n, stream);
    if (!property_value.size()) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' length mismatch '%u' != '%u', in HostEnvironmentDirectedGraph::VertexMap::Vertex::setPropertyArray()",
            property_name.c_str(), n, static_cast<unsigned int>(property_value.size()));
    }
    const size_type vertex_count = directed_graph->getVertexCount();
    memcpy(val + vertex_index * property_value.size(), property_value.data(), sizeof(T) * property_value.size());
}
template<typename T>
std::vector<T> HostEnvironmentDirectedGraph::VertexMap::Vertex::getPropertyArray(const std::string& property_name) const {
    return directed_graph->getVertexPropertyArray<T>(property_name, vertex_index, stream);
}
template<typename T>
void HostEnvironmentDirectedGraph::EdgeMap::Edge::setPropertyArray(const std::string& property_name, const std::vector<T>& property_value) const {
    size_type n = property_value.size();
    T* val = directed_graph->getEdgePropertyBuffer<T>(property_name, n, stream);
    if (!property_value.size()) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' length mismatch '%u' != '%u', in HostEnvironmentDirectedGraph::EdgeMap::Edge::setPropertyArray()",
            property_name.c_str(), n, static_cast<unsigned int>(property_value.size()));
    }
    memcpy(val + edge_index * property_value.size(), property_value.data(), sizeof(T) * property_value.size());
}
template<typename T>
std::vector<T> HostEnvironmentDirectedGraph::EdgeMap::Edge::getPropertyArray(const std::string& property_name) const {
    return directed_graph->getEdgePropertyArray<T>(property_name, edge_index, stream);
}
#endif

template<typename T>
void HostEnvironmentDirectedGraph::VertexMap::Vertex::setProperty(const std::string& property_name, T property_value) {
    size_type element_ct = 1;
    T* rtn = directed_graph->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
    rtn[vertex_index] = property_value;
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::VertexMap::Vertex::setProperty(const std::string& property_name, flamegpu::size_type element_index, T property_value) {
    size_type element_ct = N;
    T* rtn = directed_graph->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
    if (element_index > element_ct) {
        THROW exception::OutOfBoundsException("Element index %u is out of range %u, in VertexMap::Vertex::setProperty()\n",
            element_index, element_ct);
    }
    rtn[vertex_index * element_ct + element_index] = property_value;
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::VertexMap::Vertex::setProperty(const std::string& property_name, const std::array<T, N>& property_value) {
    size_type element_ct = N;
    T* rtn = directed_graph->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
    std::array<T, N>* rtn2 = reinterpret_cast<std::array<T, N>*>(rtn);
    rtn2[vertex_index] = property_value;
}
template<typename T>
T HostEnvironmentDirectedGraph::VertexMap::Vertex::getProperty(const std::string& property_name) const {
    size_type element_ct = 1;
    const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers *>(directed_graph.get())->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
    return rtn[vertex_index];
}
template<typename T, size_type N>
T HostEnvironmentDirectedGraph::VertexMap::Vertex::getProperty(const std::string& property_name, unsigned int element_index) const {
    size_type element_ct = N;
    const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(directed_graph.get())->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
    if (element_index > element_ct) {
        THROW exception::OutOfBoundsException("Element index %u is out of range %u, in VertexMap::Vertex::getProperty()\n",
            element_index, element_ct);
    }
    return rtn[vertex_index * element_ct + element_index];
}
template<typename T, size_type N>
std::array<T, N> HostEnvironmentDirectedGraph::VertexMap::Vertex::getProperty(const std::string& property_name) const {
    size_type element_ct = N;
    const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(directed_graph.get())->getVertexPropertyBuffer<T>(property_name, element_ct, stream);
    const std::array<T, N>* rtn2 = reinterpret_cast<const std::array<T, N>*>(rtn);
    return rtn2[vertex_index];
}

template<typename T>
void HostEnvironmentDirectedGraph::EdgeMap::Edge::setProperty(const std::string& property_name, const T property_value) {
    size_type element_ct = 1;
    T* val = directed_graph->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
    val[edge_index] = property_value;
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::EdgeMap::Edge::setProperty(const std::string& property_name, const size_type element_index, const T property_value) {
    size_type element_ct = N;
    T* val = directed_graph->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
    if (element_index >= element_ct) {
        THROW exception::OutOfBoundsException("Element index %u is out of range %u, in HostEnvironmentDirectedGraph::EdgeMap::Edge::setProperty()\n",
            element_index, element_ct);
    }
    val[edge_index * element_ct + element_index] = property_value;
}
template<typename T, flamegpu::size_type N>
void HostEnvironmentDirectedGraph::EdgeMap::Edge::setProperty(const std::string& property_name, const std::array<T, N>& property_value) {
    size_type element_ct = N;
    T* val = directed_graph->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
    std::array<T, N>* val2 = reinterpret_cast<std::array<T, N>*>(val);
    val2[edge_index] = property_value;
}
template<typename T>
T HostEnvironmentDirectedGraph::EdgeMap::Edge::getProperty(const std::string& property_name) const {
    size_type element_ct = 1;
    const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(directed_graph.get())->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
    return rtn[edge_index];
}
template<typename T, flamegpu::size_type N>
T HostEnvironmentDirectedGraph::EdgeMap::Edge::getProperty(const std::string& property_name, const size_type element_index) const {
    size_type element_ct = N;
    const T* rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(directed_graph.get())->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
    if (element_index >= element_ct) {
        THROW exception::OutOfBoundsException("Element index %u is out of range %u, in HostEnvironmentDirectedGraph::EdgeMap::Edge::getProperty()\n",
            element_index, element_ct);
    }
    return rtn[edge_index * element_ct + element_index];
}
template<typename T, flamegpu::size_type N>
std::array<T, N> HostEnvironmentDirectedGraph::EdgeMap::Edge::getProperty(const std::string& property_name) const {
    size_type element_ct = N;
    const T *rtn = const_cast<const detail::CUDAEnvironmentDirectedGraphBuffers*>(directed_graph.get())->getEdgePropertyBuffer<T>(property_name, element_ct, stream);
    const std::array<T, N>* rtn2 = reinterpret_cast<const std::array<T, N>*>(rtn);
    return rtn2[edge_index];
}
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_HOSTENVIRONMENTDIRECTEDGRAPH_CUH_
