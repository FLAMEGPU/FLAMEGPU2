#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAENVIRONMENTDIRECTEDGRAPHBUFFERS_CUH_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAENVIRONMENTDIRECTEDGRAPHBUFFERS_CUH_

#include <string>
#include <map>
#include <list>
#include <memory>
#include <vector>
#include <unordered_map>
#include <utility>
#include <limits>

#include "flamegpu/model/EnvironmentDirectedGraphData.cuh"
#include "flamegpu/defines.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/detail/type_decode.h"
#include "flamegpu/util/StringPair.h"

namespace flamegpu {
namespace detail {
class CUDAScatter;
namespace curve {
class HostCurve;
class CurveRTCHost;
}

/**
 * This represents the equivalent of CUDAAgent, CUDAMessage for EnvironmentDirectedGraph
 * As the graph cannot be modified on the device, the host buffers can be assumed to always holds the truth
 * It is only necessary to ensure device buffers are updated to match if the host buffers have changed
 */
class CUDAEnvironmentDirectedGraphBuffers {
    struct Buffer {
        enum Ready { None = 0, Host = 1, Device = 2, Both = 3};
        /**
         * Size of an individual element
         */
        size_t element_size;
        /**
         * Pointer to buffer in device memory (if allocated)
         */
        void *d_ptr = nullptr;
        /**
         * Pointer to buffer swap space in device memory (if allocated)
         */
        void* d_ptr_swap = nullptr;
        /**
         * Pointer to buffer in host memory (if allocated)
         */
        void *h_ptr = nullptr;
        /**
         * Which buffers are ready
         */
        mutable Ready ready = None;
        /**
         * Swap the device buffers
         */
        void swap() { std::swap(d_ptr, d_ptr_swap); }
        /**
         * If host buffer is not ready, copy from device buffer
         */
        void updateHostBuffer(size_type edge_count, cudaStream_t stream) const;
    };
    const EnvironmentDirectedGraphData &graph_description;
    std::map<std::string, Buffer> vertex_buffers;
    std::map<std::string, Buffer> edge_buffers;
    std::list<std::weak_ptr<detail::curve::HostCurve>> curve_instances;
    std::list<std::weak_ptr<detail::curve::CurveRTCHost>> rtc_curve_instances;
    size_type vertex_count;
    size_type edge_count;
    bool requires_rebuild;
    uint64_t *d_keys = nullptr, *d_keys_swap = nullptr;
    uint32_t *d_vals = nullptr, *d_vals_swap = nullptr;
    // CSR/VBM (edgesLeaving())
    unsigned int *d_pbm = nullptr, *d_pbm_swap = nullptr;
    // CSC/invertedVBM (edgesJoining()), shares d_pbm_swap
    unsigned int *d_ipbm = nullptr;
    // Copy of the vals list from constructing ipbm, required to lookup edges
    unsigned int *d_ipbm_edges = nullptr;
    // Vertex ID -> index map, ?? has been reserved, otherwise any ID is valid
    // However, the ID->index map does not utilise any compression, so non-contiguous ID ranges may lead to out of memory errors.
    unsigned int *d_vertex_index_map = nullptr;

    void allocateVertexBuffers(size_type count, cudaStream_t stream);
    void allocateEdgeBuffers(size_type count);
    void deallocateVertexBuffers();
    void deallocateEdgeBuffers();
    /*
     * Reset the internal vertex ID range tracking variables as though no vertices have been assigned IDs
     */
    void resetVertexIDBounds();
    unsigned int vertex_id_min = std::numeric_limits<unsigned int>::max();
    unsigned int vertex_id_max = std::numeric_limits<unsigned int>::min();
    /**
     * Host ID map, this allows HostAPI methods to operate using vertex id
     */
    std::map<id_t, unsigned int> h_vertex_index_map;
    util::PairMap<id_t, id_t, unsigned int> h_edge_index_map;

 public:
    /**
     * Initialises the internal map with a buffer for each vertex and edge property
     * @param description The EnvironmentDirectedGraphData to initialise the CUDABuffers for
     */
    explicit CUDAEnvironmentDirectedGraphBuffers(const EnvironmentDirectedGraphData &description);
    /**
     * Destructor
     * Frees any allocated CUDA buffers
     */
    ~CUDAEnvironmentDirectedGraphBuffers();
    /**
     * Called internally by the constructor to register curve instances to be notified of allocations
     * This should also be called when submodels are initialising
     */
    void registerCurveInstance(const std::shared_ptr<detail::curve::HostCurve>& curve);
    void registerCurveInstance(const std::shared_ptr<detail::curve::CurveRTCHost>& curve);
    /**
     * Return the description of the represented graph
     */
    const EnvironmentDirectedGraphData& getDescription() const { return graph_description; }
    /**
     * Allocates and initialises the vertex buffers
     * @param count The number of vertices to allocate each buffer for
     * @param stream CUDA stream to be used if data must be copied back from device
     */
    void setVertexCount(size_type count, cudaStream_t stream);
    /**
     * Allocates and initialises the vertex buffers
     * @param count The number of edges to allocate each buffer for
     */
    void setEdgeCount(size_type count);
    /**
     * Returns the number of vertices the graph is currently allocated to hold
     */
    size_type getVertexCount() const { return vertex_count; }
    /**
     * Returns the number of edges the graph is currently allocated to hold
     */
    size_type getEdgeCount() const { return edge_count; }
    /**
     * Attempt to assign the provided vertex_id with an index inside h_vertex_index_map
     * @param vertex_id The ID of the vertex to be created
     * @param stream The CUDA stream to use if the ID buffer requires sync
     * @return The index of the vertex with the given ID
     * @throw exception::OutOfBoundsException If the number of vertices would exceed the vertex count configured via setVertexCount()
     */
    unsigned int createIfNotExistVertex(id_t vertex_id, cudaStream_t stream);
    /**
     * Attempt to assign the provided source_vertex_id and dest_vertex_id with an index inside h_vertex_index_map
     * @param source_vertex_id The ID of the source vertex of the edge to be created
     * @param dest_vertex_id The ID of the destination vertex of the edge to be created
     * @param stream The CUDA stream to use if the ID buffer requires sync
     * @return The index of the edge with the given src and dest vertex IDs
     * @throw exception::OutOfBoundsException If the number of edges would exceed the edge count configured via setEdgeCount()
     */
    unsigned int createIfNotExistEdge(id_t source_vertex_id, id_t dest_vertex_id, cudaStream_t stream);
    /**
     * Returns the currently allocated device buffer for vertex IDs
     * @throw exception::OutOfBoundsException If the vertex buffers have not been allocated yet
     */
    id_t* getVertexIDBuffer(cudaStream_t stream);
    /**
     * Returns the currently allocated device buffer for the specified vertex property
     * @param property_name The name of the property to set
     * @param N The number of elements in the property (1 if not a property array)
     * @param stream CUDA stream to be used if data must be copied back from device
     * @tparam T The type of the property
     * @throw exception::InvalidArgument If property_name does not refer to a valid vertex property
     * @throw exception::InvalidGraphProperty If an vertex property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If the vertex buffers have not been allocated yet
     * @note If N is passed as 0, it will instead be set to the number of elements
     */
    template<typename T>
    const T* getVertexPropertyBuffer(const std::string& property_name, size_type &N, cudaStream_t stream) const;
    /**
     * Returns the currently allocated device buffer for the specified vertex property
     * Additionally marks the device buffer as out of date, use the const version if you do not wish to change the buffer
     * @param property_name The name of the property to set
     * @param N The number of elements in the property (1 if not a property array)
     * @param stream CUDA stream to be used if data must be copied back from device
     * @tparam T The type of the property
     * @throw exception::InvalidArgument If property_name does not refer to a valid vertex property
     * @throw exception::InvalidGraphProperty If an vertex property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If the vertex buffers have not been allocated yet
     * @note If N is passed as 0, it will instead be set to the number of elements
     */
    template<typename T>
    T* getVertexPropertyBuffer(const std::string& property_name, size_type &N, cudaStream_t stream);
    /**
     * Returns the currently allocated device buffer for the specified edge property
     * @param property_name The name of the property to set
     * @param N The number of elements in the property (1 if not a property array)
     * @param stream CUDA stream to be used if data must be copied back from device
     * @tparam T The type of the property
     * @throw exception::InvalidArgument If property_name does not refer to a valid edge property
     * @throw exception::InvalidGraphProperty If an edge property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If the vertex buffers have not been allocated yet
     * @note If N is passed as 0, it will instead be set to the number of elements
     */
    template<typename T>
    const T *getEdgePropertyBuffer(const std::string &property_name, size_type &N, cudaStream_t stream) const;
    /**
     * Returns the currently allocated device buffer for the specified edge property
     * Additionally marks the device buffer as out of date, use the const version if you do not wish to change the buffer
     * @param property_name The name of the property to set
     * @param N The number of elements in the property (1 if not a property array)
     * @param stream CUDA stream to be used if data must be copied back from device
     * @tparam T The type of the property
     * @throw exception::InvalidArgument If property_name does not refer to a valid edge property
     * @throw exception::InvalidGraphProperty If an edge property with the matching name and type does not exist
     * @throw exception::OutOfBoundsException If the vertex buffers have not been allocated yet
     * @note If N is passed as 0, it will instead be set to the number of elements
     */
    template<typename T>
    T* getEdgePropertyBuffer(const std::string& property_name, size_type &N, cudaStream_t stream);
#ifdef SWIG
    template<typename T>
    std::vector<T> getVertexPropertyArray(const std::string& property_name, const unsigned int vertex_index, cudaStream_t stream) const;
    template<typename T>
    std::vector<T> getEdgePropertyArray(const std::string& property_name, const unsigned int edge_index, cudaStream_t stream) const;
#endif
    void markForRebuild() { requires_rebuild = true; }
    /**
     * Update any device buffers which don't currently match the host
     * Rebuild the internal graph CSR, sort edge buffers
     * @param scatter CUDAScatter singleton instance
     * @param streamID Stream index corresponding to stream resources to use
     * @param stream The cuda stream to perform CUDA operations on
     */
    void syncDevice_async(detail::CUDAScatter& scatter, unsigned int streamID, cudaStream_t stream);
    /**
     * Updates the vertex ID buffer
     * Updates the internal host map of ID->index
     * Updates the internal tracking of th vertex ID min/max
     *
     * @param vertex_index The index of the vertex
     * @param vertex_id The ID that has been assigned to the vertex
     * @param stream The cuda stream to perform CUDA operations on
     *
     * @note This affects how much memory is allocates for the vertex id -> index map
     * @throws exception::IDCollision If the ID is already assigned to a different vertex
     */
    void setVertexID(unsigned int vertex_index, id_t vertex_id, cudaStream_t stream);
    /**
     * Returns the index of the vertex with the given ID
     * @param vertex_id The ID of the vertex of which to return the index
     *
     * @throws exception::InvalidID If the ID is not in use
     */
    unsigned int getVertexIndex(id_t vertex_id) const;
    /**
    * Updates the edge ID buffer
    * Updates the internal host map of src:dest->index
    *
    * @param edge_index The index of the edge
    * @param src_vertex_id The ID that has been assigned to the source vertex of the edge
    * @param dest_vertex_id The ID that has been assigned to the destination vertex of the edge
    *
    * @throws exception::IDCollision If the ID is already assigned to a different vertex
    */
    void setEdgeSourceDestination(unsigned int edge_index, id_t src_vertex_id, id_t dest_vertex_id);
    /**
    * Returns the index of the edge with the given source and destination vertices
    * @param src_vertex_id The ID that has been assigned to the source vertex of the edge
    * @param dest_vertex_id The ID that has been assigned to the destination vertex of the edge
    *
    * @throws exception::InvalidID If the ID is not in use
    */
    unsigned int getEdgeIndex(id_t src_vertex_id, id_t dest_vertex_id) const;
};


template<typename T>
const T* CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer(const std::string& property_name, size_type &N, cudaStream_t stream) const {
    if (!vertex_count) {
        THROW exception::OutOfBoundsException("Vertex buffers not yet allocated, in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()");
    }
    const auto &vv = graph_description.vertexProperties.find(property_name);
    if (vv == graph_description.vertexProperties.end()) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' not found, in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()", property_name.c_str());
    } else if (vv->second.type != std::type_index(typeid(typename detail::type_decode<T>::type_t))) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' type mismatch '%s' != '%s', in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()",
        property_name.c_str(), vv->second.type.name(), std::type_index(typeid(typename detail::type_decode<T>::type_t)).name());
    } else if (N == 0) {
        N = vv->second.elements;
    } else if (vv->second.elements != N) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' length mismatch '%u' != '%u', in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()",
            property_name.c_str(), N, vv->second.elements);
    }
    auto& vb = vertex_buffers.at(property_name);
    vb.updateHostBuffer(vertex_count, stream);
    return static_cast<T*>(vb.h_ptr);
}
template<typename T>
T* CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer(const std::string& property_name, size_type &N, cudaStream_t stream) {
    if (!vertex_count) {
        THROW exception::OutOfBoundsException("Vertex buffers not yet allocated, in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()");
    }
    const auto& vv = graph_description.vertexProperties.find(property_name);
    if (vv == graph_description.vertexProperties.end()) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' not found, in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()", property_name.c_str());
    } else if (vv->second.type != std::type_index(typeid(typename detail::type_decode<T>::type_t))) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' type mismatch '%s' != '%s', in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()",
            property_name.c_str(), vv->second.type.name(), std::type_index(typeid(typename detail::type_decode<T>::type_t)).name());
    } else if (N == 0) {
        N = vv->second.elements;
    } else if (vv->second.elements != N) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' length mismatch '%u' != '%u', in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyBuffer()",
            property_name.c_str(), N, vv->second.elements);
    }
    auto &vb = vertex_buffers.at(property_name);
    vb.updateHostBuffer(vertex_count, stream);
    vb.ready = Buffer::Host;
    return static_cast<T*>(vb.h_ptr);
}
template<typename T>
const T* CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer(const std::string& property_name, size_type &N, cudaStream_t stream) const {
    if (!edge_count) {
        THROW exception::OutOfBoundsException("Edge buffers not yet allocated, in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()");
    }
    const auto& ev = graph_description.edgeProperties.find(property_name);
    if (ev == graph_description.edgeProperties.end()) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' not found, in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()", property_name.c_str());
    } else if (ev->second.type != std::type_index(typeid(typename detail::type_decode<T>::type_t))) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' type mismatch '%s' != '%s', in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()",
            property_name.c_str(), ev->second.type.name(), std::type_index(typeid(typename detail::type_decode<T>::type_t)).name());
    } else if (N == 0) {
        N = ev->second.elements;
    } else if (ev->second.elements != N) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' length mismatch '%u' != '%u', in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()",
            property_name.c_str(), N, ev->second.elements);
    }
    auto &eb = edge_buffers.at(property_name);
    eb.updateHostBuffer(edge_count, stream);
    return static_cast<T*>(eb.h_ptr);
}
template<typename T>
T* CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer(const std::string& property_name, size_type &N, cudaStream_t stream) {
    if (!edge_count) {
        THROW exception::OutOfBoundsException("Edge buffers not yet allocated, in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()");
    }
    const auto& ev = graph_description.edgeProperties.find(property_name);
    if (ev == graph_description.edgeProperties.end()) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' not found, in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()", property_name.c_str());
    } else if (ev->second.type != std::type_index(typeid(typename detail::type_decode<T>::type_t))) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' type mismatch '%s' != '%s', in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()",
            property_name.c_str(), ev->second.type.name(), std::type_index(typeid(typename detail::type_decode<T>::type_t)).name());
    } else if (N == 0) {
        N = ev->second.elements;
    } else if (ev->second.elements != N) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' length mismatch '%u' != '%u', in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyBuffer()",
            property_name.c_str(), N, ev->second.elements);
    }
    auto &eb = edge_buffers.at(property_name);
    eb.updateHostBuffer(edge_count, stream);
    eb.ready = Buffer::Host;
    return static_cast<T*>(eb.h_ptr);
}
#ifdef SWIG
template<typename T>
std::vector <T> CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyArray(const std::string& property_name, const unsigned int vertex_index, cudaStream_t stream) const {
    if (!vertex_count) {
        THROW exception::OutOfBoundsException("Vertex buffers not yet allocated, in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyArray()");
    }
    const auto& vv = graph_description.vertexProperties.find(property_name);
    if (vv == graph_description.vertexProperties.end()) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' not found, in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyArray()", property_name.c_str());
    } else if (vv->second.type != std::type_index(typeid(typename detail::type_decode<T>::type_t))) {
        THROW exception::InvalidGraphProperty("Vertex property with name '%s' type mismatch '%s' != '%s', in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyArray()",
            property_name.c_str(), vv->second.type.name(), std::type_index(typeid(typename detail::type_decode<T>::type_t)).name());
    } else if (vertex_index >= vertex_count) {
        THROW exception::OutOfBoundsException("Vertex index %u is out of range %u, in CUDAEnvironmentDirectedGraphBuffers::getVertexPropertyArray()\n",
            vertex_index, vertex_count);
    }
    std::vector<T> rtn(vv->second.elements);
    auto& vb = vertex_buffers.at(property_name);
    vb.updateHostBuffer(vertex_count, stream);
    memcpy(rtn.data(), static_cast<T*>(vb.h_ptr) + vertex_index * vv->second.elements, vv->second.type_size * vv->second.elements);
    return rtn;
}
template<typename T>
std::vector<T> CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyArray(const std::string& property_name, const unsigned int edge_index, cudaStream_t stream) const {
    if (!edge_count) {
        THROW exception::OutOfBoundsException("Edge buffers not yet allocated, in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyArray()");
    }
    const auto& ev = graph_description.edgeProperties.find(property_name);
    if (ev == graph_description.edgeProperties.end()) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' not found, in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyArray()", property_name.c_str());
    }  else if (ev->second.type != std::type_index(typeid(typename detail::type_decode<T>::type_t))) {
        THROW exception::InvalidGraphProperty("Edge property with name '%s' type mismatch '%s' != '%s', in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyArray()",
            property_name.c_str(), ev->second.type.name(), std::type_index(typeid(typename detail::type_decode<T>::type_t)).name());
    } else if (edge_index >= edge_count) {
        THROW exception::OutOfBoundsException("Edge index %u is out of range %u, in CUDAEnvironmentDirectedGraphBuffers::getEdgePropertyArray()\n",
            edge_index, edge_count);
    }
    std::vector<T> rtn(ev->second.elements);
    auto& eb = edge_buffers.at(property_name);
    eb.updateHostBuffer(edge_count, stream);
    memcpy(rtn.data(), static_cast<T*>(eb.h_ptr) + edge_index * ev->second.elements, ev->second.type_size * ev->second.elements);
    return rtn;
}
#endif

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAENVIRONMENTDIRECTEDGRAPHBUFFERS_CUH_
