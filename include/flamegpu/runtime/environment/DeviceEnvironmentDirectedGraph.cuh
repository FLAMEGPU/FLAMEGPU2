#ifndef INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_DEVICEENVIRONMENTDIRECTEDGRAPH_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_DEVICEENVIRONMENTDIRECTEDGRAPH_CUH_

#include "flamegpu/defines.h"
#include "flamegpu/runtime/detail/curve/Curve.cuh"

namespace flamegpu {
/**
 * Read-only accessor to a specific directed graph
 * Directed graphs can only be modified via host functions
 */
class DeviceEnvironmentDirectedGraph {
    const detail::curve::Curve::VariableHash graph_hash;

 public:
    /**
    * This class is created when a vertex is provided to DeviceEnvironmentDirectedGraph::edgesLeaving()(id_t)
    * It provides iterator access to the subset of edges found leaving the specified vertex
    *
    * @see DeviceEnvironmentDirectedGraph::outEdges()(id_t)
    */
    class OutEdgeFilter {
        /**
        * Edge has full access to Filter, they are treated as the same class so share everything
        * Reduces/memory data duplication
        */
        friend class Message;

     public:
        /**
        * Provides access to a specific edge
        * Returned by the iterator
        * @see In::Filter::iterator
        */
        class Edge {
            /**
             * Paired Filter class which created the iterator
             */
            const OutEdgeFilter&_parent;
            /**
             * This is the index of the currently accessed edge, relative to the full edge list
             */
            unsigned int edge_index;

         public:
            /**
             * Constructs a message and directly initialises all of it's member variables
             * @note See member variable documentation for their purposes
             */
            __device__ Edge(const OutEdgeFilter&parent, const unsigned int&_edge_index)
                : _parent(parent)
                , edge_index(_edge_index) { }
            /**
             * Equality operator
             * Compares all internal member vars for equality
             * @note Does not compare _parent
             */
            __device__ bool operator==(const Edge& rhs) const {
                return this->edge_index == rhs.edge_index;
            }
            /**
             * Inequality operator
             * Returns inverse of equality operator
             * @see operator==(const Message&)
             */
            __device__ bool operator!=(const Edge& rhs) const { return !(*this == rhs); }
            /**
             * Updates the message to return variables from the next message in the message list
             * @return Returns itself
             */
            __device__ Edge& operator++() { ++edge_index; return *this; }
            /**
             * Returns the index of the current edge within the graph's full list of edges
             * @note Edge indexes are not constant, they can change if edges are updated causing the graph's CSR to be rebuilt
             */
            __device__ unsigned int getIndex() const { return edge_index; }
            /**
             * Returns the value for the current edge attached to the named property
             * @param property_name Name of the property
             * @tparam T type of the property
             * @tparam N Length of property name (this should be implicit if a string literal is passed to property_name)
             * @return The specified variable, else 0x0 if an error occurs
             */
            template<typename T, flamegpu::size_type N>
            __device__ T getProperty(const char(&property_name)[N]) const;
            /**
             * Returns the specified property array element from the current edge attached to the named property
             * @param property_name name used for accessing the property, this value should be a string literal e.g. "foobar"
             * @param index Index of the element within the property array to return
             * @tparam T Type of the edge property being accessed
             * @tparam N The length of the edge's array property, as set within the model description hierarchy
             * @tparam M Length of property name (this should be implicit if a string literal is passed to property_name)
             * @throws exception::DeviceError If name is not a valid property within the edge (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If T is not the type of property 'name' within the message (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If index is out of bounds for the property array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
             */
            template<typename T, flamegpu::size_type N, unsigned int M>
            __device__ T getProperty(const char(&property_name)[M], const unsigned int &index) const;
            /**
             * Returns the destination vertex index of this edge
             * @note getEdgeSource() is not provided, as this value is required to access an edge via iteration so should already be available
             */
            __device__ unsigned int getEdgeDestination() const;
        };
        /**
         * Stock iterator for iterating DeviceEnvironmentDirectedGraph::Edge objects
         */
        class iterator {
            /**
             * The edge returned to the user
             */
            Edge _edge;

         public:
            /**
             * Constructor
             * This iterator is constructed by DeviceEnvironmentDirectedGraph::OutEdgeFilter::begin()(id_t)
             * @see DeviceEnvironmentDirectedGraph::edgesLeaving()(id_t)
             */
            __device__ iterator(const OutEdgeFilter &parent, const unsigned int &cell_index)
                : _edge(parent, cell_index) {
                // Increment to find first edge
                ++_edge;
            }
            /**
             * Moves to the next edge
             * (Prefix increment operator)
             */
            __device__ iterator& operator++() { ++_edge;  return *this; }
            /**
             * Moves to the next edge
             * (Postfix increment operator, returns value prior to increment)
             */
            __device__ iterator operator++(int) {
                iterator temp = *this;
                ++*this;
                return temp;
            }
            /**
             * Equality operator
             * Compares edge
             */
            __device__ bool operator==(const iterator& rhs) const { return  _edge == rhs._edge; }
            /**
             * Inequality operator
             * Compares edge
             */
            __device__ bool operator!=(const iterator& rhs) const { return  _edge != rhs._edge; }
            /**
             * Dereferences the iterator to return the edge object, for accessing variables
             */
            __device__ Edge& operator*() { return _edge; }
            /**
             * Dereferences the iterator to return the edge object, for accessing variables
             */
            __device__ Edge* operator->() { return &_edge; }
        };
        /**
         * Constructor, takes the search parameters required
         * Begin key and end key specify the [begin, end) contiguous range of bucket. (inclusive begin, exclusive end)
         * @param _graph_hash Graph hash for accessing data via curve
         * @param vertex_index The index of the vertex to retrieve leaving edges for
         */
        inline __device__ OutEdgeFilter(detail::curve::Curve::VariableHash _graph_hash, const id_t &vertex_index);
        /**
         * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
            // Bin before initial bin, as the constructor calls increment operator
            return iterator(*this, bucket_begin - 1);
        }
        /**
         * Returns an iterator to the position beyond the end of the edge list subset
         * @note This iterator is the same for all edge list subsets
         */
        inline __device__ iterator end(void) const {
            // Final bin, as the constructor calls increment operator
            return iterator(*this, bucket_end - 1);
        }
        /**
         * Returns the number of edges in the filtered bucket
         */
        inline __device__ unsigned int size(void) const {
            return bucket_end - bucket_begin;
        }

     private:
        /**
         * Search bucket bounds
         */
        unsigned int bucket_begin, bucket_end;
        /**
         * Graph hash for accessing graph data
         */
        const detail::curve::Curve::VariableHash graph_hash;
    };
    /**
     * This class is created when a vertex is provided to DeviceEnvironmentDirectedGraph::edgesLeaving()(id_t)
     * It provides iterator access to the subset of edges found leaving the specified vertex
     *
     * @see DeviceEnvironmentDirectedGraph::inEdges()(id_t)
     */
    class InEdgeFilter {
        /**
        * Edge has full access to Filter, they are treated as the same class so share everything
        * Reduces/memory data duplication
        */
        friend class Message;

     public:
        /**
        * Provides access to a specific edge
        * Returned by the iterator
        * @see In::Filter::iterator
        */
        class Edge {
            /**
             * Paired Filter class which created the iterator
             */
            const InEdgeFilter&_parent;
            /**
             * This is the index of the currently accessed IPBM element, relative to the full PBM
             * Incoming edges are not sorted, so must be accessed indirectly
             */
            unsigned int ipbm_index;
            /**
             * This is the index of the currently accessed edge, relative to the full edge list
             */
            unsigned int edge_index;

         public:
            /**
             * Constructs a message and directly initialises all of it's member variables
             * @note See member variable documentation for their purposes
             */
            __device__ Edge(const InEdgeFilter&parent, const unsigned int&_ipbm_index)
                : _parent(parent)
                , ipbm_index(_ipbm_index)
                //, edge_index(0)  // Value doesn't matter until operator++() is first called
            { }
            /**
             * Equality operator
             * Compares all internal member vars for equality
             * @note Does not compare _parent
             */
            __device__ bool operator==(const Edge& rhs) const {
                return this->ipbm_index == rhs.ipbm_index;
            }
            /**
             * Inequality operator
             * Returns inverse of equality operator
             * @see operator==(const Message&)
             */
            __device__ bool operator!=(const Edge& rhs) const { return !(*this == rhs); }
            /**
             * Updates the message to return variables from the next edge in the message list
             * @return Returns itself
             */
            __device__ Edge& operator++() {
                // @todo This may not exit safely it SEATBELTS has left graph_ipbm_edges=nullptr?
                edge_index = _parent.graph_ipbm_edges[++ipbm_index];
                return *this; }
            /**
             * Returns the index of the current edge within the graph's full list of edges
             * @note Edge indexes are not constant, they can change if edges are updated causing the graph's CSR to be rebuilt
             */
            __device__ unsigned int getIndex() const { return edge_index; }
            /**
             * Returns the value for the current edge attached to the named property
             * @param property_name Name of the property
             * @tparam T type of the property
             * @tparam N Length of property name (this should be implicit if a string literal is passed to property_name)
             * @return The specified variable, else 0x0 if an error occurs
             */
            template<typename T, flamegpu::size_type N>
            __device__ T getProperty(const char(&property_name)[N]) const;
            /**
             * Returns the specified property array element from the current edge attached to the named property
             * @param property_name name used for accessing the property, this value should be a string literal e.g. "foobar"
             * @param index Index of the element within the property array to return
             * @tparam T Type of the edge property being accessed
             * @tparam N The length of the edge's array property, as set within the model description hierarchy
             * @tparam M Length of property name (this should be implicit if a string literal is passed to property_name)
             * @throws exception::DeviceError If name is not a valid property within the edge (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If T is not the type of property 'name' within the message (flamegpu must be built with SEATBELTS enabled for device error checking)
             * @throws exception::DeviceError If index is out of bounds for the property array specified by name (flamegpu must be built with SEATBELTS enabled for device error checking)
             */
            template<typename T, flamegpu::size_type N, unsigned int M>
            __device__ T getProperty(const char(&property_name)[M], const unsigned int &index) const;
            /**
             * Returns the source vertex indexof this edge
             * @note getEdgeDestination() is not provided, as this value is required to access an edge via iteration so should already be available
             */
            __device__ unsigned int getEdgeSource() const;
        };
        /**
         * Stock iterator for iterating DeviceEnvironmentDirectedGraph::Edge objects
         */
        class iterator {
            /**
             * The edge returned to the user
             */
            Edge _edge;

         public:
            /**
             * Constructor
             * This iterator is constructed by DeviceEnvironmentDirectedGraph::InEdgeFilter::begin()(id_t)
             * @see DeviceEnvironmentDirectedGraph::edgesJoining()(id_t)
             */
            __device__ iterator(const InEdgeFilter &parent, const unsigned int &cell_index)
                : _edge(parent, cell_index) {
                // Increment to find first edge
                ++_edge;
            }
            /**
             * Moves to the next edge
             * (Prefix increment operator)
             */
            __device__ iterator& operator++() { ++_edge;  return *this; }
            /**
             * Moves to the next edge
             * (Postfix increment operator, returns value prior to increment)
             */
            __device__ iterator operator++(int) {
                iterator temp = *this;
                ++*this;
                return temp;
            }
            /**
             * Equality operator
             * Compares edge
             */
            __device__ bool operator==(const iterator& rhs) const { return  _edge == rhs._edge; }
            /**
             * Inequality operator
             * Compares edge
             */
            __device__ bool operator!=(const iterator& rhs) const { return  _edge != rhs._edge; }
            /**
             * Dereferences the iterator to return the edge object, for accessing variables
             */
            __device__ Edge& operator*() { return _edge; }
            /**
             * Dereferences the iterator to return the edge object, for accessing variables
             */
            __device__ Edge* operator->() { return &_edge; }
        };
        /**
         * Constructor, takes the search parameters required
         * Begin key and end key specify the [begin, end) contiguous range of bucket. (inclusive begin, exclusive end)
         * @param _graph_hash Graph hash for accessing data via curve
         * @param vertex_index The index of the vertex to retrieve leaving edges for
        */
        inline __device__ InEdgeFilter(detail::curve::Curve::VariableHash _graph_hash, const id_t &vertex_index);
        /**
       * Returns an iterator to the start of the message list subset about the search origin
         */
        inline __device__ iterator begin(void) const {
            // Bin before initial bin, as the constructor calls increment operator
            return iterator(*this, bucket_begin - 1);
        }
        /**
         * Returns an iterator to the position beyond the end of the edge list subset
         * @note This iterator is the same for all edge list subsets
         */
        inline __device__ iterator end(void) const {
            // Final bin, as the constructor calls increment operator
            return iterator(*this, bucket_end - 1);
        }
        /**
         * Returns the number of edges in the filtered bucket
         */
        inline __device__ unsigned int size(void) const {
            return bucket_end - bucket_begin;
        }

     private:
        /**
         * Search bucket bounds
         */
        unsigned int bucket_begin, bucket_end;
        /**
         * IPBM edge list
         */
        unsigned int *graph_ipbm_edges;
        /**
         * Graph hash for accessing graph data
         */
        const detail::curve::Curve::VariableHash graph_hash;
    };
    __device__ __forceinline__ DeviceEnvironmentDirectedGraph(const detail::curve::Curve::VariableHash _graph_hash)
        : graph_hash(_graph_hash)
    { }

    __device__ __forceinline__ id_t getVertexID(unsigned int vertex_index) const;
    __device__ __forceinline__ unsigned int getVertexIndex(id_t vertex_id) const;
    template<typename T, unsigned int M>
    __device__ __forceinline__ T getVertexProperty(const char(&property_name)[M], unsigned int vertex_index) const;
    template<typename T, flamegpu::size_type N, unsigned int M>
    __device__ __forceinline__ T getVertexProperty(const char(&property_name)[M], unsigned int vertex_index, unsigned int element_index) const;

    __device__ __forceinline__ id_t getEdgeSource(unsigned int edge_index) const;
    __device__ __forceinline__ id_t getEdgeDestination(unsigned int edge_index) const;
    __device__ __forceinline__ unsigned int getEdgeIndex(unsigned int source_vertex_index, unsigned int destination_vertex_index) const;
    template<typename T, unsigned int M>
    __device__ __forceinline__ T getEdgeProperty(const char(&property_name)[M], unsigned int edge_index) const;
    template<typename T, flamegpu::size_type N, unsigned int M>
    __device__ __forceinline__ T getEdgeProperty(const char(&property_name)[M], unsigned int edge_index, unsigned int element_index) const;

    /**
     * Returns a Filter object which provides access to an edge iterator
     * for iterating a subset of edge which leave the specified vertex
     *
     * @param vertex_index The index of the vertex to retrieve leaving edges for
     */
    inline __device__ OutEdgeFilter outEdges(const id_t & vertex_index) const {
        return OutEdgeFilter(graph_hash, vertex_index);
    }
    /**
     * Returns a Filter object which provides access to an edge iterator
     * for iterating a subset of edge which join the specified vertex
     *
     * @param vertex_index The index of the vertex to retrieve joining edges for
     */
    inline __device__ InEdgeFilter inEdges(const id_t& vertex_index) const {
        return InEdgeFilter(graph_hash, vertex_index);
    }
};
__device__ DeviceEnvironmentDirectedGraph::OutEdgeFilter::OutEdgeFilter(const detail::curve::Curve::VariableHash _graph_hash, const id_t& vertex_index)
    : bucket_begin(0)
    , bucket_end(0)
    , graph_hash(_graph_hash) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Vertex "_id" always exists
    const unsigned int VERTEX_COUNT = detail::curve::DeviceCurve::getVariableCount("_id", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_vertex"));
    if (vertex_index >= VERTEX_COUNT) {
        DTHROW("Vertex index (%u) exceeds vertex count (%u), unable to iterate outgoing edges.\n", vertex_index, VERTEX_COUNT);
        return;
    }
#endif
    unsigned int* pbm = detail::curve::DeviceCurve::getEnvironmentDirectedGraphPBM(graph_hash);
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (!pbm) return;
#endif
    bucket_begin = pbm[vertex_index];
    bucket_end = pbm[vertex_index + 1];
}
__device__ DeviceEnvironmentDirectedGraph::InEdgeFilter::InEdgeFilter(const detail::curve::Curve::VariableHash _graph_hash, const id_t& vertex_index)
    : bucket_begin(0)
    , bucket_end(0)
    , graph_ipbm_edges(nullptr)
    , graph_hash(_graph_hash) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    // Vertex "_id" always exists
    const unsigned int VERTEX_COUNT = detail::curve::DeviceCurve::getVariableCount("_id", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_vertex"));
    if (vertex_index >= VERTEX_COUNT) {
        DTHROW("Vertex index (%u) exceeds vertex count (%u), unable to iterate incoming edges.\n", vertex_index, VERTEX_COUNT);
        return;
    }
#endif
    unsigned int* ipbm = detail::curve::DeviceCurve::getEnvironmentDirectedGraphIPBM(graph_hash);
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (!ipbm) return;
#endif
    bucket_begin = ipbm[vertex_index];
    bucket_end = ipbm[vertex_index + 1];
    // Grab and store a copy of the PBM edgelist pointer
    this->graph_ipbm_edges = detail::curve::DeviceCurve::getEnvironmentDirectedGraphIPBMEdges(_graph_hash);
}

template<typename T, unsigned int N>
__device__ T DeviceEnvironmentDirectedGraph::OutEdgeFilter::Edge::getProperty(const char(&property_name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (edge_index >= _parent.bucket_end) {
        DTHROW("Edge index exceeds bin length, unable to get property '%s'.\n", property_name);
        return {};
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getEnvironmentDirectedGraphEdgeProperty<T>(_parent.graph_hash, property_name, edge_index);
    return value;
}
template<typename T, flamegpu::size_type N, unsigned int M> __device__
T DeviceEnvironmentDirectedGraph::OutEdgeFilter::Edge::getProperty(const char(&property_name)[M], const unsigned int& element_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (edge_index >= _parent.bucket_end) {
        DTHROW("Edge index exceeds bin length, unable to get property '%s'.\n", property_name);
        return {};
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getEnvironmentDirectedGraphEdgeArrayProperty<T, N>(_parent.graph_hash, property_name, edge_index, element_index);
    return value;
}
template<typename T, unsigned int N>
__device__ T DeviceEnvironmentDirectedGraph::InEdgeFilter::Edge::getProperty(const char(&property_name)[N]) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int EDGE_COUNT = detail::curve::DeviceCurve::getVariableCount("_source_dest", _parent.graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_edge"));
    if (edge_index >= EDGE_COUNT) {
        DTHROW("Edge index exceeds edge count, unable to get property '%s'.\n", property_name);
        return {};
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getEnvironmentDirectedGraphEdgeProperty<T>(_parent.graph_hash, property_name, edge_index);
    return value;
}
template<typename T, flamegpu::size_type N, unsigned int M> __device__
T DeviceEnvironmentDirectedGraph::InEdgeFilter::Edge::getProperty(const char(&property_name)[M], const unsigned int& element_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int EDGE_COUNT = detail::curve::DeviceCurve::getVariableCount("_source_dest", _parent.graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_edge"));
    if (edge_index >= EDGE_COUNT) {
        DTHROW("Edge index exceeds edge count, unable to get property '%s'.\n", property_name);
        return {};
    }
#endif
    // get the value from curve using the message index.
    T value = detail::curve::DeviceCurve::getEnvironmentDirectedGraphEdgeArrayProperty<T, N>(_parent.graph_hash, property_name, edge_index, element_index);
    return value;
}
__device__ __forceinline__ unsigned int DeviceEnvironmentDirectedGraph::OutEdgeFilter::Edge::getEdgeDestination() const {
    return getProperty<id_t, 2>("_source_dest", 0);
}
__device__ __forceinline__ unsigned int DeviceEnvironmentDirectedGraph::InEdgeFilter::Edge::getEdgeSource() const {
    return getProperty<id_t, 2>("_source_dest", 1);
}
__device__ __forceinline__ id_t DeviceEnvironmentDirectedGraph::getVertexID(const unsigned int vertex_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int VERTEX_COUNT = detail::curve::DeviceCurve::getVariableCount("_id", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_vertex"));
    if (vertex_index >= VERTEX_COUNT) {
        DTHROW("Vertex index (%u) exceeds vertex count (%u), unable to get vertex ID.\n", vertex_index, VERTEX_COUNT);
        return {};
    }
#endif
    return getVertexProperty<id_t>("_id", vertex_index);
}

__device__ __forceinline__ unsigned int DeviceEnvironmentDirectedGraph::getVertexIndex(id_t vertex_id) const {
    const unsigned int VERTEX_RANGE = detail::curve::DeviceCurve::getVariableCount("_index_map", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_vertex")) - 1;  // -1 because offset is packed at the end
    const unsigned int VERTEX_OFFSET = detail::curve::DeviceCurve::getEnvironmentDirectedGraphVertexProperty<unsigned int>(graph_hash, "_index_map", VERTEX_RANGE);
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (vertex_id < VERTEX_OFFSET ||  vertex_id - VERTEX_OFFSET >= VERTEX_RANGE) {
        DTHROW("Vertex ID (%u) exceeds vertex range bounds (%u <= ID <= %u), unable to get vertex index.\n", vertex_id, VERTEX_OFFSET, VERTEX_OFFSET + VERTEX_RANGE - 1);
        return {};
    }
    const unsigned int index = getVertexProperty<unsigned int>("_index_map", vertex_id - VERTEX_OFFSET);
    if (index == 0xffffffff) {
        DTHROW("Vertex ID %u is not in use, unable to get vertex index.\n", vertex_id);
        return {};
    }
    return index;
#else
    return getVertexProperty<unsigned int>("_index_map", vertex_id - VERTEX_OFFSET);
#endif
}
template<typename T, unsigned int M>
__device__ __forceinline__ T DeviceEnvironmentDirectedGraph::getVertexProperty(const char(&property_name)[M], const unsigned int vertex_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int VERTEX_COUNT = detail::curve::DeviceCurve::getVariableCount(property_name, graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_vertex"));
    if (vertex_index >= VERTEX_COUNT) {
        DTHROW("Vertex index (%u) exceeds vertex count (%u), unable to get property '%s'.\n", vertex_index, VERTEX_COUNT, property_name);
        return {};
    }
#endif
    return detail::curve::DeviceCurve::getEnvironmentDirectedGraphVertexProperty<T>(graph_hash, property_name, vertex_index);
}
template<typename T, flamegpu::size_type N, unsigned int M>
__device__ __forceinline__ T DeviceEnvironmentDirectedGraph::getVertexProperty(const char(&property_name)[M], const unsigned int vertex_index, const unsigned int element_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int VERTEX_COUNT = detail::curve::DeviceCurve::getVariableCount(property_name, graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_vertex"));
    if (vertex_index >= VERTEX_COUNT) {
        DTHROW("Vertex index (%u) exceeds vertex count (%u), unable to get property '%s'.\n", vertex_index, VERTEX_COUNT, property_name);
        return {};
    }
#endif
    return detail::curve::DeviceCurve::getEnvironmentDirectedGraphVertexArrayProperty<T, N>(graph_hash, property_name, vertex_index, element_index);
}

__device__ __forceinline__ id_t DeviceEnvironmentDirectedGraph::getEdgeSource(const unsigned int edge_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int EDGE_COUNT = detail::curve::DeviceCurve::getVariableCount("_source_dest", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_edge"));
    if (edge_index >= EDGE_COUNT) {
        DTHROW("Edge index (%u) exceeds edge count (%u), unable to get edge source vertex.\n", edge_index, EDGE_COUNT);
        return {};
    }
#endif
    return getEdgeProperty<id_t, 2>("_source_dest", edge_index, 1);
}
__device__ __forceinline__ id_t DeviceEnvironmentDirectedGraph::getEdgeDestination(const unsigned int edge_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int EDGE_COUNT = detail::curve::DeviceCurve::getVariableCount("_source_dest", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_edge"));
    if (edge_index >= EDGE_COUNT) {
        DTHROW("Edge index (%u) exceeds edge count (%u), unable to get edge destination vertex.\n", edge_index, EDGE_COUNT);
        return {};
    }
#endif
    return getEdgeProperty<id_t, 2>("_source_dest", edge_index, 0);
}

__device__ __forceinline__ unsigned int DeviceEnvironmentDirectedGraph::getEdgeIndex(unsigned int source_vertex_index, unsigned int destination_vertex_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int VERTEX_COUNT = detail::curve::DeviceCurve::getVariableCount("_id", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_vertex"));
    if (source_vertex_index >= VERTEX_COUNT) {
        DTHROW("Source vertex index (%u) exceeds vertex count (%u), unable to get edge index.\n", source_vertex_index, VERTEX_COUNT);
        return {};
    } else if (destination_vertex_index >= VERTEX_COUNT) {
        DTHROW("Destination vertex index (%u) exceeds vertex count (%u), unable to get edge index.\n", destination_vertex_index, VERTEX_COUNT);
        return {};
    }
#endif
    unsigned int* pbm = detail::curve::DeviceCurve::getEnvironmentDirectedGraphPBM(graph_hash);
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    if (!pbm) {
        DTHROW("Graph PBM was not found.\n");
        return { };
    }
#endif
    const unsigned int bucket_begin = pbm[source_vertex_index];
    const unsigned int bucket_end = pbm[source_vertex_index + 1];
    for (unsigned int i = bucket_begin; i < bucket_end; ++i) {
        // Fetch edge destinations until we find a match
        const unsigned int t_dest_index = detail::curve::DeviceCurve::getEnvironmentDirectedGraphEdgeArrayProperty<id_t, 2>(graph_hash, "_source_dest", i, 0);
        if (t_dest_index == destination_vertex_index)
            return i;
    }
    return { };
}
template<typename T, unsigned int M>
__device__ __forceinline__ T DeviceEnvironmentDirectedGraph::getEdgeProperty(const char(&property_name)[M], const unsigned int edge_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int EDGE_COUNT = detail::curve::DeviceCurve::getVariableCount("_source_dest", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_edge"));
    if (edge_index >= EDGE_COUNT) {
        DTHROW("Edge index (%u) exceeds edge count (%u), unable to get property '%s'.\n", edge_index, EDGE_COUNT, property_name);
        return {};
    }
#endif
    return detail::curve::DeviceCurve::getEnvironmentDirectedGraphEdgeProperty<T>(graph_hash, property_name, edge_index);
}
template<typename T, flamegpu::size_type N, unsigned int M>
__device__ __forceinline__ T DeviceEnvironmentDirectedGraph::getEdgeProperty(const char(&property_name)[M], const unsigned int edge_index, const unsigned int element_index) const {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    const unsigned int EDGE_COUNT = detail::curve::DeviceCurve::getVariableCount("_source_dest", graph_hash ^ detail::curve::Curve::variableHash("_environment_directed_graph_edge"));
    if (edge_index >= EDGE_COUNT) {
        DTHROW("Edge index (%u) exceeds edge count (%u), unable to get property '%s'.\n", edge_index, EDGE_COUNT, property_name);
        return {};
    }
#endif
    return detail::curve::DeviceCurve::getEnvironmentDirectedGraphEdgeArrayProperty<T, N>(graph_hash, property_name, edge_index, element_index);
}
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_ENVIRONMENT_DEVICEENVIRONMENTDIRECTEDGRAPH_CUH_
