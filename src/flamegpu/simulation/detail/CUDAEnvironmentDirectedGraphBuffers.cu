#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"

#include <algorithm>

#include "flamegpu/simulation/detail/CUDAAgent.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/simulation/detail/CUDAScatter.cuh"
#include "flamegpu/runtime/detail/curve/HostCurve.cuh"
#include "flamegpu/detail/cuda.cuh"
#ifdef FLAMEGPU_VISUALISATION
#include "flamegpu/visualiser/ModelVis.h"
#include "flamegpu/visualiser/FLAMEGPU_Visualisation.h"
#endif
#ifdef _MSC_VER
#pragma warning(push, 1)
#pragma warning(disable : 4706 4834)
#endif  // _MSC_VER
#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_suppress 1719
#else
#pragma diag_suppress 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#include <cub/cub.cuh>

#ifdef __NVCC_DIAG_PRAGMA_SUPPORT__
#pragma nv_diag_default 1719
#else
#pragma diag_default 1719
#endif  // __NVCC_DIAG_PRAGMA_SUPPORT__
#ifdef _MSC_VER
#pragma warning(pop)
#endif  // _MSC_VER

namespace flamegpu {
namespace detail {

CUDAEnvironmentDirectedGraphBuffers::CUDAEnvironmentDirectedGraphBuffers(const EnvironmentDirectedGraphData& description)
    : graph_description(description)
    , vertex_count(0)
    , edge_count(0)
    , requires_rebuild(false) {
    // Initialise the maps
    for (const auto &v : graph_description.vertexProperties)
        vertex_buffers.emplace(v.first, Buffer{v.second.type_size * v.second.elements});
    for (const auto& e : graph_description.edgeProperties)
        edge_buffers.emplace(e.first, Buffer{e.second.type_size * e.second.elements });
}
CUDAEnvironmentDirectedGraphBuffers::~CUDAEnvironmentDirectedGraphBuffers() {
    deallocateVertexBuffers();
    vertex_buffers.clear();
    deallocateEdgeBuffers();
    edge_buffers.clear();
}
void CUDAEnvironmentDirectedGraphBuffers::registerCurveInstance(const std::shared_ptr<detail::curve::HostCurve>& curve) {
    curve_instances.push_back(std::weak_ptr(curve));
}
void CUDAEnvironmentDirectedGraphBuffers::registerCurveInstance(const std::shared_ptr<detail::curve::CurveRTCHost>& curve) {
    rtc_curve_instances.push_back(std::weak_ptr(curve));
}
void CUDAEnvironmentDirectedGraphBuffers::allocateVertexBuffers(const size_type count, const cudaStream_t stream) {
    for (auto& v : graph_description.vertexProperties) {
        auto &vb = vertex_buffers.at(v.first);
        if (!vb.d_ptr) {
            gpuErrchk(cudaMalloc(&vb.d_ptr, count * v.second.type_size * v.second.elements));
            // gpuErrchk(cudaMalloc(&vb.d_ptr_swap, count * v.second.type_size * v.second.elements));  // Todo: required?
            for (const auto & _curve : curve_instances) {
                if (const auto curve = _curve.lock())
                    curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, v.first, vb.d_ptr, count);
            }
            for (const auto& _curve : rtc_curve_instances) {
                if (const auto curve = _curve.lock()) {
                    memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, v.first), &vb.d_ptr, sizeof(void*));
                    curve->setEnvironmentDirectedGraphVertexPropertyCount(graph_description.name, v.first, count);
                }
            }
        } else {
            THROW exception::UnknownInternalError("Device buffer already allocated, in CUDAEnvironmentDirectedGraphBuffers::allocateVertexBuffers()");
        }
        if (!vb.h_ptr) {
            vb.h_ptr = malloc(count * v.second.type_size * v.second.elements);
        } else {
            THROW exception::UnknownInternalError("Host buffer already allocated, in CUDAEnvironmentDirectedGraphBuffers::allocateVertexBuffers()");
        }
        vb.ready = Buffer::Both;
    }
    // Min length 4, as pbm_swap is used when building graph
    gpuErrchk(cudaMalloc(&d_pbm, sizeof(unsigned int) * std::max<int>(count + 1, 4)));
    gpuErrchk(cudaMalloc(&d_pbm_swap, sizeof(unsigned int) * std::max<int>(count + 1, 4)));
    gpuErrchk(cudaMalloc(&d_ipbm, sizeof(unsigned int)* std::max<int>(count + 1, 4)));
    // Initialise PBMs incase they doesn't contain edges
    gpuErrchk(cudaMemsetAsync(d_pbm, 0, (count + 1) * sizeof(unsigned int), stream));
    gpuErrchk(cudaMemsetAsync(d_ipbm, 0, (count + 1) * sizeof(unsigned int), stream));
    for (const auto& _curve : curve_instances) {
        if (const auto curve = _curve.lock()) {
            curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_PBM_VARIABLE_NAME, d_pbm, 1);
            curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_IPBM_VARIABLE_NAME, d_ipbm, 1);
            curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME, d_ipbm, 1);  // IPBM needs to point somewhere
            curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME, d_vertex_index_map, 1);  // ID map needs to point somewhere
        }
    }
    for (const auto& _curve : rtc_curve_instances) {
        if (const auto curve = _curve.lock()) {
            memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_PBM_VARIABLE_NAME), &d_pbm, sizeof(void*));
            memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_IPBM_VARIABLE_NAME), &d_ipbm, sizeof(void*));
            memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME), &d_ipbm, sizeof(void*));  // IPBM needs to point somewhere
            memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME), &d_ipbm, sizeof(void*));  // ID map needs to point somewhere
            curve->setEnvironmentDirectedGraphVertexPropertyCount(graph_description.name, GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME, 1);  // 1 because offset is packed at the end
        }
    }
    vertex_count = count;
}
void CUDAEnvironmentDirectedGraphBuffers::allocateEdgeBuffers(const size_type count) {
    for (auto& e : graph_description.edgeProperties) {
        auto& eb = edge_buffers.at(e.first);
        if (!eb.d_ptr) {
            gpuErrchk(cudaMalloc(&eb.d_ptr, count * e.second.type_size * e.second.elements));
            gpuErrchk(cudaMalloc(&eb.d_ptr_swap, count * e.second.type_size * e.second.elements));
            for (const auto& _curve : curve_instances) {
                if (const auto curve = _curve.lock())
                    curve->setEnvironmentDirectedGraphEdgeProperty(graph_description.name, e.first, eb.d_ptr, count);
            }
            for (const auto& _curve : rtc_curve_instances) {
                if (const auto curve = _curve.lock()) {
                    memcpy(curve->getEnvironmentDirectedGraphEdgePropertyCachePtr(graph_description.name, e.first), &eb.d_ptr, sizeof(void*));
                    curve->setEnvironmentDirectedGraphEdgePropertyCount(graph_description.name, e.first, count);
                }
            }
        } else {
            THROW exception::UnknownInternalError("Device buffer already allocated, in CUDAEnvironmentDirectedGraphBuffers::allocateEdgeBuffers()");
        }
        if (!eb.h_ptr) {
            eb.h_ptr = malloc(count * e.second.type_size * e.second.elements);
        } else {
            THROW exception::UnknownInternalError("Host buffer already allocated, in CUDAEnvironmentDirectedGraphBuffers::allocateEdgeBuffers()");
        }
        eb.ready = Buffer::Both;
    }
    gpuErrchk(cudaMalloc(&d_keys, sizeof(uint64_t) * count));
    gpuErrchk(cudaMalloc(&d_keys_swap, sizeof(uint64_t) * count));
    gpuErrchk(cudaMalloc(&d_vals, sizeof(uint32_t) * (count + 1)));
    gpuErrchk(cudaMalloc(&d_vals_swap, sizeof(uint32_t) * (count + 1)));
    gpuErrchk(cudaMalloc(&d_ipbm_edges, sizeof(uint32_t) * (count + 1)));
    for (const auto& _curve : curve_instances) {
        if (const auto curve = _curve.lock()) {
            curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME, d_ipbm_edges, 1);
        }
    }
    for (const auto& _curve : rtc_curve_instances) {
        if (const auto curve = _curve.lock()) {
            memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME), &d_ipbm_edges, sizeof(void*));
            curve->setEnvironmentDirectedGraphVertexPropertyCount(graph_description.name, GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME, 1);  // 1 because offset is packed at the end
        }
    }
    edge_count = count;
}
void CUDAEnvironmentDirectedGraphBuffers::deallocateVertexBuffers() {
    for (auto& v : vertex_buffers) {
        if (v.second.d_ptr) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(v.second.d_ptr));
            gpuErrchk(flamegpu::detail::cuda::cudaFree(v.second.d_ptr_swap));
            v.second.d_ptr = nullptr;
        }
        if (v.second.h_ptr) {
            free(v.second.h_ptr);
            v.second.h_ptr = nullptr;
        }
    }
    if (d_pbm) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_pbm));
        d_pbm = nullptr;
    }
    if (d_pbm_swap) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_pbm_swap));
        d_pbm_swap = nullptr;
    }
    if (d_ipbm) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_ipbm));
        d_ipbm = nullptr;
    }
    if (d_vertex_index_map) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_vertex_index_map));
        d_vertex_index_map = nullptr;
    }
    vertex_count = 0;
    h_vertex_index_map.clear();
}
void CUDAEnvironmentDirectedGraphBuffers::deallocateEdgeBuffers() {
    for (auto& e : edge_buffers) {
        if (e.second.d_ptr) {
            gpuErrchk(flamegpu::detail::cuda::cudaFree(e.second.d_ptr));
            gpuErrchk(flamegpu::detail::cuda::cudaFree(e.second.d_ptr_swap));
            e.second.d_ptr = nullptr;
        }
        if (e.second.h_ptr) {
            free(e.second.h_ptr);
            e.second.h_ptr = nullptr;
        }
    }
    if (d_keys) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_keys));
        d_keys = nullptr;
    }
    if (d_vals) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_vals));
        d_vals = nullptr;
    }
    if (d_keys_swap) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_keys_swap));
        d_keys_swap = nullptr;
    }
    if (d_vals_swap) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_vals_swap));
        d_vals_swap = nullptr;
    }
    if (d_ipbm_edges) {
        gpuErrchk(flamegpu::detail::cuda::cudaFree(d_ipbm_edges));
        d_ipbm_edges = nullptr;
    }
    edge_count = 0;
    h_edge_index_map.clear();
}

void CUDAEnvironmentDirectedGraphBuffers::setVertexCount(const size_type count, const cudaStream_t stream) {
    if (vertex_count) {
        deallocateVertexBuffers();
    }
    allocateVertexBuffers(count, stream);
    // Default Init host, mark device out of date
    for (auto& v : graph_description.vertexProperties) {
        auto& vb = vertex_buffers.at(v.first);
        vb.ready = Buffer::Host;
        if (v.first == ID_VARIABLE_NAME) {  // ID needs default 0
            memset(vb.h_ptr, ID_NOT_SET, vertex_count * v.second.type_size * v.second.elements);
            continue;
        }
        // Possibly faster if we checked default_value == 0 and memset, but awkward with vague type and lack of template
        for (unsigned int i = 0; i < vertex_count; ++i) {
            // TODO is this just copy-paste junk?
            memcpy(static_cast<char*>(vb.h_ptr) + i * v.second.type_size * v.second.elements, v.second.default_value, v.second.type_size * v.second.elements);
        }
    }
    // Vertex data has been reset, so ID bounds are nolonger valid
    resetVertexIDBounds();
}

void CUDAEnvironmentDirectedGraphBuffers::setEdgeCount(const size_type count) {
    if (edge_count)
        deallocateEdgeBuffers();
    allocateEdgeBuffers(count);
    // Default Init host, mark device out of date
    for (auto& e : graph_description.edgeProperties) {
        auto& eb = edge_buffers.at(e.first);
        eb.ready = Buffer::Host;
        // Possibly faster if we checked default_value == 0 and memset, but awkward with vague type and lack of template
        for (unsigned int i = 0; i < edge_count; ++i) {
            // TODO is this just copy-paste junk?
            memcpy(static_cast<char*>(eb.h_ptr) + i * e.second.type_size * e.second.elements, e.second.default_value, e.second.type_size * e.second.elements);
        }
    }
}
id_t* CUDAEnvironmentDirectedGraphBuffers::getVertexIDBuffer(const cudaStream_t stream) {
    size_type element_ct = 1;
    return getVertexPropertyBuffer<id_t>(ID_VARIABLE_NAME, element_ct, stream);
}

__global__ void fillKVPairs(uint32_t *keys, uint32_t *vals, const unsigned int *srcdest, unsigned int count, const unsigned int *idMap, const unsigned int id_offset) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < count) {
        // To subsort by destination too, we treat the pair of uint32 as a uint64
        keys[index * 2 + 0] = idMap[srcdest[index * 2 + 0] - id_offset];
        keys[index * 2 + 1] = idMap[srcdest[index * 2 + 1] - id_offset];
        vals[index] = index;
    }
}
__global__ void fillKVPairs_inverted(uint32_t* keys, uint32_t* vals, const unsigned int* srcdest, unsigned int count, const unsigned int *idMap, const unsigned int id_offset) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < count) {
        // To subsort by destination too, we treat the pair of uint32 as a uint64
        // To invert we must switch the order of the contained uint32's
        keys[index * 2 + 0] = idMap[srcdest[index * 2 + 1] - id_offset];
        keys[index * 2 + 1] = idMap[srcdest[index * 2 + 0] - id_offset];
        vals[index] = index;
    }
}
__global__ void findBinStart(unsigned int *pbm, uint64_t* keys, unsigned int edge_count, unsigned int vertex_count) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < edge_count) {
        // Bins correspond to the first uint32 of the pair
        const uint32_t my_vertex = reinterpret_cast<const uint32_t*>(keys)[(index * 2)+1];
        if (index == 0 || my_vertex != reinterpret_cast<const uint32_t*>(keys)[((index - 1) * 2)+1]) {
            // Store the Index of the first edges for the corresponding vertex
            pbm[my_vertex] = index;
        }
    }
    // 1 thread must init the final cell of the PBM
    if (index == 0) {
        pbm[vertex_count] = edge_count;
    }
}
/**
* This utility class provides a wrapper for `unsigned int *`
* It causes the pointer to iterate in reverse backwards
*/
struct ReverseIterator {
    using difference_type = unsigned int;
    using value_type = unsigned int;
    using pointer = unsigned int*;
    using reference = unsigned int&;
    using iterator_category = std::random_access_iterator_tag;
    __host__ __device__ explicit ReverseIterator(unsigned int* _p) : p(_p) { }

    // __device__ ReverseIterator& operator=(const ReverseIterator& other) = default;
    __device__ ReverseIterator operator++ (int a) { p -= a;  return *this; }
    __device__ ReverseIterator operator++ () { p--;  return *this; }
    __device__ unsigned int &operator *() const { return *p; }
    __device__ ReverseIterator operator+(const int& b) const { return ReverseIterator(p - b); }
    __device__ unsigned int &operator[](int b) const { return *(p-b); }
    unsigned int* p;
};
// Borrowed from CUB DeviceScan docs
struct CustomMin {
    template <typename T>
    CUB_RUNTIME_FUNCTION __forceinline__
        T operator()(const T& a, const T& b) const {
        return (b < a) ? b : a;
    }
};
__global__ void buildIDMap(const id_t *IDsIn, unsigned int *indexOut, const unsigned int count, unsigned int *error_count, unsigned int vertex_id_min, unsigned int vertex_id_max) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < count) {
        id_t my_thread_id = IDsIn[thread_index];
        // Skip IDs that weren't set
        if (my_thread_id == ID_NOT_SET) {
            atomicInc(error_count + 2, UINT_MAX);
            return;
        }
        if (vertex_id_min <= my_thread_id && my_thread_id <= vertex_id_max) {
            const unsigned int rtn = atomicExch(indexOut + my_thread_id - vertex_id_min, thread_index);
            if (rtn != 0xffffffff) {
                // Report ID collision
                atomicInc(error_count + 0, UINT_MAX);
            }
        } else {
            // Report out of range ID (this should not happen, it's an internal error if it does)
            atomicInc(error_count + 1, UINT_MAX);
        }
    }
}
__global__ void validateSrcDest(id_t *edgeSrcDest, unsigned int *idMap, const unsigned int edge_count, unsigned int *errors, unsigned int vertex_id_min, unsigned int vertex_id_max) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < edge_count) {
        const id_t my_src_id = edgeSrcDest[thread_index * 2 + 1];
        const id_t my_dest_id = edgeSrcDest[thread_index * 2 + 0];
        if (my_src_id == ID_NOT_SET) {
            atomicInc(errors + 0, UINT_MAX);
        } else if (vertex_id_min <= my_src_id && my_src_id <= vertex_id_max) {
            if (idMap[my_src_id - vertex_id_min] == 0xffffffff) {
                atomicInc(errors + 2, UINT_MAX);
            }
        } else {
            atomicInc(errors + 2, UINT_MAX);
        }
        if (my_dest_id == ID_NOT_SET) {
            atomicInc(errors + 1, UINT_MAX);
        } else if (vertex_id_min <= my_dest_id && my_dest_id <= vertex_id_max) {
            if (idMap[my_dest_id - vertex_id_min] == 0xffffffff) {
                atomicInc(errors + 3, UINT_MAX);
            }
        } else {
            atomicInc(errors + 3, UINT_MAX);
        }
    }
}
__global__ void translateSrcDest(id_t *edgeSrcDest, unsigned int *idMap, const unsigned int edge_count, unsigned int *errors, unsigned int vertex_id_min, unsigned int vertex_id_max) {
    const unsigned int thread_index = blockIdx.x * blockDim.x + threadIdx.x;
    if (thread_index < edge_count) {
        const id_t my_src_id = edgeSrcDest[thread_index * 2 + 1];
        const id_t my_dest_id = edgeSrcDest[thread_index * 2 + 0];
        const unsigned int src_id = idMap[my_src_id - vertex_id_min];
        const unsigned int dest_id = idMap[my_dest_id - vertex_id_min];
        edgeSrcDest[thread_index * 2 + 1] = src_id;
        edgeSrcDest[thread_index * 2 + 0] = dest_id;
    }
}
void CUDAEnvironmentDirectedGraphBuffers::syncDevice_async(detail::CUDAScatter& scatter, const unsigned int streamID, const cudaStream_t stream) {
    bool has_changed = false;
    // Copy variable buffers to device
    if (vertex_count) {
        for (auto& v : graph_description.vertexProperties) {
            auto& vb = vertex_buffers.at(v.first);
            if (vb.ready == Buffer::Host) {
                gpuErrchk(cudaMemcpyAsync(vb.d_ptr, vb.h_ptr, vertex_count * v.second.type_size * v.second.elements, cudaMemcpyHostToDevice, stream));
                vb.ready = Buffer::Both;
                has_changed = true;
            }
        }
    }
    if (edge_count) {
        for (auto& e : graph_description.edgeProperties) {
            auto& eb = edge_buffers.at(e.first);
            if (eb.ready == Buffer::Host) {
                gpuErrchk(cudaMemcpyAsync(eb.d_ptr, eb.h_ptr, edge_count * e.second.type_size * e.second.elements, cudaMemcpyHostToDevice, stream));
                eb.ready = Buffer::Both;
                has_changed = true;
            }
        }
    }
    if (requires_rebuild && vertex_count && edge_count) {
        // Construct the vertex ID : index map
        {
            if (vertex_id_min == std::numeric_limits<unsigned int>::max() || vertex_id_max == std::numeric_limits<unsigned int>::min()) {
                THROW flamegpu::exception::IDOutOfBounds("No IDs have been set, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()");
            }
            const unsigned int ID_RANGE = 1 + vertex_id_max - vertex_id_min;
            if (ID_RANGE < vertex_count) {
                THROW flamegpu::exception::IDNotSet("Not all vertices have been assigned a unique ID, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()");
            }
            if (d_vertex_index_map) {
                gpuErrchk(flamegpu::detail::cuda::cudaFree(d_vertex_index_map));
            }
            if (cudaMalloc(&d_vertex_index_map, sizeof(unsigned int) * (ID_RANGE + 1)) != cudaSuccess) {
                THROW flamegpu::exception::OutOfMemory("Out of memory when allocating ID->index map, Vertex IDs cover too wide a range (%u) consider contiguous IDs, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()", ID_RANGE);
            }
            // Copy the offset to the end of the map
            gpuErrchk(cudaMemcpyAsync(d_vertex_index_map + ID_RANGE, &vertex_id_min, sizeof(unsigned int), cudaMemcpyHostToDevice, stream));
            // Add the ID->index map var to curve
            for (const auto& _curve : curve_instances) {
                if (const auto curve = _curve.lock())
                    curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME, d_vertex_index_map, ID_RANGE + 1);  // +1 because offset is packed at the end
            }
            for (const auto& _curve : rtc_curve_instances) {
                if (const auto curve = _curve.lock()) {
                    memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME), &d_vertex_index_map, sizeof(void*));
                    curve->setEnvironmentDirectedGraphVertexPropertyCount(graph_description.name, GRAPH_VERTEX_INDEX_MAP_VARIABLE_NAME, ID_RANGE + 1);  // +1 because offset is packed at the end
                }
            }
            {  // Build the map
                const auto& v_id_b = vertex_buffers.at(ID_VARIABLE_NAME);
                gpuErrchk(cudaMemsetAsync(d_vertex_index_map, 0xffffffff, ID_RANGE * sizeof(unsigned int), stream));
                gpuErrchk(cudaMemsetAsync(d_pbm_swap, 0, 3 * sizeof(unsigned int), stream));  // We will use spare pbm_swap to count errors, save allocating more memory
                const unsigned int BLOCK_SZ = 512;
                const unsigned int BLOCK_CT = static_cast<unsigned int>(ceil(vertex_count / static_cast<float>(BLOCK_SZ)));
                buildIDMap << <BLOCK_CT, BLOCK_SZ, 0, stream >> > (static_cast<id_t*>(v_id_b.d_ptr), d_vertex_index_map, vertex_count, d_pbm_swap, vertex_id_min, vertex_id_max);
                gpuErrchkLaunch();
                unsigned int err_collision_range[3];
                gpuErrchk(cudaMemcpyAsync(err_collision_range, d_pbm_swap, 3 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
                gpuErrchk(cudaStreamSynchronize(stream));
                if (err_collision_range[2] > 0) {
                    THROW flamegpu::exception::IDNotSet("Graph contains %u vertices which have not had their ID set, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()", err_collision_range[2]);
                } else if (err_collision_range[0] > 0) {
                    THROW flamegpu::exception::IDCollision("Graph contains invalid vertex IDs, %u vertices reported ID collisions, vertex IDs must be unique or unset, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()", err_collision_range[0]);
                } else if (err_collision_range[1] > 0) {
                    THROW flamegpu::exception::UnknownInternalError("Graph contains invalid vertex IDs, %u vertices reported an ID that does not satisfy %u < ID < %u, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()", err_collision_range[1], vertex_id_min, vertex_id_max);
                }
            }
            {  // Validate that edge source/dest pairs correspond to valid IDs
                const auto& e_srcdest_b = edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME);
                gpuErrchk(cudaMemsetAsync(d_pbm_swap, 0, 4 * sizeof(unsigned int), stream));  // We will use spare pbm_swap to count errors, save allocating more memory
                const unsigned int BLOCK_SZ = 512;
                const unsigned int BLOCK_CT = static_cast<unsigned int>(ceil(edge_count / static_cast<float>(BLOCK_SZ)));
                validateSrcDest << <BLOCK_CT, BLOCK_SZ, 0, stream >> > (static_cast<id_t*>(e_srcdest_b.d_ptr), d_vertex_index_map, edge_count, d_pbm_swap, vertex_id_min, vertex_id_max);
                gpuErrchkLaunch();
                unsigned int err_collision_range[4];  // {src_notset, dest_notset, src_invalid, dest_invalid}
                gpuErrchk(cudaMemcpyAsync(err_collision_range, d_pbm_swap, 4 * sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
                gpuErrchk(cudaStreamSynchronize(stream));
                if (err_collision_range[0] > 0 || err_collision_range[1] > 0) {
                    THROW flamegpu::exception::IDNotSet("Graph contains %u and %u edges which have not had their source and destinations set respectively, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()", err_collision_range[0], err_collision_range[1]);
                } else if (err_collision_range[2] > 0 || err_collision_range[3] > 0) {
                    THROW flamegpu::exception::InvalidID("Graph contains %u and %u edges which have invalid source and destinations set respectively, in CUDAEnvironmentDirectedGraphBuffers::syncDevice_async()", err_collision_range[2], err_collision_range[3]);
                }
            }
        }
        // Rebuild the CSR/VBM (edgesLeaving())
        {
            // Fill Key/Val Pairs
            int blockSize;  // The launch configurator returned block size
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, fillKVPairs, 32, 0));  // Randomly 32
            int gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            fillKVPairs << <gridSize, blockSize, 0, stream >> > (reinterpret_cast<uint32_t*>(d_keys), d_vals, static_cast<unsigned int*>(edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME).d_ptr), edge_count, d_vertex_index_map, vertex_id_min);
            gpuErrchkLaunch();
            // Sort Key/Val Pairs according to src->dest
            auto& cub_temp = scatter.CubTemp(streamID);
            size_t temp_req = 0;
            gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, temp_req, d_keys, d_keys_swap, d_vals, d_vals_swap, edge_count, 0, sizeof(uint64_t) * 8, stream));
            cub_temp.resize(temp_req);
            gpuErrchk(cub::DeviceRadixSort::SortPairs(cub_temp.getPtr(), cub_temp.getSize(), d_keys, d_keys_swap, d_vals, d_vals_swap, edge_count, 0, sizeof(uint64_t) * 8, stream));
            // Build PBM (For vertices with edges)
            gpuErrchk(cudaMemset(d_pbm, 0xffffffff, (vertex_count + 1) * sizeof(unsigned int)));
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, findBinStart, 32, 0));  // Randomly 32
            gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            findBinStart << <gridSize, blockSize, 0, stream >> > (d_pbm, d_keys_swap, edge_count, vertex_count);
            gpuErrchkLaunch();
            // Build PBM (Fill vertices with no edges)
            temp_req = 0;
            gpuErrchk(cub::DeviceScan::InclusiveScan(nullptr, temp_req, ReverseIterator(d_pbm + vertex_count), ReverseIterator(d_pbm_swap + vertex_count), CustomMin(), vertex_count + 1, stream));
            cub_temp.resize(temp_req);
            gpuErrchk(cub::DeviceScan::InclusiveScan(cub_temp.getPtr(), cub_temp.getSize(), ReverseIterator(d_pbm + vertex_count), ReverseIterator(d_pbm_swap + vertex_count), CustomMin(), vertex_count + 1, stream));
            // Sort edge variables
            std::vector<detail::CUDAScatter::ScatterData> sd;
            for (auto& edge : edge_buffers) {
                edge.second.swap();
                sd.push_back(detail::CUDAScatter::ScatterData{edge.second.element_size, reinterpret_cast<char*>(edge.second.d_ptr_swap), reinterpret_cast<char*>(edge.second.d_ptr)});
            }
            scatter.scatterPosition_async(streamID, stream, d_vals_swap, sd, edge_count);
            // Swap all the swap pointers, so the junk data is in swap
            std::swap(d_keys, d_keys_swap);
            std::swap(d_vals, d_vals_swap);
            std::swap(d_pbm, d_pbm_swap);
            // Update which buffers curve points to
            for (auto& e : graph_description.edgeProperties) {
                auto& eb = edge_buffers.at(e.first);
                for (const auto& _curve : curve_instances) {
                    if (const auto curve = _curve.lock())
                        curve->setEnvironmentDirectedGraphEdgeProperty(graph_description.name, e.first, eb.d_ptr, edge_count);
                }
                for (const auto& _curve : rtc_curve_instances) {
                    if (const auto curve = _curve.lock())
                        memcpy(curve->getEnvironmentDirectedGraphEdgePropertyCachePtr(graph_description.name, e.first), &eb.d_ptr, sizeof(void*));
                }
                eb.ready = Buffer::Device;
            }
            for (const auto& _curve : curve_instances) {
                if (const auto curve = _curve.lock())
                    curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_PBM_VARIABLE_NAME, d_pbm, 1);
            }
            for (const auto& _curve : rtc_curve_instances) {
                if (const auto curve = _curve.lock())
                    memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_PBM_VARIABLE_NAME), &d_pbm, sizeof(void*));
            }
        }
        {  // Rebuild the CSC/Inverted VBM (edgesJoining())
            int blockSize;  // The launch configurator returned block size
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, fillKVPairs, 32, 0));  // Randomly 32
            int gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            fillKVPairs_inverted << <gridSize, blockSize, 0, stream >> > (reinterpret_cast<uint32_t*>(d_keys), d_vals, static_cast<unsigned int*>(edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME).d_ptr), edge_count, d_vertex_index_map, vertex_id_min);
            gpuErrchkLaunch();
            // Sort Key/Val Pairs according to dest->src
            // Cub temp has already been resized above
            auto& cub_temp = scatter.CubTemp(streamID);
            gpuErrchk(cub::DeviceRadixSort::SortPairs(cub_temp.getPtr(), cub_temp.getSize(), d_keys, d_keys_swap, d_vals, d_vals_swap, edge_count, 0, sizeof(uint64_t) * 8, stream));
            // Build inverted PBM (For vertices with edges)
            gpuErrchk(cudaMemset(d_ipbm, 0xffffffff, (vertex_count + 1) * sizeof(unsigned int)));
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, findBinStart, 32, 0));  // Randomly 32
            gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            findBinStart << <gridSize, blockSize, 0, stream >> > (d_ipbm, d_keys_swap, edge_count, vertex_count);
            gpuErrchkLaunch();
            // Build inverted PBM (Fill vertices with no edges)
            gpuErrchk(cub::DeviceScan::InclusiveScan(cub_temp.getPtr(), cub_temp.getSize(), ReverseIterator(d_ipbm + vertex_count), ReverseIterator(d_pbm_swap + vertex_count), CustomMin(), vertex_count + 1, stream));
            // Swap all the swap pointers, so the junk data is in swap
            std::swap(d_keys, d_keys_swap);
            std::swap(d_ipbm_edges, d_vals_swap);
            std::swap(d_ipbm, d_pbm_swap);
            // Update which buffers curve points to
            for (const auto& _curve : curve_instances) {
                if (const auto curve = _curve.lock()) {
                    curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_IPBM_VARIABLE_NAME, d_ipbm, 1);
                    curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME, d_ipbm_edges, 1);
                }
            }
            for (const auto& _curve : rtc_curve_instances) {
                if (const auto curve = _curve.lock()) {
                    memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_IPBM_VARIABLE_NAME), &d_ipbm, sizeof(void*));
                    memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_IPBM_EDGES_VARIABLE_NAME), &d_ipbm_edges, sizeof(void*));
                }
            }
        }
        {  // Translate edge source/dest pairs and validate that they correspond to valid IDs
            const auto& e_srcdest_b = edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME);
            e_srcdest_b.updateHostBuffer(edge_count, stream);  // Copy back to host, before we translate device IDs
            const unsigned int BLOCK_SZ = 512;
            const unsigned int BLOCK_CT = static_cast<unsigned int>(ceil(edge_count / static_cast<float>(BLOCK_SZ)));
            translateSrcDest << <BLOCK_CT, BLOCK_SZ, 0, stream >> > (static_cast<id_t*>(e_srcdest_b.d_ptr), d_vertex_index_map, edge_count, d_pbm_swap, vertex_id_min, vertex_id_max);
            gpuErrchkLaunch()
            // Rebuild the edge index map
            h_edge_index_map.clear();
            for (unsigned int i = 0; i < edge_count; ++i) {
                h_edge_index_map.emplace(std::pair{static_cast<id_t*>(e_srcdest_b.h_ptr)[i * 2 + 1], static_cast<id_t*>(e_srcdest_b.h_ptr)[i * 2 + 0]}, i);
            }
        }
        requires_rebuild = false;
        has_changed = true;
    }
    if (has_changed) {
#ifdef FLAMEGPU_VISUALISATION
        if (auto vis = visualisation.lock()) {
            vis->visualiser->lockDynamicLinesMutex();
            vis->rebuildEnvGraph(graph_description.name);
            vis->visualiser->updateDynamicLine(std::string("graph_") + graph_description.name);
            vis->visualiser->releaseDynamicLinesMutex();
        }
#endif
    }
}

void CUDAEnvironmentDirectedGraphBuffers::Buffer::updateHostBuffer(size_type edge_count, cudaStream_t stream) const {
    if (ready == Device) {
        gpuErrchk(cudaMemcpyAsync(h_ptr, d_ptr, edge_count * element_size, cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        ready = Both;
    }
}
void CUDAEnvironmentDirectedGraphBuffers::resetVertexIDBounds() {
    vertex_id_min = std::numeric_limits<unsigned int>::max();
    vertex_id_max = std::numeric_limits<unsigned int>::min();
}
void CUDAEnvironmentDirectedGraphBuffers::setVertexID(unsigned int vertex_index, id_t vertex_id, cudaStream_t stream) {
    if (vertex_index >= vertex_count) {
        THROW exception::OutOfBoundsException("Vertex index exceeds bounds %u >= %u, "
            "in CUDAEnvironmentDirectedGraphBuffers::setVertexID()\n", vertex_index, vertex_count);
    } else if (vertex_id == ID_NOT_SET) {
        THROW exception::IDOutOfBounds("Vertex ID of %u is not valid, "
            "in CUDAEnvironmentDirectedGraphBuffers::setVertexID()\n", ID_NOT_SET);
    }
    // Purge old vertex ID from host map
    auto& vb = vertex_buffers.at(ID_VARIABLE_NAME);
    vb.updateHostBuffer(vertex_count, stream);
    if (static_cast<id_t*>(vb.h_ptr)[vertex_index] != ID_NOT_SET) {
        h_vertex_index_map.erase(static_cast<id_t*>(vb.h_ptr)[vertex_index]);
    }

    // Add new vertex ID to host map (validate it's not already in us)
    const auto find = h_vertex_index_map.find(vertex_id);
    if (find != h_vertex_index_map.end()) {
        THROW exception::IDCollision("ID collision, %u has already been assigned to vertex at index %u, "
            "in CUDAEnvironmentDirectedGraphBuffers::setVertexID()\n", vertex_id, find->second);
    }
    h_vertex_index_map.emplace(vertex_id, vertex_index);

    // Update vertex's ID in buffer
    static_cast<id_t*>(vb.h_ptr)[vertex_index] = vertex_id;
    vb.ready = Buffer::Host;

    // Update range calc (naive, can be wrong if IDs are changed)
    vertex_id_min = std::min(vertex_id_min, vertex_id);
    vertex_id_max = std::max(vertex_id_max, vertex_id);
}
unsigned int CUDAEnvironmentDirectedGraphBuffers::getVertexIndex(id_t vertex_id) const {
    const auto find = h_vertex_index_map.find(vertex_id);
    if (find == h_vertex_index_map.end()) {
        THROW exception::InvalidID("No vertex found with ID %u, in CUDAEnvironmentDirectedGraphBuffers::getVertexIndex()\n", vertex_id);
    }
    return find->second;
}
void CUDAEnvironmentDirectedGraphBuffers::setEdgeSourceDestination(unsigned int edge_index, id_t src_vertex_id, id_t dest_vertex_id) {
    if (edge_index >= edge_count) {
        THROW exception::OutOfBoundsException("Edge index exceeds bounds %u >= %u, "
            "in CUDAEnvironmentDirectedGraphBuffers::setEdgeSourceDestination()\n", edge_index, edge_count);
    } else if (src_vertex_id == ID_NOT_SET) {
        THROW exception::IDOutOfBounds("Source vertex ID of %u is not valid, "
            "in CUDAEnvironmentDirectedGraphBuffers::setEdgeSourceDestination()\n", ID_NOT_SET);
    } else if (dest_vertex_id == ID_NOT_SET) {
        THROW exception::IDOutOfBounds("Destination vertex ID of %u is not valid, "
            "in CUDAEnvironmentDirectedGraphBuffers::setEdgeSourceDestination()\n", ID_NOT_SET);
    }
    // Purge old edge src/dest from host map
    auto& eb = edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME);
    // Don't need to update buffer, src_dest is not stored as ID on device
    id_t& edge_dest = static_cast<id_t*>(eb.h_ptr)[edge_index * 2 + 0];
    id_t& edge_src = static_cast<id_t*>(eb.h_ptr)[edge_index * 2 + 1];

    if (edge_src != ID_NOT_SET && edge_dest != ID_NOT_SET) {
        h_edge_index_map.erase({edge_src, edge_dest});
    } else if ((edge_src == ID_NOT_SET) ^ (edge_dest == ID_NOT_SET)) {
        THROW exception::UnknownInternalError("Edge found without both source and destination set, "
            "in CUDAEnvironmentDirectedGraphBuffers::setEdgeSourceDestination()\n");
    }

    // Add new edge ID to host map (validate it's not already in use)
    const auto find = h_edge_index_map.find({src_vertex_id, dest_vertex_id});
    if (find != h_edge_index_map.end()) {
        THROW exception::IDCollision("Edge collision, an edge has already been assigned source %u dest %u at index %u, "
            "in CUDAEnvironmentDirectedGraphBuffers::setEdgeSourceDestination()\n", src_vertex_id, dest_vertex_id, find->second);
    }
    h_edge_index_map.emplace(std::pair{src_vertex_id, dest_vertex_id}, edge_index);

    // Update edge's src dest in buffer
    edge_dest = dest_vertex_id;
    edge_src = src_vertex_id;
    eb.ready = Buffer::Host;

    // Require rebuild before use
    markForRebuild();
}
unsigned int CUDAEnvironmentDirectedGraphBuffers::getEdgeIndex(id_t src_vertex_id, id_t dest_vertex_id) const {
    const auto find = h_edge_index_map.find({src_vertex_id, dest_vertex_id});
    if (find == h_edge_index_map.end()) {
        THROW exception::InvalidID("No edge found with source %u, dest %u, in CUDAEnvironmentDirectedGraphBuffers::getEdgeIndex()\n", src_vertex_id, dest_vertex_id);
    }
    return find->second;
}

unsigned int CUDAEnvironmentDirectedGraphBuffers::createIfNotExistVertex(id_t vertex_id, const cudaStream_t stream) {
    if (vertex_id == ID_NOT_SET) {
        THROW exception::IDOutOfBounds("Vertex ID of %u is not valid, "
            "in CUDAEnvironmentDirectedGraphBuffers::createIfNotExistVertex()\n", ID_NOT_SET);
    }
    const auto it = h_vertex_index_map.find(vertex_id);
    if (it != h_vertex_index_map.end()) {
        return it->second;
    }
    if (h_vertex_index_map.size() < vertex_count) {
        const unsigned int vertex_index = static_cast<unsigned int>(h_vertex_index_map.size());
        h_vertex_index_map.emplace(vertex_id, vertex_index);
        // Update vertex's ID in buffer
        auto& vb = vertex_buffers.at(ID_VARIABLE_NAME);
        vb.updateHostBuffer(vertex_count, stream);
        static_cast<id_t*>(vb.h_ptr)[vertex_index] = vertex_id;
        vb.ready = Buffer::Host;
        // Update range calc
        vertex_id_min = std::min(vertex_id_min, vertex_id);
        vertex_id_max = std::max(vertex_id_max, vertex_id);
        return vertex_index;
    }
    THROW exception::OutOfBoundsException("Creating vertex with ID %u would exceed available vertices (%u), "
        "in CUDAEnvironmentDirectedGraphBuffers::createIfNotExistVertex()\n", vertex_id, vertex_count);
}
unsigned int CUDAEnvironmentDirectedGraphBuffers::createIfNotExistEdge(id_t source_vertex_id, id_t dest_vertex_id, const cudaStream_t stream) {
    if (source_vertex_id == ID_NOT_SET || dest_vertex_id== ID_NOT_SET) {
        THROW exception::IDOutOfBounds("Vertex ID of %u is not valid, "
            "in CUDAEnvironmentDirectedGraphBuffers::createIfNotExistEdge()\n", ID_NOT_SET);
    }
    const auto it = h_edge_index_map.find({source_vertex_id, dest_vertex_id});
    if (it != h_edge_index_map.end()) {
        return it->second;
    }
    if (h_edge_index_map.size() < edge_count) {
        const unsigned int edge_index = static_cast<unsigned int>(h_edge_index_map.size());
        h_edge_index_map.emplace(std::pair{source_vertex_id, dest_vertex_id}, edge_index);
        // Update vertex's ID in buffer
        auto& eb = edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME);
        static_cast<id_t*>(eb.h_ptr)[edge_index * 2 + 0] = dest_vertex_id;
        static_cast<id_t*>(eb.h_ptr)[edge_index * 2 + 1] = source_vertex_id;
        eb.ready = Buffer::Host;
        // Require rebuild before use
        markForRebuild();
        return edge_index;
    }
    THROW exception::OutOfBoundsException("Creating edge with src %u dest %u would exceed available edges (%u), "
        "in CUDAEnvironmentDirectedGraphBuffers::createIfNotExistEdge()\n", source_vertex_id, dest_vertex_id, vertex_count);
}
}  // namespace detail
}  // namespace flamegpu
