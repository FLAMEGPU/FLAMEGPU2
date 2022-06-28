#include "flamegpu/simulation/detail/CUDAEnvironmentDirectedGraphBuffers.cuh"

#include <algorithm>

#include "flamegpu/simulation/detail/CUDAAgent.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/simulation/detail/CUDAScatter.cuh"
#include "flamegpu/runtime/detail/curve/HostCurve.cuh"
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
void CUDAEnvironmentDirectedGraphBuffers::allocateVertexBuffers(const size_type count) {
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
    gpuErrchk(cudaMalloc(&d_pbm, sizeof(unsigned int) * (count + 1)));
    gpuErrchk(cudaMalloc(&d_pbm_swap, sizeof(unsigned int) * (count + 1)));
    gpuErrchk(cudaMalloc(&d_ipbm, sizeof(unsigned int)* (count + 1)));
    // for (const auto& _curve : curve_instances) {
    //     if (const auto curve = _curve.lock())
    //         curve->setEnvironmentDirectedGraphVertexProperty(graph_description.name, GRAPH_VERTEX_PBM_VARIABLE_NAME, d_pbm, 1);
    // }
    // for (const auto& _curve : rtc_curve_instances) {
    //     if (const auto curve = _curve.lock())
    //         memcpy(curve->getEnvironmentDirectedGraphVertexPropertyCachePtr(graph_description.name, GRAPH_VERTEX_PBM_VARIABLE_NAME), &d_pbm, sizeof(void*));
    // }
    // TODO: memset pbm to 0?
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
    // TODO: memset keys and vals to 0?
    edge_count = count;
}
void CUDAEnvironmentDirectedGraphBuffers::deallocateVertexBuffers() {
    for (auto& v : vertex_buffers) {
        if (v.second.d_ptr) {
            gpuErrchk(cudaFree(v.second.d_ptr));
            gpuErrchk(cudaFree(v.second.d_ptr_swap));
            v.second.d_ptr = nullptr;
        }
        if (v.second.h_ptr) {
            free(v.second.h_ptr);
            v.second.h_ptr = nullptr;
        }
    }
    if (d_pbm) {
        gpuErrchk(cudaFree(d_pbm));
        d_pbm = nullptr;
    }
    if (d_pbm_swap) {
        gpuErrchk(cudaFree(d_pbm_swap));
        d_pbm_swap = nullptr;
    }
    if (d_ipbm) {
        gpuErrchk(cudaFree(d_ipbm));
        d_ipbm = nullptr;
    }
    vertex_count = 0;
}
void CUDAEnvironmentDirectedGraphBuffers::deallocateEdgeBuffers() {
    for (auto& e : edge_buffers) {
        if (e.second.d_ptr) {
            gpuErrchk(cudaFree(e.second.d_ptr));
            gpuErrchk(cudaFree(e.second.d_ptr_swap));
            e.second.d_ptr = nullptr;
        }
        if (e.second.h_ptr) {
            free(e.second.h_ptr);
            e.second.h_ptr = nullptr;
        }
    }
    if (d_keys) {
        gpuErrchk(cudaFree(d_keys));
        d_keys = nullptr;
    }
    if (d_vals) {
        gpuErrchk(cudaFree(d_vals));
        d_vals = nullptr;
    }
    if (d_keys_swap) {
        gpuErrchk(cudaFree(d_keys_swap));
        d_keys_swap = nullptr;
    }
    if (d_vals_swap) {
        gpuErrchk(cudaFree(d_vals_swap));
        d_vals_swap = nullptr;
    }
    if (d_ipbm_edges) {
        gpuErrchk(cudaFree(d_ipbm_edges));
        d_ipbm_edges = nullptr;
    }
    edge_count = 0;
}

void CUDAEnvironmentDirectedGraphBuffers::setVertexCount(const size_type count) {
    if (vertex_count) {
        deallocateVertexBuffers();
    }
    allocateVertexBuffers(count);
    // Default Init host, mark device out of date
    for (auto& v : graph_description.vertexProperties) {
        auto& vb = vertex_buffers.at(v.first);
        vb.ready = Buffer::Host;
        if (v.first == ID_VARIABLE_NAME) {  // ID needs default 0
            memset(vb.h_ptr, 0, vertex_count * v.second.type_size * v.second.elements);
            continue;
        }
        // Possibly faster if we checked default_value == 0 and memset, but awkward with vague type and lack of template
        for (unsigned int i = 0; i < vertex_count; ++i) {
            // TODO is this just copy-paste junk?
            memcpy(static_cast<char*>(vb.h_ptr) + i * v.second.type_size * v.second.elements, v.second.default_value, v.second.type_size * v.second.elements);
        }
    }
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

__global__ void fillKVPairs(uint64_t *keys, uint32_t *vals, const unsigned int *srcdest, unsigned int count) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < count) {
        // To subsort by destination too, we treat the pair of uint32 as a uint64
        keys[index] = reinterpret_cast<const uint64_t*>(srcdest)[index];
        vals[index] = index;
    }
}
__global__ void fillKVPairs_inverted(uint32_t* keys, uint32_t* vals, const unsigned int* srcdest, unsigned int count) {
    unsigned int index = (blockIdx.x * blockDim.x) + threadIdx.x;
    if (index < count) {
        // To subsort by destination too, we treat the pair of uint32 as a uint64
        // To invert we must switch the order of the contained uint32's
        keys[index * 2 + 0] = srcdest[index * 2 + 1];
        keys[index * 2 + 1] = srcdest[index * 2 + 0];
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
void CUDAEnvironmentDirectedGraphBuffers::syncDevice_async(detail::CUDAScatter& scatter, const unsigned int streamID, const cudaStream_t stream) {
    // Copy variable buffers to device
    if (vertex_count) {
        for (auto& v : graph_description.vertexProperties) {
            auto& vb = vertex_buffers.at(v.first);
            if (vb.ready == Buffer::Host) {
                gpuErrchk(cudaMemcpyAsync(vb.d_ptr, vb.h_ptr, vertex_count * v.second.type_size * v.second.elements, cudaMemcpyHostToDevice, stream));
                vb.ready = Buffer::Both;
            }
        }
    }
    if (edge_count) {
        for (auto& e : graph_description.edgeProperties) {
            auto& eb = edge_buffers.at(e.first);
            if (eb.ready == Buffer::Host) {
                gpuErrchk(cudaMemcpyAsync(eb.d_ptr, eb.h_ptr, edge_count * e.second.type_size * e.second.elements, cudaMemcpyHostToDevice, stream));
                eb.ready = Buffer::Both;
            }
        }
    }
    if (vertex_count && edge_count && requires_rebuild) {
        {  // Rebuild the CSR/VBM (edgesLeaving())
            // Fill Key/Val Pairs
            int blockSize;  // The launch configurator returned block size
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, fillKVPairs, 32, 0));  // Randomly 32
            int gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            fillKVPairs<<<gridSize, blockSize, 0, stream>>>(d_keys, d_vals, static_cast<unsigned int*>(edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME).d_ptr), edge_count);
            gpuErrchkLaunch();
            // Sort Key/Val Pairs according to src->dest
            auto &cub_temp = scatter.CubTemp(streamID);
            size_t temp_req = 0;
            gpuErrchk(cub::DeviceRadixSort::SortPairs(nullptr, temp_req, d_keys, d_keys_swap, d_vals, d_vals_swap, edge_count, 0, sizeof(uint64_t) * 8, stream));
            cub_temp.resize(temp_req);
            gpuErrchk(cub::DeviceRadixSort::SortPairs(cub_temp.getPtr(), cub_temp.getSize(), d_keys, d_keys_swap, d_vals, d_vals_swap, edge_count, 0, sizeof(uint64_t) * 8, stream));
            // Build PBM (For vertices with edges)
            gpuErrchk(cudaMemset(d_pbm, 0xffffffff, (vertex_count + 1) * sizeof(unsigned int)));
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, findBinStart, 32, 0));  // Randomly 32
            gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            findBinStart<<<gridSize, blockSize, 0, stream>>>(d_pbm, d_keys_swap, edge_count, vertex_count);
            gpuErrchkLaunch();
            // Build PBM (Fill vertices with no edges)
            temp_req = 0;
            gpuErrchk(cub::DeviceScan::InclusiveScan(nullptr, temp_req, ReverseIterator(d_pbm + vertex_count), ReverseIterator(d_pbm_swap + vertex_count), CustomMin(), vertex_count + 1, stream));
            cub_temp.resize(temp_req);
            gpuErrchk(cub::DeviceScan::InclusiveScan(cub_temp.getPtr(), cub_temp.getSize(), ReverseIterator(d_pbm + vertex_count), ReverseIterator(d_pbm_swap + vertex_count), CustomMin(), vertex_count + 1, stream));
            // Sort edge variables
            std::vector<detail::CUDAScatter::ScatterData> sd;
            for (auto &edge : edge_buffers) {
                edge.second.swap();
                sd.push_back(detail::CUDAScatter::ScatterData{edge.second.element_size,  reinterpret_cast<char*>(edge.second.d_ptr_swap),  reinterpret_cast<char*>(edge.second.d_ptr)});
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
            gpuErrchk(cudaStreamSynchronize(stream));  // Not strictly necessary?
            // @TODO When is best time to copy edge buffers back to host after sort?
        }
        {  // Rebuild the CSC/Inverted VBM (edgesJoining())
            int blockSize;  // The launch configurator returned block size
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, fillKVPairs, 32, 0));  // Randomly 32
            int gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            fillKVPairs_inverted<<<gridSize, blockSize, 0, stream>>>(reinterpret_cast<uint32_t*>(d_keys), d_vals, static_cast<unsigned int*>(edge_buffers.at(GRAPH_SOURCE_DEST_VARIABLE_NAME).d_ptr), edge_count);
            gpuErrchkLaunch();
            // Sort Key/Val Pairs according to dest->src
            // Cub temp has already been resized above
            auto& cub_temp = scatter.CubTemp(streamID);
            gpuErrchk(cub::DeviceRadixSort::SortPairs(cub_temp.getPtr(), cub_temp.getSize(), d_keys, d_keys_swap, d_vals, d_vals_swap, edge_count, 0, sizeof(uint64_t) * 8, stream));
            // Build inverted PBM (For vertices with edges)
            gpuErrchk(cudaMemset(d_ipbm, 0xffffffff, (vertex_count + 1) * sizeof(unsigned int)));
            gpuErrchk(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blockSize, findBinStart, 32, 0));  // Randomly 32
            gridSize = (edge_count + blockSize - 1) / blockSize;  // Round up according to array size
            findBinStart<<<gridSize, blockSize, 0, stream>>>(d_ipbm, d_keys_swap, edge_count, vertex_count);
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
    }
}

void CUDAEnvironmentDirectedGraphBuffers::Buffer::updateHostBuffer(size_type edge_count, cudaStream_t stream) const {
    if (ready == Device) {
        gpuErrchk(cudaMemcpyAsync(h_ptr, d_ptr, edge_count * element_size, cudaMemcpyDeviceToHost, stream));
        gpuErrchk(cudaStreamSynchronize(stream));
        ready = Both;
    }
}
}  // namespace detail
}  // namespace flamegpu
