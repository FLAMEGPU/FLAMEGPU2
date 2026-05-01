#include <cassert>

#include "flamegpu/simulation/detail/CUDAScanCompaction.h"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"
#include "flamegpu/simulation/CUDASimulation.h"
#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/gpu/types.hpp"
#include "flamegpu/detail/cuda.cuh"

namespace flamegpu {
namespace detail {

/**
 * CUDAScanCompaction methods
 */
void CUDAScanCompaction::resize(const unsigned int newCount, const Type& type, const unsigned int streamId) {
    assert(streamId < MAX_STREAMS);
    assert(type < MAX_TYPES);
    configs[type][streamId].resize_scan_flag(newCount);
}

void CUDAScanCompaction::zero_async(const Type& type, flamegpu::detail::gpu::Stream_t stream, unsigned int streamId) {
    assert(streamId < MAX_STREAMS);
    assert(type < MAX_TYPES);
    configs[type][streamId].zero_scan_flag_async(stream);
}

const CUDAScanCompactionConfig &CUDAScanCompaction::getConfig(const Type& type, const unsigned int streamId) {
    return configs[type][streamId];
}
CUDAScanCompactionConfig &CUDAScanCompaction::Config(const Type& type, const unsigned int streamId) {
    return configs[type][streamId];
}
/**
 *
 */
CUDAScanCompactionConfig::~CUDAScanCompactionConfig() {
    free_scan_flag();
}
void CUDAScanCompactionConfig::free_scan_flag() {
    if (d_ptrs.scan_flag) {
        flamegpu::detail::gpuCheck(flamegpu::detail::cuda::cudaFree(d_ptrs.scan_flag));
        d_ptrs.scan_flag = nullptr;
    }
    if (d_ptrs.position) {
        flamegpu::detail::gpuCheck(flamegpu::detail::cuda::cudaFree(d_ptrs.position));
        d_ptrs.position = nullptr;
    }
}

void CUDAScanCompactionConfig::zero_scan_flag_async(flamegpu::detail::gpu::Stream_t stream) {
    if (d_ptrs.position) {
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(MemsetAsync)(d_ptrs.position, 0, scan_flag_len * sizeof(unsigned int), stream));
    }
    if (d_ptrs.scan_flag) {
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(MemsetAsync)(d_ptrs.scan_flag, 0, scan_flag_len * sizeof(unsigned int), stream));
    }
}

void CUDAScanCompactionConfig::resize_scan_flag(const unsigned int count) {
    if (count + 1 > scan_flag_len) {
        free_scan_flag();
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(Malloc)(&d_ptrs.scan_flag, (count + 1) * sizeof(unsigned int)));  // +1 so we can get the total from the scan
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(Malloc)(&d_ptrs.position, (count + 1) * sizeof(unsigned int)));  // +1 so we can get the total from the scan
        scan_flag_len = count + 1;
    }
}

}  // namespace detail
}  // namespace flamegpu
