#include <cassert>

#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"


namespace flamegpu_internal {
namespace CUDAScanCompaction {
    /**
    * These will remain unallocated until used
    * They exist so that the correct array can be used with only the stream index known
    */
    __device__ CUDAScanCompactionPtrs ds_configs[MAX_TYPES][MAX_STREAMS];
    /**
    * Host mirror of ds_configs
    */
    CUDAScanCompactionConfig hd_configs[MAX_TYPES][MAX_STREAMS];

}  // namespace CUDAScanCompaction
}  // namespace flamegpu_internal


__host__ void CUDAScanCompactionConfig::free_scan_flag() {
    if (d_ptrs.scan_flag) {
        gpuErrchk(cudaFree(d_ptrs.scan_flag));
    }
    if (d_ptrs.position) {
        gpuErrchk(cudaFree(d_ptrs.position));
    }
}

__host__ void CUDAScanCompactionConfig::zero_scan_flag() {
    if (d_ptrs.position) {
        gpuErrchk(cudaMemset(d_ptrs.position, 0, scan_flag_len * sizeof(unsigned int)));
    }
    if (d_ptrs.scan_flag) {
        gpuErrchk(cudaMemset(d_ptrs.scan_flag, 0, scan_flag_len * sizeof(unsigned int)));
    }
}

__host__ void CUDAScanCompactionConfig::resize_scan_flag(const unsigned int& count) {
    if (count + 1 > scan_flag_len) {
        free_scan_flag();
        gpuErrchk(cudaMalloc(&d_ptrs.scan_flag, (count + 1) * sizeof(unsigned int)));  // +1 so we can get the total from the scan
        gpuErrchk(cudaMalloc(&d_ptrs.position, (count + 1) * sizeof(unsigned int)));  // +1 so we can get the total from the scan
        // Calculate offset of this object from start of array, then divide by size of this object, and multiply by size of device object
        ptrdiff_t output_dist = (std::distance(reinterpret_cast<char*>(flamegpu_internal::CUDAScanCompaction::hd_configs), reinterpret_cast<char*>(this)) / sizeof(CUDAScanCompactionConfig)) * sizeof(CUDAScanCompactionPtrs);
        gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::CUDAScanCompaction::ds_configs, &this->d_ptrs, sizeof(CUDAScanCompactionPtrs), output_dist));
        scan_flag_len = count + 1;
    }
}


void flamegpu_internal::CUDAScanCompaction::resize(const unsigned int& newCount, const flamegpu_internal::CUDAScanCompaction::Type& type, const unsigned int& streamId) {
    assert(streamId < MAX_STREAMS);
    assert(type < MAX_TYPES);
    hd_configs[type][streamId].resize_scan_flag(newCount);
}

void flamegpu_internal::CUDAScanCompaction::zero(const flamegpu_internal::CUDAScanCompaction::Type& type, const unsigned int& streamId) {
    assert(streamId < MAX_STREAMS);
    assert(type < MAX_TYPES);
    hd_configs[type][streamId].zero_scan_flag();
}


