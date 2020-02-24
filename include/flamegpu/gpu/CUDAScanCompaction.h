#ifndef INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_

#include <cuda_runtime.h>
#include "CUDAErrorChecking.h"
#include <cassert>

/**
 * PLEASE NOTE: This implementation currently assumes there is only one instance of CUDAAgentModel executing at once
 * PLEASE NOTE: There is not currently a mechanism to release these (could trigger something via CUDAAgentModel destructor)
 */

/**
 * Could make this cleaner with an array of a nested struct and enums for access, rather than copy paste/rename
 */
struct CUDAScanCompactionPtrs {
    /**
    * Array to mark whether an item is to be retained
    */
    unsigned int *scan_flag = nullptr;
    /**
    * scan_flag is exclusive summed into this array if messages are optional
    */
    unsigned int *position = nullptr;
};
struct CUDAScanCompactionConfig {
    CUDAScanCompactionConfig()
        : scan_flag_len(0)
        , hd_cub_temp(nullptr)
        , cub_temp_size(0)
        , cub_temp_size_max_list_size(0)
    { }
    unsigned int scan_flag_len = 0;

    CUDAScanCompactionPtrs d_ptrs;

    void *hd_cub_temp = nullptr;
    size_t cub_temp_size = 0;
    unsigned int cub_temp_size_max_list_size = 0;

    __host__ void free_scan_flag() {
        if (d_ptrs.scan_flag) {
            gpuErrchk(cudaFree(d_ptrs.scan_flag));
        }
        if (d_ptrs.position) {
            gpuErrchk(cudaFree(d_ptrs.position));
        }
    }
    __host__ void resize_scan_flag(const unsigned int &count);
    __host__ void zero_scan_flag() {
        if (d_ptrs.position) {
            gpuErrchk(cudaMemset(d_ptrs.position, 0, scan_flag_len * sizeof(unsigned int)));
        }
        if (d_ptrs.scan_flag) {
            gpuErrchk(cudaMemset(d_ptrs.scan_flag, 0, scan_flag_len * sizeof(unsigned int)));
        }
    }
};
typedef CUDAScanCompactionPtrs CUDASCPtrs;  // Shorthand
typedef CUDAScanCompactionConfig CUDASCConfig;  // Shorthand

namespace flamegpu_internal {
namespace CUDAScanCompaction {
    /**
     * Try and be a bit more dynamic by using 2d array here
     */
    enum Type : unsigned int {
        MESSAGE_OUTPUT = 0,
        AGENT_DEATH = 1,
        AGENT_OUTPUT = 2
    };
    /**
    * Number of valid values in Type
    */
    static const unsigned int MAX_TYPES = 3;
    /** 
     * As of Compute Cability 7.5; 128 is the max concurrent streams
     */
    static const unsigned int MAX_STREAMS = 128;
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    extern __device__ CUDAScanCompactionPtrs ds_configs[MAX_TYPES][MAX_STREAMS];
    /**
     * Host mirror of ds_configs
     */
    extern CUDAScanCompactionConfig hd_configs[MAX_TYPES][MAX_STREAMS];

    inline void resize(const unsigned int &newCount, const Type &type, const unsigned int &streamId) {
        assert(streamId < MAX_STREAMS);
        assert(type < MAX_TYPES);
        hd_configs[type][streamId].resize_scan_flag(newCount);
    }
    inline void zero(const Type &type, const unsigned int &streamId) {
        assert(streamId < MAX_STREAMS);
        assert(type < MAX_TYPES);
        hd_configs[type][streamId].zero_scan_flag();
    }
}  // namespace CUDAScanCompaction
}  // namespace flamegpu_internal


__host__ inline void CUDAScanCompactionConfig::resize_scan_flag(const unsigned int &count) {
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
#endif  // INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_
