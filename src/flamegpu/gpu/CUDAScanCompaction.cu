#include "flamegpu/gpu/CUDAScanCompaction.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"


namespace flamegpu_internal {
namespace CUDAScanCompaction {
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    __device__ CUDAScanCompactionPtrs ds_agent_configs[MAX_STREAMS];
    /**
     * Host mirror of ds_agent_configs
     */
    CUDAScanCompactionConfig hd_agent_configs[MAX_STREAMS] = { };  // {} Should trigger default init
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    __device__ CUDAScanCompactionPtrs ds_message_configs[MAX_STREAMS];
    /**
     * Host mirror of ds_message_configs
     */
    CUDAScanCompactionConfig hd_message_configs[MAX_STREAMS];
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
        // Blindly calculate our stream ID, based on the offset between 'this' and 'hd_stream_configs[0]'
        ptrdiff_t actor_stream_id = std::distance(flamegpu_internal::CUDAScanCompaction::hd_agent_configs, this);
        ptrdiff_t message_stream_id = std::distance(flamegpu_internal::CUDAScanCompaction::hd_message_configs, this);
        if (actor_stream_id >= 0 && actor_stream_id < flamegpu_internal::CUDAScanCompaction::MAX_STREAMS) {
            gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::CUDAScanCompaction::ds_agent_configs, &this->d_ptrs, sizeof(CUDAScanCompactionPtrs), actor_stream_id * sizeof(CUDAScanCompactionPtrs)));
        }
        else if (message_stream_id >= 0 && message_stream_id < flamegpu_internal::CUDAScanCompaction::MAX_STREAMS) {
            gpuErrchk(cudaMemcpyToSymbol(flamegpu_internal::CUDAScanCompaction::ds_message_configs, &this->d_ptrs, sizeof(CUDAScanCompactionConfig), message_stream_id * sizeof(CUDAScanCompactionConfig)));
        }
        else {
            assert(false);  // This is being called on one that isn't part of the array????
        }
        scan_flag_len = count + 1;
    }
}