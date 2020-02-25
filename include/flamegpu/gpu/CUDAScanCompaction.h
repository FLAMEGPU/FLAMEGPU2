#ifndef INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_

//#include <cuda_runtime.h>
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

    __host__ void free_scan_flag();
    __host__ void resize_scan_flag(const unsigned int &count);
    __host__ void zero_scan_flag();
};
typedef CUDAScanCompactionPtrs CUDASCPtrs;  // Shorthand
typedef CUDAScanCompactionConfig CUDASCConfig;  // Shorthand

namespace flamegpu_internal {
namespace CUDAScanCompaction {
    /** 
     * As of Compute Cability 7.5; 128 is the max concurrent streams
     */
    static const unsigned int MAX_STREAMS = 128;
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    extern __device__ CUDAScanCompactionPtrs ds_agent_configs[MAX_STREAMS];
    /**
     * Host mirror of ds_agent_configs
     */
    extern CUDAScanCompactionConfig hd_agent_configs[MAX_STREAMS];
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    extern __device__ CUDAScanCompactionPtrs ds_message_configs[MAX_STREAMS];
    /**
     * Host mirror of ds_message_configs
     */
    extern CUDAScanCompactionConfig hd_message_configs[MAX_STREAMS];

    inline void resizeAgents(const unsigned int &newCount, const unsigned int &streamId) {
        assert(streamId < MAX_STREAMS);
        hd_agent_configs[streamId].resize_scan_flag(newCount);
    }
    inline void resizeMessages(const unsigned int &newCount, const unsigned int &streamId) {
        assert(streamId < MAX_STREAMS);
        hd_message_configs[streamId].resize_scan_flag(newCount);
    }
    inline void zeroAgents(const unsigned int &streamId) {
        assert(streamId < MAX_STREAMS);
        hd_agent_configs[streamId].zero_scan_flag();
    }
    inline void zeroMessages(const unsigned int &streamId) {
        assert(streamId < MAX_STREAMS);
        hd_message_configs[streamId].zero_scan_flag();
    }
}  // namespace CUDAScanCompaction
}  // namespace flamegpu_internal



#endif  // INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_
