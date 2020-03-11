#ifndef INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_
#define INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_

//#include <cuda_runtime.h>
//#include "CUDAErrorChecking.h"
//#include <cassert>

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

    void* hd_cub_temp = nullptr;
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

    void resize(const unsigned int& newCount, const Type& type, const unsigned int& streamId);
    void zero(const Type& type, const unsigned int& streamId);
}  // namespace CUDAScanCompaction
}  // namespace flamegpu_internal



#endif  // INCLUDE_FLAMEGPU_GPU_CUDASCANCOMPACTION_H_
