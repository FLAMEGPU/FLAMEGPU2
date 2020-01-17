#include "flamegpu/gpu/CUDAScanCompaction.h"

namespace flamegpu_internal {
namespace CUDAScanCompaction {
    /**
     * These will remain unallocated until used
     * They exist so that the correct array can be used with only the stream index known
     */
    __device__ CUDAScanCompactionPtrs ds_actor_configs[MAX_STREAMS];
    /**
     * Host mirror of ds_actor_configs
     */
    CUDAScanCompactionConfig hd_actor_configs[MAX_STREAMS];
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
