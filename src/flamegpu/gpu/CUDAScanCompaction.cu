#include "flamegpu/gpu/CUDAScanCompaction.h"

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
