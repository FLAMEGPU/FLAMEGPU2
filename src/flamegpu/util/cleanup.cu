#include "flamegpu/util/cleanup.h"

#ifdef FLAMEGPU_ENABLE_MPI
#include <mpi.h>
#endif

#ifdef FLAMEGPU_USE_CUDA
#include <cuda_runtime.h>
#endif

#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"
#include "flamegpu/detail/JitifyCache.h"

namespace flamegpu {
namespace util {

void cleanup() {
#ifdef FLAMEGPU_ENABLE_MPI
    int init_flag = 0;
    int fin_flag = 0;
    // MPI can only be init and finalized once
    MPI_Initialized(&init_flag);
    MPI_Finalized(&fin_flag);
    if (init_flag && !fin_flag) {
        MPI_Finalize();
    }
#endif
    int originalDevice = 0;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDevice)(&originalDevice));
    // Reset all cuda devices for memcheck / profiling purposes.
    int devices = 0;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceCount)(&devices));
    // @todo - this would be better to be only devices touched by flamegpu since the last call to cleanup.
    for (int device = 0; device < devices; ++device) {
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(SetDevice)(device));
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceReset)());
    }
    // resume the old device, but do not create a new context via reset or memsets
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(SetDevice)(originalDevice));
}

#ifdef FLAMEGPU_USE_CUDA
void clearRTCDiskCache() {
    detail::JitifyCache::clearDiskCache();
}
#endif  // FLAMEGPU_USE_CUDA

}  // namespace util
}  // namespace flamegpu
