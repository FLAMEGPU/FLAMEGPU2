#include "flamegpu/util/cleanup.h"

#include <cuda_runtime.h>
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"
#include "flamegpu/detail/JitifyCache.h"

namespace flamegpu {
namespace util {

void cleanup() {
    int originalDevice = 0;
    gpuErrchk(cudaGetDevice(&originalDevice));
    // Reset all cuda devices for memcheck / profiling purposes.
    int devices = 0;
    gpuErrchk(cudaGetDeviceCount(&devices));
    // @todo - this would be better to be only devices touched by flamegpu since the last call to cleanup.
    for (int device = 0; device < devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaDeviceReset());
    }
    // resume the old device, but do not create a new context via reset or memsets
    gpuErrchk(cudaSetDevice(originalDevice));
}

void clearRTCDiskCache() {
    detail::JitifyCache::clearDiskCache();
}

}  // namespace util
}  // namespace flamegpu
