#include "flamegpu/util/cleanup.h"

#include <cuda_runtime.h>
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"
#include "flamegpu/util/detail/JitifyCache.h"

namespace flamegpu {
namespace util {

void cleanup() {
    // Reset all cuda devices for memcheck / profiling purposes.
    int devices = 0;
    gpuErrchk(cudaGetDeviceCount(&devices));
    for (int device = 0; device < devices; ++device) {
        gpuErrchk(cudaSetDevice(device));
        gpuErrchk(cudaDeviceReset());
    }
}

void clearRTCDiskCache() {
    detail::JitifyCache::clearDiskCache();
}

}  // namespace util
}  // namespace flamegpu
