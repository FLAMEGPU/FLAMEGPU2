#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_COMPUTE_CAPABILITY_CUH_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_COMPUTE_CAPABILITY_CUH_

#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"

namespace flamegpu {
namespace util {
namespace detail {
namespace compute_capability {

/**
 * get the compute capability for a device
 * @param deviceIndex the index of the device to be queried
 * @return integer value representing the compute capability, i.e 70 for SM_70
 */
int getComputeCapability(int deviceIndex);

/**
 * Get the minimum compute capability which this file was compiled for.
 * Specified via the MIN_ARCH macro, as __CUDA_ARCH__ is only defined for device compilation.
 */
int minimumCompiledComputeCapability();

/**
 * Check that the current executable has been built with a low enough compute capability for the current device.
 * This assumes JIT support is enabled for future (major) architectures.
 * If the compile time flag MIN_ARCH was not specified, no decision can be made so it is assumed to be successful.
 * @param deviceIndex the index of the device to be checked.
 * @return boolean indicating if the executable can run on the specified device.
 */
bool checkComputeCapability(int deviceIndex);

}  // namespace compute_capability
}  // namespace detail
}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_COMPUTE_CAPABILITY_CUH_
