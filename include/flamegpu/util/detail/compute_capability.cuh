#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_COMPUTE_CAPABILITY_CUH_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_COMPUTE_CAPABILITY_CUH_

#include <vector>
#include <string>
#include <set>

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

/**
 * Get the comptue capabilities supported by the linked NVRTC, irrespective of whether FLAMEGPU was configured for that architecture.
 * CUDA 11.2 or greater provides methods to make this dynamic. Older versions of CUDA are hardcoded (11.1, 11.0 and 10.x only).
 * @return vector of compute capability integers ((major * 10) + minor) in ascending order 
 */
std::vector<int> getNVRTCSupportedComputeCapabilties();


/**
 * Get the best matching compute capability from a vector of compute capabililties in ascending order
 * I.e. get the maximum CC value which is less than or equal to the target CC
 *
 * This method has been separated from JitifyCache::compileKernel so that it can be tested generically, without having to write tests which are relative to the linked nvrtc and/or the current device.
 * 
 * @param target compute capability to find the best match for
 * @param archictectures a vector of architectures in ascending order
 * @return the best compute capability to use (the largest value LE target), or 0 if none are appropriate.
 */
int selectAppropraiteComputeCapability(const int target, const std::vector<int>& architectures);

/**
 * Get the device name reported by CUDA runtime API
 * @param deviceIndex the index of the device to be queried
 * @return string representing the device name E.g. "NVIDIA GeForce RTX3080"
 */
const std::string getDeviceName(int deviceIndex);

/**
 * Get the device names reported by CUDA runtime API
 * @param set<int> of device id's to be queried
 * @return comma seperated string of device names E.g. "NVIDIA GeForce RTX3080, NVIDIAGe Force RTX3070"
 */
const std::string getDeviceNames(std::set<int> devices);
}  // namespace compute_capability
}  // namespace detail
}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_COMPUTE_CAPABILITY_CUH_
