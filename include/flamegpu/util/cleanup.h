#ifndef INCLUDE_FLAMEGPU_UTIL_CLEANUP_H_
#define INCLUDE_FLAMEGPU_UTIL_CLEANUP_H_

/**
 * Prodvides a utility method to cleanup after flamegpu. Currently for the only implementation (CUDA) this resets all devices.
 */

namespace flamegpu {
namespace util {

/**
 * Method to cleanup / finalise use of FLAMEGPU. 
 * For CUDA implementations, this resets all CUDA devices in the current system, to ensure that CUDA tools such as cuda-memcheck, compute-sanitizer and Nsight Compute.
 * 
 * This method should ideally be called as the final method prior to an `exit` or the `return` of the main method, as it is costly and can invalidate device memory allocations, potentially breaking application state.
 */
void cleanup();

}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_CLEANUP_H_
