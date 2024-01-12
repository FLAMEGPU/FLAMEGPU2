#ifndef INCLUDE_FLAMEGPU_UTIL_CLEANUP_H_
#define INCLUDE_FLAMEGPU_UTIL_CLEANUP_H_

/*
 * Provides a utility method to cleanup after flamegpu. Currently for the only implementation (CUDA) this resets all devices.
 */

namespace flamegpu {
namespace util {

/**
 * Method to cleanup / finalise use of FLAMEGPU (and MPI). 
 * For CUDA implementations, this resets all CUDA devices in the current system, to ensure that CUDA tools such as cuda-memcheck, compute-sanitizer and Nsight Compute.
 * 
 * This method should ideally be called as the final method prior to an `exit` or the `return` of the main method, as it is costly and can invalidate device memory allocations, potentially breaking application state.
 */
void cleanup();
/**
 * Clear the cache of compiled RTC agent functions
 * This is normally located within the operating system's temporary directory, and persists between executions of FLAMEGPU
 * e.g. %temp%/flamegpu/jitifycache
 */
void clearRTCDiskCache();

}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_CLEANUP_H_
