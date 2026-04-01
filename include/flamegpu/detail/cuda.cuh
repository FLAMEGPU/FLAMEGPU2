#ifndef INCLUDE_FLAMEGPU_DETAIL_CUDA_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_CUDA_CUH_

#ifdef FLAMEGPU_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#define FLAMEGPU_GPU_RUNTIME_PREFIX cuda
#define FLAMEGPU_GPU_DRIVER_PREFIX cu
#define FLAMEGPU_GPU_RUNTIME_PREFIX_UPPER CUDA
#define FLAMEGPU_GPU_DRIVER_PREFIX_UPPER CU
#endif

#ifdef FLAMEGPU_USE_HIP
#include <hip/hip_runtime.h>
#define FLAMEGPU_GPU_RUNTIME_PREFIX hip
#define FLAMEGPU_GPU_DRIVER_PREFIX hip
#define FLAMEGPU_GPU_RUNTIME_PREFIX_UPPER HIP
#define FLAMEGPU_GPU_DRIVER_PREFIX_UPPER HIP
#endif

#ifndef __CUDACC_RTC__
#include <limits>
#include <cstdint>
#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {
namespace detail {

/**
 * Collection of cuda related utility methods for internal use.
 * Mostly to allow for graceful handling of device resets when cuda is called by dtors
 * @todo - we should check for unified addressing support prior to use of cudaPointerGetAttributes, but it should be available for all valid flamegpu targets (x64 linux and windows). Tegra's might be an edge case.
 
 Todo:
    - Split this into several headers, inside and outside detail:
    - flamegpu/gpu/macros.h
      - macros for the gpu abstraction layer.
    - flamegpu/gpu/types.h
      - typedefs for the gpu abstraction layer, that are public-facing.
    - flamegpu/gpu/detail/gpu.cuh/cu, i.e wrapped cudaFree (which will aslo be warpped hip free)
    - flamegpu/gpu/detail/...
 */
namespace cuda {

// Define macros for the lazy cuda/hip abstraction layer. This requires multiple macros for string manipulation
#define FLAMEGPU_GPU_CONCAT_INNER(a, b) a ## b
#define FLAMEGPU_GPU_CONCAT(a, b) FLAMEGPU_GPU_CONCAT_INNER(a, b)

#if defined(FLAMEGPU_GPU_RUNTIME_PREFIX) && defined(FLAMEGPU_GPU_DRIVER_PREFIX)
#define FLAMEGPU_GPU_RUNTIME_SYMBOL(STMT) FLAMEGPU_GPU_CONCAT(FLAMEGPU_GPU_RUNTIME_PREFIX, STMT)
#define FLAMEGPU_GPU_DRIVER_SYMBOL(STMT) FLAMEGPU_GPU_CONCAT(FLAMEGPU_GPU_DRIVER_PREFIX, STMT)
#else
#define FLAMEGPU_GPU_RUNTIME_SYMBOL(STMT) STMT
#define FLAMEGPU_GPU_DRIVER_SYMBOL(STMT) STMT
#error "CUDA or HIP must be enabled"
#endif

// Using statement for cuda/hip error_t, which is part of the private API?
// Todo: Should this just use the macro instead?
// Todo: Move type definitions and macros for this to a separate header for lighter includes? flamegpu/gpu/types.h or similar, and then flamegpu/gpu/macros.h and flamegpu/gpu/
// Should this actually be in detail?
#if defined(FLAMEGPU_USE_CUDA)
using Error_t = cudaError_t;
#elif defined(FLAMEGPU_USE_HIP)
using Error_t = hipError_t;
#else
// naked struct for intellisense, this should never occur for actual compilation
typedef struct error* Error_t;
#endif

// Using statement for cuda/hip streams, which are part of the public API
// Todo: should this just use the macro instead? Should this actually be in detail?
#if defined(FLAMEGPU_USE_CUDA)
using Stream_t = cudaStream_t;
#elif defined(FLAMEGPU_USE_HIP)
using Stream_t = hipStream_t;
#else
// naked struct for intellisense, this should never occur for actual compilation
typedef struct stream* Stream_t;
#endif


// pointerAttributes is _t in hip :(
#if defined(FLAMEGPU_USE_CUDA)
using PointerAttributes_t = cudaPointerAttributes;
#elif defined(FLAMEGPU_USE_HIP)
using PointerAttributes_t = hipPointerAttribute_t;
#else
// naked struct for intellisense, this should never occur for actual compilation
typedef struct pointerAttribtues* PointerAttributes_t;
#endif

/**
 * Wrapped cudaFree which checks that the pointer is a valid device pointer in the current CUDA context prior to deallocation.
 * This also prevents double free errors from being raised, as the pointer attributes in that case are the same as a reset device, would need to check the primary context too?
 * @param devPtr device pointer to memory to free
 * @return forward the cuda error status from the inner cudaFree call
 */
inline Error_t cudaFree(void* devPtr) {
    Error_t status = FLAMEGPU_GPU_RUNTIME_SYMBOL(Success);
    // Check the pointer attributes to detect if it is a valid ptr for the current context.
    // @todo - version which checks the device ordinal is a match for the active context too, potentially flip-flopping the device.
    PointerAttributes_t attributes = {};
    status = FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, devPtr);
    // valid device pointers have a type of cudaMemoryTypeDevice (2), or we could check the device is non negative (and matching the current device index?), or the devicePointer will be non null.
    if (status == FLAMEGPU_GPU_RUNTIME_SYMBOL(Success) && attributes.type == FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeDevice)) {
        status = ::FLAMEGPU_GPU_RUNTIME_SYMBOL(Free)(devPtr);
        // Forward any status on
        return status;
    }
    // If the pointer attributes were not correct, return FLAMEGPU_GPU_RUNTIME_SYMBOL(Success) to avoid bad error checking.
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(Success);
}

/**
 * Wrapped cudaFreeHost which checks that the pointer is a valid page-locked pointer prior to freeing.
 * This also prevents double free errors from being raised, as the pointer attributes in that case are the same as a reset device, would need to check the primary context too?
 * @param devPtr pointer to memory to free
 * @return forward the cuda error status from the inner cudaFreeHost call
 */
inline Error_t cudaFreeHost(void* devPtr) {
    Error_t status = FLAMEGPU_GPU_RUNTIME_SYMBOL(Success);
    // Check the pointer attributes to detect if it is a valid ptr for the current context.
    // @todo - version which checks the device ordinal is a match for the active context too, potentially flip-flopping the device.
    PointerAttributes_t attributes = {};
    status = FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, devPtr);
    // valid pointers allocated using cudaHostAlloc have a type of cudaMemoryTypeHost
    if (status == FLAMEGPU_GPU_RUNTIME_SYMBOL(Success) && attributes.type == FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost)) {
        status = ::FLAMEGPU_GPU_RUNTIME_SYMBOL(FreeHost)(devPtr);
        // Forward on any cuda errors returned.
        return status;
    }
    // If the pointer attributes were not correct, return FLAMEGPU_GPU_RUNTIME_SYMBOL(Success) to avoid bad error checking.
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(Success);
}

/**
 * Use the cuda Driver API to check that the primary context for a given device (the device which is shared with the runtime API) is active.
 * This method will silently absorb any cuda driver api errors, as it's intended use case is when the cuda driver may be uninitialised / shutdown.
 * @param ordinal device index to query for
 * @return bool indicating if primary context for the given device is active or not
 */
inline bool cuDevicePrimaryContextIsActive(int ordinal) {
#ifdef FLAMEGPU_USE_CUDA
    // Throw an exception if a negative device ordinal is passed
    if (ordinal < 0) {
        THROW exception::InvalidCUDAdevice("CUDA Device ordinals must be non-negative integers, in detail::cuda::cuDevicePrimaryContextIsActive()");
    }

    int deviceCount = 0;
    CUresult cuErr = CUDA_SUCCESS;
    // Get the device count, possible errors are all about bad context / state  deinitialisation, so eat those silently.
    cuErr = cuDeviceGetCount(&deviceCount);
    if (cuErr == CUDA_SUCCESS) {
        // If the device count is 0, throw.
        if (deviceCount == 0) {
            THROW exception::InvalidCUDAdevice("Error no CUDA devices found!, in detail::cuda::cuDevicePrimaryContextIsActive()");
        }
        // If the ordinal is invalid, throw
        if (ordinal >= deviceCount) {
            THROW exception::InvalidCUDAdevice("Requested CUDA device %d is not valid, only %d CUDA devices available!, in detail::cuda::cuDevicePrimaryContextIsActive()", ordinal, deviceCount);
        }
        // Get the CUdevice handle, silently dismissing any cuErrors as they are falsey
        CUdevice deviceHandle;
        cuErr = cuDeviceGet(&deviceHandle, ordinal);
        if (cuErr == CUDA_SUCCESS) {
            // Get the status of the primary context, again silently treating any cuda driver api errors returned as false-y values as they are effectively what we are checking for with this method.
            unsigned int primaryCtxflags = 0;
            int primaryCtxIsActive = false;
            cuErr = cuDevicePrimaryCtxGetState(deviceHandle, &primaryCtxflags, &primaryCtxIsActive);
            if (cuErr == CUDA_SUCCESS) {
                return primaryCtxIsActive;
            }
        }
    }
    // If we could not return the active state, return false.
    return false;
#else
    // Todo: Don't think hip has an equivalent to this? @todo
    return true;
#endif
}

/**
 * use the CUDA 12+ driver api to get the unique id of the current CUDA context, with error checking.
 * This method will silenlty consume any cuda errors, as it is expected to (potentially) be called during CUDA shutdown.
 * @return the unique id for the CUDA context
 */
inline std::uint64_t cuGetCurrentContextUniqueID() {
#ifdef FLAMEGPU_USE_CUDA
    static_assert(sizeof(unsigned long long int) == sizeof(std::uint64_t));  // NOLINT
    CUresult cuErr = CUDA_SUCCESS;
    // Get the handle to the current context
    CUcontext ctx = NULL;
    cuErr = cuCtxGetCurrent(&ctx);
    if (cuErr == CUDA_SUCCESS) {
        // Getand return the unique id
        unsigned long long int ctxid = std::numeric_limits<std::uint64_t>::max();  // NOLINT
        cuErr = cuCtxGetId(ctx, &ctxid);
        if (cuErr == CUDA_SUCCESS) {
            return static_cast<std::uint64_t>(ctxid);
        }
    }
    return std::numeric_limits<std::uint64_t>::max();
#else
    // Todo: Don't think hip has an equivalent to this? @todo
    return std::numeric_limits<std::uint64_t>::max();
#endif
}

}  // namespace cuda
}  // namespace detail
}  // namespace flamegpu
#endif  // __CUDACC_RTC__
#endif  // INCLUDE_FLAMEGPU_DETAIL_CUDA_CUH_
