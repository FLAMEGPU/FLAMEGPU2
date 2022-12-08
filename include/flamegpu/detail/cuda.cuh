#ifndef INCLUDE_FLAMEGPU_DETAIL_CUDA_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_CUDA_CUH_

/**
 * Collection of cuda related utility methods for internal use.
 * Mostly to allow for graceful handling of device resets when cuda is called by dtors
 * @todo - we should check for unified addressing support prior to use of cudaPointerGetAttributes, but it should be available for all valide flamegpu targets (x64 linux and windows). Tegra's might be an edge case.
 */
#include <cuda_runtime.h>
#include <cuda.h>
#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {
namespace detail {
namespace cuda {

/**
 * Wrapped cudaFree which checks that the pointer is a valid device pointer in the current CUDA context prior to deallocation.
 * This also prevents double free errors from being raised, as the pointer attributes in that case are the same as a reset device, would need to check the primary context too?
 * @param devPtr device pointer to memory to free
 * @return forward the cuda error status from the inner cudaFree call
 */
inline cudaError_t cudaFree(void* devPtr) {
    cudaError_t status = cudaSuccess;
    // Check the pointer attribtues to detect if it is a valid ptr for the current context.
    // @todo - version which checks the device ordinal is a match for the active context too, potenitally flip-flopping the device.
    cudaPointerAttributes attributes = {};
    status = cudaPointerGetAttributes(&attributes, devPtr);
    // valid device pointers have a type of cudaMemoryTypeDevice (2), or we could check the device is non negative (and matching the current device index?), or the devicePointer will be non null.
    if (status == cudaSuccess && attributes.type == cudaMemoryTypeDevice) {
        status = ::cudaFree(devPtr);
        // Forward any status on
        return status;
    }
    // If the pointer attribtues were not correct, return cudaSuccess to avoid bad error checking.
    return cudaSuccess;
}

/**
 * Wrapped cudaFreeHost which checks that the pointer is a valid page-locked pointer prior to freeing.
 * This also prevents double free errors from being raised, as the pointer attributes in that case are the same as a reset device, would need to check the primary context too?
 * @param devPtr pointer to memory to free
 * @return forward the cuda error status from the inner cudaFreeHost call
 */
inline cudaError_t cudaFreeHost(void* devPtr) {
    cudaError_t status = cudaSuccess;
    // Check the pointer attribtues to detect if it is a valid ptr for the current context.
    // @todo - version which checks the device ordinal is a match for the active context too, potenitally flip-flopping the device.
    cudaPointerAttributes attributes = {};
    status = cudaPointerGetAttributes(&attributes, devPtr);
    // valid pointers allocated using cudaMallocHost have a type of cudaMemoryTypeHost
    if (status == cudaSuccess && attributes.type == cudaMemoryTypeHost) {
        status = ::cudaFreeHost(devPtr);
        // Forward on any cuda errors returned.
        return status;
    }
    // If the pointer attribtues were not correct, return cudaSuccess to avoid bad error checking.
    return cudaSuccess;
}

/**
 * Use the cuda Driver API to check that the primary context for a given device (the device which is shared with the runtime API) is active.
 * This method will silenty absorb any cuda driver api errors, as it's intended use case is when the cuda driver may be uninitlaised / shutdown.
 * @param ordinal device index to query for
 * @return bool indicating if primary context for the given device is active or not
 */
inline bool cuDevicePrimaryContextIsActive(int ordinal) {
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
}

}  // namespace cuda
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_CUDA_CUH_
