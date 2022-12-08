#include <cuda_runtime.h>

#include <vector>
#include "flamegpu/detail/cuda.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

#include "gtest/gtest.h"
namespace flamegpu {


// Test that wrapped cudaFree works.
TEST(TestUtilDetailCuda, cudaFree) {
    int * d_ptr = nullptr;
    cudaError_t status = cudaSuccess;
    // manually allocate a device pointer
    gpuErrchk(cudaMalloc(&d_ptr, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    cudaPointerAttributes attributes = {};
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
    // call the wrapped cuda free method
    status = detail::cuda::cudaFree(d_ptr);
    // It should not have thrown any cuda errors in normal use.
    EXPECT_EQ(status, cudaSuccess);
    // The pointer will still have a non nullptr value, but it will no longer be a valid device ptr.
    EXPECT_NE(d_ptr, nullptr);
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, cudaMemoryTypeUnregistered);
    // Try a double free.
    status = detail::cuda::cudaFree(d_ptr);
    // This will appear to succeed (a double free is identical to a device reset then free according from cudaPointerGetAttributes' perspective), which is a difference from actual cudaFree which would return cudaErrorInvalidValue.
    EXPECT_EQ(status, cudaSuccess);
    // reset the ptr
    d_ptr = nullptr;
    // Allocate the pointer again
    gpuErrchk(cudaMalloc(&d_ptr, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    attributes = {};
    gpuErrchk(cudaPointerGetAttributes(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, cudaMemoryTypeDevice);
    // Trigger a device reset
    cudaDeviceReset();
    // Attempt to free the ptr, this method should claim all things are fine (as the dev ptr has implicitly been freed)
    status = detail::cuda::cudaFree(d_ptr);
    EXPECT_EQ(status, cudaSuccess);
}

// Test that the wrapped cudaFreeHost works.
TEST(TestUtilDetailCuda, cudaFreeHost) {
    int * p_ptr = nullptr;
    cudaError_t status = cudaSuccess;
    // manually allocate a page-locked host pointer
    gpuErrchk(cudaMallocHost(&p_ptr, sizeof(int)));
    // Validate that the ptr is a valid page-locked host pointer
    cudaPointerAttributes attributes = {};
    gpuErrchk(cudaPointerGetAttributes(&attributes, p_ptr));
    // this appears to return cudaMemoryTypeHost, even though it should return cudaMemoryTypeHost
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    // call the wrapped cuda free method
    status = detail::cuda::cudaFreeHost(p_ptr);
    // It should not have thrown any cuda errors in normal use.
    EXPECT_EQ(status, cudaSuccess);
    // The pointer will still have a non nullptr value, but it will no longer be a valid page-locked ptr.
    EXPECT_NE(p_ptr, nullptr);
    gpuErrchk(cudaPointerGetAttributes(&attributes, p_ptr));
    EXPECT_EQ(attributes.type, cudaMemoryTypeUnregistered);

    // Try a double free.
    status = detail::cuda::cudaFreeHost(p_ptr);
    // This will appear to succeed (a double free is identical to a device reset then free according from cudaPointerGetAttributes' perspective), which is a difference from actual cudaFreeHost which would return cudaErrorInvalidValue.
    EXPECT_EQ(status, cudaSuccess);
    // reset the ptr
    p_ptr = nullptr;
    // Allocate the pointer again
    gpuErrchk(cudaMallocHost(&p_ptr, sizeof(int)));
    // Validate that the ptr is a valid page-locked host pointer
    attributes = {};
    gpuErrchk(cudaPointerGetAttributes(&attributes, p_ptr));
    // this appears to return cudaMemoryTypeHost, even though it should return cudaMemoryTypeHost
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);
    // Trigger a device reset
    cudaDeviceReset();
    // Attempt to free the ptr, this method should claim all things are fine (as the dev ptr has implicitly been freed)
    status = detail::cuda::cudaFreeHost(p_ptr);
    EXPECT_EQ(status, cudaSuccess);
}

// Test that getting the primary context works, Difficult to trigger failure cases for this method, so coverage is subpar.
TEST(TestUtilDetailCuda, cuDevicePrimaryContextIsActive) {
    // Make sure device 0 is active for this test.
    gpuErrchk(cudaSetDevice(0));
    // Initialise a cudaContext, incase it somehow hasn't been already.
    gpuErrchk(cudaFree(0));
    // check if the primary context is active or not for device 0, it shoudl be.
    bool isActive = false;
    isActive = detail::cuda::cuDevicePrimaryContextIsActive(0);
    EXPECT_EQ(isActive, true);
    // Call device reset and check again without establishing a new context, it should not be active.
    gpuErrchk(cudaDeviceReset());
    isActive = detail::cuda::cuDevicePrimaryContextIsActive(0);
    EXPECT_EQ(isActive, false);
    // Check that exceptions will be raised correctly when passing bad device ordinals.
    // Expect an exception if the ordinal is negative
    EXPECT_THROW(detail::cuda::cuDevicePrimaryContextIsActive(-1), exception::InvalidCUDAdevice);
    // First grab the device count, to check for exceptions when the device ordinal is too big.
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        // Expect an exception if the ordinal is too big.
        EXPECT_THROW(detail::cuda::cuDevicePrimaryContextIsActive(deviceCount), exception::InvalidCUDAdevice);
    }
}

}  // namespace flamegpu
