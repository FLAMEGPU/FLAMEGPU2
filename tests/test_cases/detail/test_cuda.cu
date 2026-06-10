#ifdef FLAMEGPU_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"
#include "flamegpu/detail/gpu/types.hpp"
#include "flamegpu/detail/gpu/cuda_hip_dispatch.hpp"
#include "flamegpu/detail/cuda.cuh"

#include "gtest/gtest.h"
namespace flamegpu {

#if FLAMEGPU_USE_HIP
using cudaPointerAttributes = hipPointerAttribute_t;
#endif


// Test that wrapped cudaFree works.
TEST(TestUtilDetailCuda, cudaFree) {
    int * d_ptr = nullptr;
    flamegpu::detail::gpu::Error_t status = flamegpu::detail::gpu::gpuSuccess;
    // manually allocate a device pointer
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuMalloc(&d_ptr, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    cudaPointerAttributes attributes = {};
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuPointerGetAttributes(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, flamegpu::detail::gpu::gpuMemoryTypeDevice);
    // call the wrapped cuda free method
    status = detail::cuda::cudaFree(d_ptr);
    // It should not have thrown any cuda errors in normal use.
    EXPECT_EQ(status, flamegpu::detail::gpu::gpuSuccess);
    // The pointer will still have a non nullptr value, but it will no longer be a valid device ptr.
    EXPECT_NE(d_ptr, nullptr);
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuPointerGetAttributes(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, flamegpu::detail::gpu::gpuMemoryTypeUnregistered);
    // Try a double free.
    status = detail::cuda::cudaFree(d_ptr);
    // This will appear to succeed (a double free is identical to a device reset then free according from flamegpu::detail::gpu::gpuPointerGetAttributes' perspective), which is a difference from actual cudaFree which would return cudaErrorInvalidValue.
    EXPECT_EQ(status, flamegpu::detail::gpu::gpuSuccess);
    // reset the ptr
    d_ptr = nullptr;
    // Allocate the pointer again
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuMalloc(&d_ptr, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    attributes = {};
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuPointerGetAttributes(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, flamegpu::detail::gpu::gpuMemoryTypeDevice);
    // Trigger a device reset
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuDeviceReset());
    // Attempt to free the ptr, this method should claim all things are fine (as the dev ptr has implicitly been freed)
    status = detail::cuda::cudaFree(d_ptr);
    EXPECT_EQ(status, flamegpu::detail::gpu::gpuSuccess);
}

// Test that the wrapped cudaFreeHost works.
TEST(TestUtilDetailCuda, cudaFreeHost) {
    int * p_ptr = nullptr;
    flamegpu::detail::gpu::Error_t status = flamegpu::detail::gpu::gpuSuccess;
    // manually allocate a page-locked host pointer
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuHostAlloc(reinterpret_cast<void**>(&p_ptr), sizeof(int), flamegpu::detail::gpu::gpuHostAllocDefault));
    // Validate that the ptr is a valid page-locked host pointer
    cudaPointerAttributes attributes = {};
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuPointerGetAttributes(&attributes, p_ptr));
    // this appears to return flamegpu::detail::gpu::gpuMemoryTypeHost, even though it should return flamegpu::detail::gpu::gpuMemoryTypeHost
    EXPECT_EQ(attributes.type, flamegpu::detail::gpu::gpuMemoryTypeHost);
    // call the wrapped cuda free method
    status = detail::cuda::cudaFreeHost(p_ptr);
    // It should not have thrown any cuda errors in normal use.
    EXPECT_EQ(status, flamegpu::detail::gpu::gpuSuccess);
    // The pointer will still have a non nullptr value, but it will no longer be a valid page-locked ptr.
    EXPECT_NE(p_ptr, nullptr);
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuPointerGetAttributes(&attributes, p_ptr));
    EXPECT_EQ(attributes.type, flamegpu::detail::gpu::gpuMemoryTypeUnregistered);

    // Try a double free.
    status = detail::cuda::cudaFreeHost(p_ptr);
    // This will appear to succeed (a double free is identical to a device reset then free according from flamegpu::detail::gpu::gpuPointerGetAttributes' perspective), which is a difference from actual cudaFreeHost which would return cudaErrorInvalidValue.
    EXPECT_EQ(status, flamegpu::detail::gpu::gpuSuccess);
    // reset the ptr
    p_ptr = nullptr;
    // Allocate the pointer again
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuHostAlloc(reinterpret_cast<void**>(&p_ptr), sizeof(int), flamegpu::detail::gpu::gpuHostAllocDefault));
    // Validate that the ptr is a valid page-locked host pointer
    attributes = {};
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuPointerGetAttributes(&attributes, p_ptr));
    // this appears to return flamegpu::detail::gpu::gpuMemoryTypeHost, even though it should return flamegpu::detail::gpu::gpuMemoryTypeHost
    EXPECT_EQ(attributes.type, flamegpu::detail::gpu::gpuMemoryTypeHost);
    // Trigger a device reset
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuDeviceReset());
    // Attempt to free the ptr, this method should claim all things are fine (as the dev ptr has implicitly been freed)
    status = detail::cuda::cudaFreeHost(p_ptr);
    EXPECT_EQ(status, flamegpu::detail::gpu::gpuSuccess);
}

// Test that getting the primary context works, Difficult to trigger failure cases for this method, so coverage is subpar.
TEST(TestUtilDetailCuda, cuDevicePrimaryContextIsActive) {
#ifdef FLAMEGPU_USE_CUDA
    // Make sure device 0 is active for this test.
    flamegpu::detail::gpuCheck(cudaSetDevice(0));
    // Initialise a cudaContext, incase it somehow hasn't been already.
    flamegpu::detail::gpuCheck(cudaFree(0));
    // check if the primary context is active or not for device 0, it shoudl be.
    bool isActive = false;
    isActive = detail::cuda::cuDevicePrimaryContextIsActive(0);
    EXPECT_EQ(isActive, true);
    // Call device reset and check again without establishing a new context, it should not be active.
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuDeviceReset());
    isActive = detail::cuda::cuDevicePrimaryContextIsActive(0);
    EXPECT_EQ(isActive, false);
    // Check that exceptions will be raised correctly when passing bad device ordinals.
    // Expect an exception if the ordinal is negative
    EXPECT_THROW(detail::cuda::cuDevicePrimaryContextIsActive(-1), exception::InvalidCUDAdevice);
    // First grab the device count, to check for exceptions when the device ordinal is too big.
    int deviceCount = 0;
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuGetDeviceCount(&deviceCount));
    if (deviceCount > 0) {
        // Expect an exception if the ordinal is too big.
        EXPECT_THROW(detail::cuda::cuDevicePrimaryContextIsActive(deviceCount), exception::InvalidCUDAdevice);
    }
#else  // FLAMEGPU_USE_CUDA
    GTEST_SKIP() << "Test not yet implemented for HIP/ROCm/AMD";
#endif  // FLAMEGPU_USE_CUDA
}

}  // namespace flamegpu
