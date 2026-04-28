#ifdef FLAMEGPU_USE_CUDA
#include <cuda_runtime.h>
#endif

#include <vector>
#include "flamegpu/detail/cuda.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

#include "gtest/gtest.h"
namespace flamegpu {

#if FLAMEGPU_USE_HIP
using cudaPointerAttributes = hipPointerAttribute_t;
#endif


// Test that wrapped cudaFree works.
TEST(TestUtilDetailCuda, cudaFree) {
    int * d_ptr = nullptr;
    flamegpu::detail::gpu::Error_t status = FLAMEGPU_GPU_RUNTIME_SYMBOL(Success);
    // manually allocate a device pointer
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(Malloc)(&d_ptr, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    cudaPointerAttributes attributes = {};
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeDevice));
    // call the wrapped cuda free method
    status = detail::cuda::cudaFree(d_ptr);
    // It should not have thrown any cuda errors in normal use.
    EXPECT_EQ(status, FLAMEGPU_GPU_RUNTIME_SYMBOL(Success));
    // The pointer will still have a non nullptr value, but it will no longer be a valid device ptr.
    EXPECT_NE(d_ptr, nullptr);
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeUnregistered));
    // Try a double free.
    status = detail::cuda::cudaFree(d_ptr);
    // This will appear to succeed (a double free is identical to a device reset then free according from FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)' perspective), which is a difference from actual cudaFree which would return cudaErrorInvalidValue.
    EXPECT_EQ(status, FLAMEGPU_GPU_RUNTIME_SYMBOL(Success));
    // reset the ptr
    d_ptr = nullptr;
    // Allocate the pointer again
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(Malloc)(&d_ptr, sizeof(int)));
    // Validate that the ptr is a valid device pointer
    attributes = {};
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, d_ptr));
    EXPECT_EQ(attributes.type, FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeDevice));
    // Trigger a device reset
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceReset)());
    // Attempt to free the ptr, this method should claim all things are fine (as the dev ptr has implicitly been freed)
    status = detail::cuda::cudaFree(d_ptr);
    EXPECT_EQ(status, FLAMEGPU_GPU_RUNTIME_SYMBOL(Success));
}

// Test that the wrapped cudaFreeHost works.
TEST(TestUtilDetailCuda, cudaFreeHost) {
    int * p_ptr = nullptr;
    flamegpu::detail::gpu::Error_t status = FLAMEGPU_GPU_RUNTIME_SYMBOL(Success);
    // manually allocate a page-locked host pointer
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(HostAlloc)((void**)&p_ptr, sizeof(int), FLAMEGPU_GPU_RUNTIME_SYMBOL(HostAllocDefault)));
    // Validate that the ptr is a valid page-locked host pointer
    cudaPointerAttributes attributes = {};
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, p_ptr));
    // this appears to return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost), even though it should return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost)
    EXPECT_EQ(attributes.type, FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost));
    // call the wrapped cuda free method
    status = detail::cuda::cudaFreeHost(p_ptr);
    // It should not have thrown any cuda errors in normal use.
    EXPECT_EQ(status, FLAMEGPU_GPU_RUNTIME_SYMBOL(Success));
    // The pointer will still have a non nullptr value, but it will no longer be a valid page-locked ptr.
    EXPECT_NE(p_ptr, nullptr);
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, p_ptr));
    EXPECT_EQ(attributes.type, FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeUnregistered));

    // Try a double free.
    status = detail::cuda::cudaFreeHost(p_ptr);
    // This will appear to succeed (a double free is identical to a device reset then free according from FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)' perspective), which is a difference from actual cudaFreeHost which would return cudaErrorInvalidValue.
    EXPECT_EQ(status, FLAMEGPU_GPU_RUNTIME_SYMBOL(Success));
    // reset the ptr
    p_ptr = nullptr;
    // Allocate the pointer again
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(HostAlloc)((void**)&p_ptr, sizeof(int), FLAMEGPU_GPU_RUNTIME_SYMBOL(HostAllocDefault)));
    // Validate that the ptr is a valid page-locked host pointer
    attributes = {};
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(&attributes, p_ptr));
    // this appears to return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost), even though it should return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost)
    EXPECT_EQ(attributes.type, FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost));
    // Trigger a device reset
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceReset)());
    // Attempt to free the ptr, this method should claim all things are fine (as the dev ptr has implicitly been freed)
    status = detail::cuda::cudaFreeHost(p_ptr);
    EXPECT_EQ(status, FLAMEGPU_GPU_RUNTIME_SYMBOL(Success));
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
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceReset)());
    isActive = detail::cuda::cuDevicePrimaryContextIsActive(0);
    EXPECT_EQ(isActive, false);
    // Check that exceptions will be raised correctly when passing bad device ordinals.
    // Expect an exception if the ordinal is negative
    EXPECT_THROW(detail::cuda::cuDevicePrimaryContextIsActive(-1), exception::InvalidCUDAdevice);
    // First grab the device count, to check for exceptions when the device ordinal is too big.
    int deviceCount = 0;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceCount)(&deviceCount));
    if (deviceCount > 0) {
        // Expect an exception if the ordinal is too big.
        EXPECT_THROW(detail::cuda::cuDevicePrimaryContextIsActive(deviceCount), exception::InvalidCUDAdevice);
    }
#else  // FLAMEGPU_USE_CUDA
    GTEST_SKIP() << "Test not yet implemented for HIP/ROCm/AMD";
#endif  // FLAMEGPU_USE_CUDA
}

}  // namespace flamegpu
