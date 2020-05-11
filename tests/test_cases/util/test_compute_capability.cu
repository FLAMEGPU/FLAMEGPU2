#include <cuda_runtime.h>

#include "flamegpu/util/compute_capability.cuh"
#include "flamegpu/gpu/CUDAErrorChecking.h"

#include "gtest/gtest.h"

// Test the getting of a device's compute capability.
TEST(TestUtilComputeCapability, getComputeCapability) {
    // Get the number of cuda devices
    int device_count = 0;
    if (cudaSuccess != cudaGetDeviceCount(&device_count) || device_count <= 0) {
        return;
    }
    // For each CUDA device, get the compute capability and check it.
    for (int i = 0; i < device_count; i++) {
        // Manually get the arch as the reference.
        int major = 0;
        int minor = 0;
        gpuErrchk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, i));
        gpuErrchk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, i));
        int reference = (10 * major) + minor;
        // The function should return the reference value.
        EXPECT_EQ(util::compute_capability::getComputeCapability(i), reference);
    }

    // If the function is given a bad index, it should throw.
    EXPECT_ANY_THROW(util::compute_capability::getComputeCapability(-1));
    EXPECT_ANY_THROW(util::compute_capability::getComputeCapability(device_count));
}

// Test getting the minimum compiled cuda capabillity.
TEST(TestUtilComputeCapability, minimumCompiledComputeCapability) {
    // If the macro is defined, the returned value should match, otherwise it should be 0.
    #if defined(MIN_CUDA_ARCH)
        EXPECT_EQ(util::compute_capability::minimumCompiledComputeCapability(), MIN_CUDA_ARCH);
    #else
        EXPECT_EQ(util::compute_capability::minimumCompiledComputeCapability(), 0);
    #endif
}

// Test checking the compute capability of a specific device.
TEST(TestUtilComputeCapability, checkComputeCapability) {
    // Get the number of cuda devices
    int device_count = 0;
    if (cudaSuccess != cudaGetDeviceCount(&device_count) || device_count <= 0) {
        return;
    }
    // Get the minimum cc compiled for, previously tested.
    int min_cc = util::compute_capability::minimumCompiledComputeCapability();
    // For each CUDA device, get the compute capability and comapre it against
    for (int i = 0; i < device_count; i++) {
        // This function is tested elsewhere, so use it here.
        int cc = util::compute_capability::getComputeCapability(i);
        EXPECT_EQ(util::compute_capability::checkComputeCapability(i), cc >= min_cc);
    }

    // If the function is given a bad index, it should throw and the result is irrelevant.
    EXPECT_ANY_THROW(util::compute_capability::checkComputeCapability(-1));
    EXPECT_ANY_THROW(util::compute_capability::checkComputeCapability(device_count));
}
