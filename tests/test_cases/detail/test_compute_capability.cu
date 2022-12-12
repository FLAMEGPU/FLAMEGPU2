#include <cuda_runtime.h>

#include <vector>
#include "flamegpu/detail/compute_capability.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

#include "gtest/gtest.h"
namespace flamegpu {


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
        EXPECT_EQ(detail::compute_capability::getComputeCapability(i), reference);
    }

    // If the function is given a bad index, it should throw.
    EXPECT_ANY_THROW(detail::compute_capability::getComputeCapability(-1));
    EXPECT_ANY_THROW(detail::compute_capability::getComputeCapability(device_count));
}

// Test getting the minimum compiled cuda capabillity.
TEST(TestUtilComputeCapability, minimumCompiledComputeCapability) {
    // If the macro is defined, the returned value should match, otherwise it should be 0.
    #if defined(FLAMEGPU_TEST_MIN_CUDA_ARCH)
        EXPECT_EQ(detail::compute_capability::minimumCompiledComputeCapability(), FLAMEGPU_TEST_MIN_CUDA_ARCH);
    #else
        EXPECT_EQ(detail::compute_capability::minimumCompiledComputeCapability(), 0);
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
    int min_cc = detail::compute_capability::minimumCompiledComputeCapability();
    // For each CUDA device, get the compute capability and comapre it against
    for (int i = 0; i < device_count; i++) {
        // This function is tested elsewhere, so use it here.
        int cc = detail::compute_capability::getComputeCapability(i);
        EXPECT_EQ(detail::compute_capability::checkComputeCapability(i), cc >= min_cc);
    }

    // If the function is given a bad index, it should throw and the result is irrelevant.
    EXPECT_ANY_THROW(detail::compute_capability::checkComputeCapability(-1));
    EXPECT_ANY_THROW(detail::compute_capability::checkComputeCapability(device_count));
}

/**
 * Test getting the nvrtc supported compute capabilities. 
 * This depends on the CUDA version used, and the dynamically linked nvrtc (when CUDA >= 11.2) so this is not ideal to test. 
 */
TEST(TestUtilComputeCapability, getNVRTCSupportedComputeCapabilties) {
    std::vector<int> architectures = detail::compute_capability::getNVRTCSupportedComputeCapabilties();

    // CUDA 11.2+ we do not know what values or how many this should return, so just assume a non zero number will be returned (in case of future additions / removals)
    #if (__CUDACC_VER_MAJOR__ > 11) || ((__CUDACC_VER_MAJOR__ == 11) && __CUDACC_VER_MINOR__ >= 2)
        EXPECT_GT(architectures.size(), 0);
    // CUDA 11.1 suports 35 to 86, (13 arch's)
    #elif (__CUDACC_VER_MAJOR__ == 11) && __CUDACC_VER_MINOR__ == 1
        EXPECT_EQ(architectures.size(), 13);
    // CUDA 11.0 supports 35 to 80 (12 arch's)
    #elif (__CUDACC_VER_MAJOR__ == 11) && __CUDACC_VER_MINOR__ == 0
        EXPECT_EQ(architectures.size(), 12);
    // CUDA 10.x supports 30 to 75 (13 arch's)
    #elif (__CUDACC_VER_MAJOR__ >= 10)
        EXPECT_EQ(architectures.size(), 13);
    // Otherwise there will be 0.
    #else
        EXPECT_EQ(architectures.size(), 0);
    #endif
}

/**
 * Test that given an ascending order of compute capabilities, and a target compute capability, greatest value which is LE the target is found, or 0 otherwise.
 */
TEST(TestUtilComputeCapability, selectAppropraiteComputeCapability) {
    // Check an exact match should be found
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(86, {86}), 86);
    // Check a miss but with a lower value returns the lower value
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(86, {80}), 80);
    // Check a miss without a valid value returns 0
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(86, {90}), 0);
    // Check a miss occurs when no values are present in the vector.
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(86, {}), 0);

    // CUDA 11.1-11.6, 35 to 86, 86 and 60 should be found, 30 should not.
    std::vector<int> CUDA_11_1_ARCHES = {35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86};
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(86, CUDA_11_1_ARCHES), 86);
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(60, CUDA_11_1_ARCHES), 60);
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(30, CUDA_11_1_ARCHES), 0);

    // CUDA 11.0, 86 should not be found, but 80 should be used instead. 60 should be found, 30 should not.
    std::vector<int> CUDA_11_0_ARCHES = {35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80};
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(86, CUDA_11_0_ARCHES), 80);
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(60, CUDA_11_0_ARCHES), 60);
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(30, CUDA_11_0_ARCHES), 0);
    // CUDA 10.0, 86 should not be found, 75 should be used. 60 should be found, 30 should eb found.
    std::vector<int> CUDA_10_0_ARCHES = {30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75};
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(86, CUDA_10_0_ARCHES), 75);
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(60, CUDA_10_0_ARCHES), 60);
    EXPECT_EQ(detail::compute_capability::selectAppropraiteComputeCapability(30, CUDA_10_0_ARCHES), 30);
}

}  // namespace flamegpu
