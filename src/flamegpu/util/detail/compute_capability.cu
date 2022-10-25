#include <nvrtc.h>

#include <cassert>

#include "flamegpu/util/detail/compute_capability.cuh"
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"


namespace flamegpu {
namespace util {
namespace detail {

int compute_capability::getComputeCapability(int deviceIndex) {
    int major = 0;
    int minor = 0;

    // Throw an exception if the deviceIndex is negative.
    if (deviceIndex < 0) {
        THROW exception::InvalidCUDAdevice();
    }

    // Ensure deviceIndex is valid.
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    if (deviceIndex >= deviceCount) {
        // Throw an excpetion if the device index is bad.
        THROW exception::InvalidCUDAdevice();
    }
    // Load device attributes
    gpuErrchk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));
    gpuErrchk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
    // Compute the arch integer value.
    int arch = (10 * major) + minor;
    return arch;
}

int compute_capability::minimumCompiledComputeCapability() {
    #if defined(MIN_CUDA_ARCH)
        return MIN_CUDA_ARCH;
    #else
        // Return 0 as a default minimum?
        return 0;
    #endif
}

bool compute_capability::checkComputeCapability(int deviceIndex) {
    // If the compile time minimum architecture is defined, fetch the device's compute capability and check that the executable (probably) supports this device.
    if (getComputeCapability(deviceIndex) < minimumCompiledComputeCapability()) {
        return false;
    } else {
        return true;
    }
}

std::vector<int> compute_capability::getNVRTCSupportedComputeCapabilties() {
// NVRTC included with CUDA 11.2+ includes methods to query the supported architectures and CUDA from 11.2+
// Also changes the soname rules such that nvrtc.11.2.so is vald for all nvrtc >= 11.2, and libnvrtc.12.so for CUDA 12.x etc, so this is different at runtime not compile time for future versions, so use the methods
#if (__CUDACC_VER_MAJOR__ > 11) || ((__CUDACC_VER_MAJOR__ == 11) && __CUDACC_VER_MINOR__ >= 2)
    nvrtcResult nvrtcStatus = NVRTC_SUCCESS;
    int nvrtcNumSupportedArchs = 0;
    // Query the number of architecture flags supported by this nvrtc, to allocate enough memory
    nvrtcStatus = nvrtcGetNumSupportedArchs(&nvrtcNumSupportedArchs);
    if (nvrtcStatus == NVRTC_SUCCESS && nvrtcNumSupportedArchs > 0) {
        // prepare a large enough std::vector for the results
        std::vector<int> nvrtcSupportedArchs = std::vector<int>(nvrtcNumSupportedArchs);
        assert(nvrtcSupportedArchs.size() >= nvrtcNumSupportedArchs);
        nvrtcStatus = nvrtcGetSupportedArchs(nvrtcSupportedArchs.data());
        if (nvrtcStatus == NVRTC_SUCCESS) {
            // Return the populated std::vector, this should be RVO'd
            return nvrtcSupportedArchs;
        }
    }
    // If any of the above functions failed, we have no idea what arch's are supported, so assume none are?
    return {};
// Older CUDA's do not support this, but this is simple to hard-code for CUDA 11.0/11.1  (and our deprected CUDA 10.x).
// CUDA 11.1 suports 35 to 86
#elif (__CUDACC_VER_MAJOR__ == 11) && __CUDACC_VER_MINOR__ == 1
    return {35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80, 86};
// CUDA 11.0 supports 35 to 80
#elif (__CUDACC_VER_MAJOR__ == 11) && __CUDACC_VER_MINOR__ == 0
    return {35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75, 80};
// CUDA 10.x supports 30 to 75
#elif (__CUDACC_VER_MAJOR__ >= 10)
    return {30, 32, 35, 37, 50, 52, 53, 60, 61, 62, 70, 72, 75};
// This should be all cases for FLAME GPU 2, but leave the fallback branch just in case
#else
    return {};
#endif
}

int compute_capability::selectAppropraiteComputeCapability(const int target, const std::vector<int>& architectures) {
    int maxArch = 0;
    for (const int arch : architectures) {
        if (arch <= target && arch > maxArch) {
            maxArch = arch;
            // The vector is in ascending order, so we can potentially early exit
            if (arch == target) {
                return target;
            }
        }
    }
    return maxArch;
}

}  // namespace detail
}  // namespace util
}  // namespace flamegpu
