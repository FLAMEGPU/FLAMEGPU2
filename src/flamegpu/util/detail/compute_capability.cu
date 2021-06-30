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

}  // namespace detail
}  // namespace util
}  // namespace flamegpu
