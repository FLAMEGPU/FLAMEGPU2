#include "flamegpu/util/compute_capability.cuh"
#include "flamegpu/gpu/CUDAErrorChecking.h"

int util::compute_capability::getComputeCapability(int deviceIndex) {
        int major = 0;
        int minor = 0;
        gpuErrchk(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, deviceIndex));
        gpuErrchk(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, deviceIndex));
        int arch = (10 * major) + minor;
        return arch;
}

int util::compute_capability::minimumCompiledComputeCapability() {
    #if defined(MIN_ARCH)
        return MIN_ARCH;
    #else
        // Return 0 as a default minimum?
        return 0;
    #endif
}

bool util::compute_capability::checkComputeCapability(int deviceIndex) {
    // If the compile time minimum architecture is defined, fetch the device's compute capability and check that the executable (probably) supports this device.
    #if defined(MIN_ARCH)
        if (getComputeCapability(deviceIndex) < MIN_ARCH) {
            return false;
        } else {
            return true;
        }
    #else
        // If not defined, we cannot make a decision so assume it will work?
        return true;
    #endif
}
