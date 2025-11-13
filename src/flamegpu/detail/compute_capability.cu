#include <nvrtc.h>

#include <cassert>
#include <array>
#include <vector>
#include <string>

#include "flamegpu/detail/compute_capability.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"


namespace flamegpu {
namespace detail {

namespace {
    /**
     * Templated variadic template constexpr method which returns a std::array from a number of arguments
     *
     * @parameter values multiple arguments which should all be of the same (int) type
     */
    template <typename... Args>
    constexpr auto macro_to_array(Args... values) {
        constexpr std::size_t N = sizeof...(Args);
        return std::array<int, N>{ (static_cast<int>(values))... };
    }
}  // namespace

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
    // extract the 0th value from the __CUDA_ARCH_LIST__ macro, and int divide by 10 to get a 2 or 3 digit integer compute capability
    #if defined(__CUDA_ARCH_LIST__)
        // Macro wrapper for getting the __CUDA_ARCH_LIST__ as a std::array of int
        #define MACRO_TO_ARRAY_WRAPPER() macro_to_array<int>(__CUDA_ARCH_LIST__)
        auto archs = MACRO_TO_ARRAY_WRAPPER();
        #undef MACRO_TO_ARRAY_WRAPPER
        if (archs.size() >= 1) {
            // return the 0th item int divided by 10
            return archs[0] / 10;
        }
    #endif
    // If the macro was not defined, or no architectures were found, return 0
    return 0;
}

std::string compute_capability::compiledCompiledComputeCapabilitiesString() {
    // Get a std::array<int> containign the values from the macro, via another macro
    #if defined(__CUDA_ARCH_LIST__)
        // Macro wrapper for getting the __CUDA_ARCH_LIST__ as a std::array of int
        #define MACRO_TO_ARRAY_WRAPPER() macro_to_array<int>(__CUDA_ARCH_LIST__)
        auto archs = MACRO_TO_ARRAY_WRAPPER();
        #undef MACRO_TO_ARRAY_WRAPPER
        // Build a semi-colon separated string (as CMAKE_CUDA_ARCHITECTURES must be space-separated)
        bool first = true;
        std::string result = "";
        for (const int& arch : archs) {
            if (!first) {
                result += ";";
            }
            // div by 10, as __CUDA_ARCH_LIST values have an extra 0 appended to them
            result += std::to_string(arch / 10);
            first = false;
        }
        return result;
    #else
        return std::array<int, 1>{0};
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
// Older CUDA's do not support this, but this is simple to hard-code for CUDA 11.0/11.1 (and CUDA 10.x).
// CUDA 11.1 supports 35 to 86
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
    for (const int &arch : architectures) {
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


const std::string compute_capability::getDeviceName(int deviceIndex) {
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
    // Load device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceIndex);

    return std::string(prop.name);
}

const std::string compute_capability::getDeviceNames(std::set<int> devices) {
    std::string device_names;
    bool first = true;
    // Get the count of devices
    int deviceCount = 0;
    gpuErrchk(cudaGetDeviceCount(&deviceCount));
    // If no devices were passed in, add each device to the set of devices.
    if (devices.size() == 0) {
        for (int i = 0; i < deviceCount; i++) {
            devices.emplace_hint(devices.end(), i);
        }
    }
    for (int device_id : devices) {
        // Throw an exception if the deviceIndex is negative.
        if (device_id < 0) {
            THROW exception::InvalidCUDAdevice();
        }
        // Ensure deviceIndex is valid.
        if (device_id >= deviceCount) {
            // Throw an exception if the device index is bad.
            THROW exception::InvalidCUDAdevice();
        }
        // Load device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device_id);
        if (!first)
            device_names.append(", ");
        device_names.append(prop.name);
        first = false;
    }
    return device_names;
}


}  // namespace detail
}  // namespace flamegpu
