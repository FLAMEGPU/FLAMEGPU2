#include <set>
#include <string>

#include "flamegpu/detail/compute_capability.cuh"
#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"


namespace flamegpu {
namespace detail {
namespace gpu {

namespace {

// using type-alias to abstract the differently named HIP/CUDA device properties struct, in the anon-namespace for scope
#if defined(FLAMEGPU_USE_HIP)
    using DeviceProp_t = hipDeviceProp_t;
#else  // defined(FLAMEGPU_USE_CUDA)
    using DeviceProp_t = cudaDeviceProp;
#endif  // defined(bFLAMEGPU_USE_HIP)

}  // namespace

const std::string getDeviceName(int deviceIndex) {
    // Throw an exception if the deviceIndex is negative.
    if (deviceIndex < 0) {
        THROW exception::InvalidCUDAdevice();
    }

    // Ensure deviceIndex is valid.
    int deviceCount = 0;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceCount)(&deviceCount));
    if (deviceIndex >= deviceCount) {
        // Throw an excpetion if the device index is bad.
        THROW exception::InvalidCUDAdevice();
    }
    // Load device properties
    DeviceProp_t prop;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceProperties)(&prop, deviceIndex));

    return std::string(prop.name);
}

const std::string getDeviceNames(std::set<int> devices) {
    std::string device_names;
    bool first = true;
    // Get the count of devices
    int deviceCount = 0;
    flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceCount)(&deviceCount));
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
        DeviceProp_t prop;
        flamegpu::detail::gpuCheck(FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceProperties)(&prop, device_id));
        if (!first)
            device_names.append(", ");
        device_names.append(prop.name);
        first = false;
    }
    return device_names;
}

}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu
