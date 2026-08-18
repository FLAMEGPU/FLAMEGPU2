#include "flamegpu/detail/wddm.cuh"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"

namespace flamegpu {
namespace detail {

bool wddm::deviceIsWDDM(int deviceIndex) {
    // Throw an exception if the deviceIndex is negative.
    if (deviceIndex < 0) {
        THROW exception::InvalidCUDAdevice();
    }
    // Ensure deviceIndex is valid.
    int deviceCount = 0;
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuGetDeviceCount(&deviceCount));
    if (deviceIndex >= deviceCount) {
        // Throw an excpetion if the device index is bad.
        THROW exception::InvalidCUDAdevice();
    }
    // Assume not WDDM
    bool isWDDM = false;
    // If on windows with CUDA, check the tccDriver attribute, setting the result value to true if it is not tcc
    #if defined(_MSC_VER) && defined(FLAMEGPU_USE_CUDA)
        int tccDriver = 0;
        // Load device attributes
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuDeviceGetAttribute(&tccDriver, FLAMEGPU_GPU_RUNTIME_SYMBOL(DevAttrTccDriver), deviceIndex));
        // Compute the return value
        isWDDM = !tccDriver;
    #endif
    return isWDDM;
}

bool wddm::deviceIsWDDM() {
    // Get the current device
    int currentDeviceIndex = 0;
    flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuGetDevice(&currentDeviceIndex));
    // Get the wddm status for that device
    bool isWDDM = wddm::deviceIsWDDM(currentDeviceIndex);
    return isWDDM;
}

}  // namespace detail
}  // namespace flamegpu
