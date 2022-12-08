#include "flamegpu/detail/wddm.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

namespace flamegpu {
namespace detail {

bool wddm::deviceIsWDDM(int deviceIndex) {
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
    // @todo - once WSL does not require insider builds, check how this behaves. AFAIK WSL is only supported for WDDM, but this may be incorrect.
    bool isWDDM = false;
    #ifdef _MSC_VER
        int tccDriver = 0;
        // Load device attributes
        gpuErrchk(cudaDeviceGetAttribute(&tccDriver, cudaDevAttrTccDriver, deviceIndex));
        // Compute the return value
        isWDDM = !tccDriver;
    #endif
    return isWDDM;
}

bool wddm::deviceIsWDDM() {
    // Get the current device
    int currentDeviceIndex = 0;
    gpuErrchk(cudaGetDevice(&currentDeviceIndex));
    // Get the wddm status for that device
    bool isWDDM = wddm::deviceIsWDDM(currentDeviceIndex);
    return isWDDM;
}

}  // namespace detail
}  // namespace flamegpu
