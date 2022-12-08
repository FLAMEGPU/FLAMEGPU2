#include <cuda_runtime.h>

#include "flamegpu/detail/wddm.cuh"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

#include "gtest/gtest.h"
namespace flamegpu {


// Test getting the WDDM status of a device.
TEST(TestUtilWDDM, deviceIsWDDM) {
    // The output of these methods depends on the device it is running on, and will not be easy to mock.
    // Instead, it compares the computed value by the library against a locally calculated value, likely using the same code as in the implementation.

    // Get the number of cuda devices
    int device_count = 0;
    if (cudaSuccess != cudaGetDeviceCount(&device_count) || device_count <= 0) {
        return;
    }
    // For each CUDA device, get the wddm value and check it.
    // Do not only check this if _MSC_VER, unsure how this will behave if WDDM + WSL
    for (int i = 0; i < device_count; i++) {
        bool reference = false;
        #ifdef _MSC_VER
            int tccDriver = 0;
            // Get if the driver is TCC or not
            gpuErrchk(cudaDeviceGetAttribute(&tccDriver, cudaDevAttrTccDriver, i));
            // WDDM driver is if not the tcc driver, and on windows.
            reference = !tccDriver;
        #endif
        // The function should return the reference value.
        EXPECT_EQ(detail::wddm::deviceIsWDDM(i), reference);
    }

    // If the function is given a bad index, it should throw.
    EXPECT_ANY_THROW(detail::wddm::deviceIsWDDM(-1));
    EXPECT_ANY_THROW(detail::wddm::deviceIsWDDM(device_count));

    // Also check for the current device.
    int currentDeviceIndex = 0;
    gpuErrchk(cudaGetDevice(&currentDeviceIndex));
    bool reference = false;
    #ifdef _MSC_VER
        int tccDriver = 0;
        // Get if the driver is TCC or not
        gpuErrchk(cudaDeviceGetAttribute(&tccDriver, cudaDevAttrTccDriver, currentDeviceIndex));
        // WDDM driver is if not the tcc driver, and on windows.
        reference = !tccDriver;
    #endif
    EXPECT_EQ(detail::wddm::deviceIsWDDM(), reference);
}
}  // namespace flamegpu
