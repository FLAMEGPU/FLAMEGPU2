#include "flamegpu/util/nvtx.h"
#include "flamegpu/simulation/detail/CUDAErrorChecking.cuh"

#include "gtest/gtest.h"

namespace flamegpu {

// Test nvtx versions, and whether or not use causes any potential issues.
TEST(TestUtilNVTX, nvtx) {
    // NVTX push, pop and range are no ops when disabled, and would need to query cupti or similar to detect if they actually pushed markers or not.

    // Check the namepsaced constexpr is set or not, we can only compare it to the macro
    const bool is_enabled = flamegpu::util::nvtx::ENABLED;
    #if defined(FLAMEGPU_USE_NVTX)
        EXPECT_EQ(is_enabled, true);
    #else
        EXPECT_EQ(is_enabled, false);
    #endif

    // Push a marker
    flamegpu::util::nvtx::push("Test_RANGE");
    // Pop the marker
    flamegpu::util::nvtx::pop();
    // Make a scoped range (push, pop)
    {
        flamegpu::util::nvtx::Range r{"Test_NVTX_RANGE"};
    }

    // If NVTX is enabled, we can check macros to determine which version was loaded.
    #if defined(FLAMEGPU_USE_NVTX) && defined(__CUDACC_VER_MAJOR__) && defined(NVTX_VERSION)
        int cuda_major = __CUDACC_VER_MAJOR__;
        int nvtx_version = NVTX_VERSION;
        if (cuda_major >= 10) {
            // CUDA >= 10.0 should be using NVTX3 or newer
            EXPECT_GE(nvtx_version, 3);
        } else {
            // If CUDA is < 10.0, should be using nvtx 1 or 2
            EXPECT_GE(nvtx_version, 0);
            EXPECT_LT(nvtx_version, 3);
        }
    #endif
}
}  // namespace flamegpu
