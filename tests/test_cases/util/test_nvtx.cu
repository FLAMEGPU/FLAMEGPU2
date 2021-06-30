#include "flamegpu/util/nvtx.h"
#include "flamegpu/gpu/detail/CUDAErrorChecking.cuh"

#include "gtest/gtest.h"

namespace flamegpu {


// Test nvtx versions, and whether or not use causes any potential issues.
TEST(TestUtilNVTX, nvtx) {
    // NVTX PUSH, POP and RANGE have no measurable side effects whether USE_NVTX is enabled or not, so use alone shouldn't cause any issues. This is effectively a compile time/linker test. This should probably be resolved (but has a cost).

    // Push a marker
    NVTX_PUSH("Test_RANGE");
    // Pop the marker
    NVTX_POP();
    // Make a scoped range (push, pop)
    {
        NVTX_RANGE("Test_NVTX_RANGE");
    }

    // If NVTX is enabled, we can check macros to determine which version was loaded.
    #if defined(USE_NVTX) && defined(__CUDACC_VER_MAJOR__) && defined(NVTX_VERSION)
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
