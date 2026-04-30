#include "flamegpu/util/nvtx.h"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"

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
}
}  // namespace flamegpu
