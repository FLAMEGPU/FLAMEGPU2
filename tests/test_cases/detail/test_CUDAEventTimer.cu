#include <thread>
#include <chrono>
#include "flamegpu/detail/CUDAEventTimer.cuh"
#include "flamegpu/detail/wddm.cuh"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"

#include "gtest/gtest.h"
namespace flamegpu {


namespace test_CUDAEventTimer {

// Suppress deprecation warnings via pragmas so this can be tested in CI, but still issue the deprecation warning otherwise.

#if defined(__GNUC__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__clang__)
    #pragma clang diagnostic push
    #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(_MSC_VER)
    #pragma warning(push)
    #pragma warning(disable: 4996)
#endif

/**
 * CUDAEventTimer is a deprecated alias of GPUEventTimer. 

 This test suite just checks that it can be imported and an instance constructed.

 See test_GPUEventTimer for actual behavuioural testing
 */
TEST(TestDetailCUDAEventTimer, CUDAEventTimer) {
    detail::Timer * timer = nullptr;
    EXPECT_NO_THROW(timer = new detail::CUDAEventTimer());
    EXPECT_NO_THROW(delete timer);
}

#if defined(__GNUC__)
    #pragma GCC diagnostic pop
#elif defined(__clang__)
    #pragma clang diagnostic pop
#elif defined(_MSC_VER)
    #pragma warning(pop)
#endif

}  // namespace test_CUDAEventTimer
}  // namespace flamegpu
