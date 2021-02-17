#include <thread>
#include <chrono>
#include "flamegpu/util/CUDAEventTimer.cuh"
#include "flamegpu/util/nvtx.h"
#include "flamegpu/gpu/CUDAErrorChecking.h"

#include "gtest/gtest.h"

namespace test_CUDAEventTimer {

/**
 * This tests if the cudaEventTimer correctly times an event. 
 * GPUs using WDDM driver can be inaccurate.
 *   - eventRecord appear to be buffered to the device. 
 *   - So need to device sync before the threadSleep.
 *   - they appear to be accurate for timing the actual device work, just not wrt. the host (even in the default stream)
 *   - Needs further investigation at some point (@todo). May be worth falling back to chrono::steady_clock and notify the reduced precision if wddm? 
 */
TEST(TestUtilCUDAEventTimer, CUDAEventTimer) {
    NVTX_RANGE("TestUtilCUDAEventTimer:CUDAEventTimer");
    // Create an event timer, time should be 0 initially.
    util::CUDAEventTimer * timer = nullptr;
    EXPECT_NO_THROW(timer = new util::CUDAEventTimer());
    EXPECT_THROW(timer->getElapsedMilliseconds(), UnsycnedCUDAEventTimer);
    // Time an arbitrary event, and check the value is approximately correct.
    timer->start();
    const int sleep_duration_seconds = 1;
    const int min_expected_millis = static_cast<int>(sleep_duration_seconds * 1000. * 0.9);
    // WDDM check (windows only)
#ifdef _MSC_VER
    int deviceIndex = 0;
    int tccDriver = 0;  // Assume WDDM initially.
    gpuErrchk(cudaGetDevice(&deviceIndex));
    gpuErrchk(cudaDeviceGetAttribute(&tccDriver, cudaDevAttrTccDriver, deviceIndex));
    if (tccDriver != 1) {
        // Sync before the sleep to ensure the event has been recorded by wddm.
        // checking the status of the default stream did not appear to fix this?
        gpuErrchk(cudaDeviceSynchronize());
    }
#endif
    // Sleep for some amount of time.
    std::this_thread::sleep_for(std::chrono::seconds(sleep_duration_seconds));
    // Stop the timer.
    timer->stop();
    // Sync the timer and compare the recorded against the expected time.
    EXPECT_GE(timer->sync(), min_expected_millis);
    EXPECT_GE(timer->getElapsedMilliseconds(), min_expected_millis);
    // Trigger the destructor.
    EXPECT_NO_THROW(delete timer);
    // Reset the device for profiling?
    gpuErrchk(cudaDeviceReset());
}

}  // namespace test_CUDAEventTimer
