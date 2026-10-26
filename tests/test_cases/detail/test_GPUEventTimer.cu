#include <thread>
#include <chrono>
#include "flamegpu/detail/GPUEventTimer.cuh"
#include "flamegpu/detail/wddm.cuh"
#include "flamegpu/detail/gpu/gpu_api_error_checking.cuh"

#include "gtest/gtest.h"
namespace flamegpu {


namespace test_GPUEventTimer {

/**
 * This tests if the GPUEventTimer correctly times an event. 
 * timing on (CUDA) GPUs using WDDM driver can be inaccurate w.r.t the host.
 *   - eventRecord appear to be buffered to the device. 
 *   - So need to device sync before the threadSleep.
 *   - they appear to be accurate for timing the actual device work, just not wrt. the host (even in the default stream)
 *   - Needs further investigation at some point (@todo).
 *
 * Todo: Add tests covering use of streamDestroy if the timer object(s) have been copied or moved.
 */
TEST(TestDetailGPUEventTimer, GPUEventTimer) {
    // Create an event timer, time should be 0 initially.
    detail::Timer * timer = nullptr;
    EXPECT_NO_THROW(timer = new detail::GPUEventTimer());
    // Expect an exception if sync is called via getElapsed* if start() has not yet been called.
    EXPECT_THROW(timer->getElapsedMilliseconds(), exception::TimerException);
    // Time an arbitrary event, and check the value is approximately correct.
    timer->start();
    // Expect an exception if sync is called via getElapsed* if stop() has not yet been called.
    EXPECT_THROW(timer->getElapsedMilliseconds(), exception::TimerException);
    const int sleep_duration_seconds = 1;
    const double min_expected_seconds = sleep_duration_seconds * 0.9;
    const double min_expected_millis = min_expected_seconds * 1000.0;
    // If the WDDM driver is being used, this test is only accurate if the  start event is synchronised (pushed to the device) prior to the sleep.
    // Essentially, GPUEventTimers should not be used to time host code, they are only accurate for the device code which they wrap.
    if (detail::wddm::deviceIsWDDM()) {
        flamegpu::detail::gpuCheck(flamegpu::detail::gpu::gpuDeviceSynchronize());
    }
    // Sleep for some amount of time.
    std::this_thread::sleep_for(std::chrono::seconds(sleep_duration_seconds));
    // Stop the timer.
    timer->stop();
    // Get the elapsed time. This implicitly synchronises the timer.
    EXPECT_GE(timer->getElapsedMilliseconds(), min_expected_millis);
    // Also check the seconds method.
    EXPECT_GE(timer->getElapsedSeconds(), min_expected_seconds);
    // Trigger the destructor.
    EXPECT_NO_THROW(delete timer);
}

}  // namespace test_GPUEventTimer
}  // namespace flamegpu
