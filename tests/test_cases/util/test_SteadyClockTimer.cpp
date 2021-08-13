#include <thread>
#include <chrono>
#include <ratio>
#include "flamegpu/util/detail/Timer.h"
#include "flamegpu/util/detail/SteadyClockTimer.h"

#include "gtest/gtest.h"
namespace flamegpu {


TEST(TestSteadyClockTimer, SteadyClockTimer) {
    // Create a steady clock timer via the base class
    util::detail::Timer * timer = nullptr;
    EXPECT_NO_THROW(timer = new util::detail::SteadyClockTimer());
    // Expect an exception if sync is called via getElapsed* if start() has not yet been called.
    EXPECT_THROW(timer->getElapsedMilliseconds(), exception::TimerException);
    // Time an arbitrary event, and check the value is approximately correct.
    timer->start();
    // Expect an exception if sync is called via getElapsed* if stop() has not yet been called.
    EXPECT_THROW(timer->getElapsedMilliseconds(), exception::TimerException);
    const int sleep_duration_seconds = 1;
    const float min_expected_seconds = sleep_duration_seconds * 0.9f;
    const float min_expected_millis = min_expected_seconds * 1000.0f;

    std::this_thread::sleep_for(std::chrono::seconds(sleep_duration_seconds));
    timer->stop();
    EXPECT_GE(timer->getElapsedMilliseconds(), min_expected_millis);
    EXPECT_GE(timer->getElapsedSeconds(), min_expected_seconds);

    // Trigger the destructor.
    EXPECT_NO_THROW(delete timer);
}
}  // namespace flamegpu
