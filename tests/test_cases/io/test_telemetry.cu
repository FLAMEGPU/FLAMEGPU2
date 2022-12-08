#include <cstdio>
#include <cstdlib>

#include "flamegpu/io/Telemetry.h"

#include "gtest/gtest.h"

namespace flamegpu {
namespace tests {
namespace test_telemetry {

TEST(TestTelemetry, telemetryEnableDisable) {
    // Telemetry has been disabled by the test suite handle, so it should be off by default at this point in time. We cannot check the initial parsing of this value unfortunately (without making a copy in a global?)
    EXPECT_FALSE(flamegpu::io::Telemetry::isEnabled());
    // Turn telemetry on, check it is enabled.
    EXPECT_NO_THROW(flamegpu::io::Telemetry::enable());
    EXPECT_TRUE(flamegpu::io::Telemetry::isEnabled());
    // Turn it off again, making sure it is disabled
    EXPECT_NO_THROW(flamegpu::io::Telemetry::disable());
    EXPECT_FALSE(flamegpu::io::Telemetry::isEnabled());
}

TEST(TestTelemetry, suppressNotice) {
    // We cannot check the value of suppression, and cannot re-enable the suppression warning with the protection / annon namespace, so all we can do is check the method exists.
    EXPECT_NO_THROW(flamegpu::io::Telemetry::suppressNotice());
}

}  // namespace test_telemetry
}  // namespace tests
}  // namespace flamegpu
