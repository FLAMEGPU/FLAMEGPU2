#include <string>

#include "flamegpu/util/Environment.h"
#include "flamegpu/io/Telemetry.h"
#include "gtest/gtest.h"

namespace flamegpu {
namespace tests {
namespace test_environment {
// Check that the environmetn helper functions
TEST(TestEnvironment, TestEnvironment) {
    EXPECT_FALSE(flamegpu::io::Telemetry::globalTelemetryEnabled());                  // Should have been disabled globally
    EXPECT_TRUE(flamegpu::util::hasEnvironmentVariable("SILENCE_TELEMETRY_NOTICE"));  // Should have been set globally
}
// Check that the test environment varibale for expected variables
TEST(TestEnvironment, TestEnvironmentHelpers) {
    // Value not expected
    EXPECT_FALSE(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
    // Set the value to various options and check expected value
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE", "On");
    EXPECT_TRUE(std::string("On") == flamegpu::util::getEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
    EXPECT_TRUE(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
    // Set the value to various options for off and check expected value
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE", "OFF");
    EXPECT_FALSE(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE", "Off");
    EXPECT_FALSE(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE", "0");
    EXPECT_FALSE(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE", "FALSE");
    EXPECT_FALSE(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
    flamegpu::util::setEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE", "False");
    EXPECT_FALSE(flamegpu::util::hasEnvironmentVariable("FLAMEGPU_TEST_ENVIRONMENT_TEST_VARIABLE"));
}
}  // namespace test_environment
}  // namespace tests
}  // namespace flamegpu
