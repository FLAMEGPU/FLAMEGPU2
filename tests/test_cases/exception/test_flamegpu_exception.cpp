
#include "gtest/gtest.h"

#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {
namespace test_flamegpu_exception {

// Test the public methods on FLAMEGPUException objects, to ensure that va_args behaviour works as intended etc.
// FLAMEGPUException is an abstract type, so must use one of the  derived classes.


TEST(FLAMEGPUExceptionTest, message) {
    // Check that the default message of a derived exception behaves correctly.
    flamegpu::exception::CUDAError defaultExceptionMessage;
    EXPECT_STREQ(defaultExceptionMessage.what(), "CUDA returned an error code!");

    // Check that an empty message works.
    flamegpu::exception::CUDAError emptyExceptionMessage("");
    EXPECT_STREQ(emptyExceptionMessage.what(), "");

    // Check that a manual message is set correctly
    flamegpu::exception::CUDAError manualExceptionMessage("test");
    EXPECT_STREQ(manualExceptionMessage.what(), "test");

    // Check printf like behaviour for integers.
    flamegpu::exception::CUDAError intFormat("%d", 12);
    EXPECT_STREQ(intFormat.what(), "12");

    // Check printf like behaviour for strings.
    flamegpu::exception::CUDAError strFormat("%s", "test");
    EXPECT_STREQ(strFormat.what(), "test");

    // Check printf like behaviour for multiformat.
    flamegpu::exception::CUDAError sdFormat("%s %d", "test", 12);
    EXPECT_STREQ(sdFormat.what(), "test 12");
}

TEST(FLAMEGPUExceptionTest, exception_type) {
    // Test that the exception type behaves as intended
    flamegpu::exception::CUDAError ce;
    EXPECT_STREQ(ce.exception_type(), "CUDAError");

    flamegpu::exception::ReservedName rn;
    EXPECT_STREQ(rn.exception_type(), "ReservedName");
}

TEST(FLAMEGPUExceptionTest, setLocation) {
    // Test setting line and file mutates .waht
    const char* f = __FILE__;
    unsigned int l = __LINE__;
    const char * m = "test";
    flamegpu::exception::FLAMEGPUException::setLocation(f, l);
    flamegpu::exception::CUDAError ce(m);
    std::string expected_what = std::string(f) + "(" + std::to_string(l) + "): " + m;
    EXPECT_STREQ(ce.what(), expected_what.c_str());
}

// Define a new class extending FALMEGPUException which does not take a default_message in the constructor.

}  // namespace test_flamegpu_exception
}  // namespace flamegpu
