#include <string>

#include "flamegpu/version.h"
#include "gtest/gtest.h"

// Test that the macros are defined and and are of the correct type.
TEST(TestVersion, version) {
    const unsigned int macro_version = FLAMEGPU_VERSION;

    const unsigned int namespaced_version = flamegpu::VERSION;
    const unsigned int namespaced_version_major = flamegpu::VERSION_MAJOR;
    const unsigned int namespaced_version_minor = flamegpu::VERSION_MINOR;
    const unsigned int namespaced_version_patch = flamegpu::VERSION_PATCH;
    const char * namespaced_version_prerelease = flamegpu::VERSION_PRERELEASE;
    const char * namespaced_version_buildmetadata = flamegpu::VERSION_BUILDMETADATA;
    const char * namespaced_version_string = flamegpu::VERSION_STRING;
    const char * namespaced_version_full = flamegpu::VERSION_FULL;

    EXPECT_EQ(macro_version, namespaced_version);
    // Major must be a positive integer >= 2
    EXPECT_GE(namespaced_version_major, 2u);
    // Minor must be a non negative integer
    EXPECT_GE(namespaced_version_minor, 0u);
    // Patch be a non negative integer
    EXPECT_GE(namespaced_version_patch, 0u);

    // Perelease must be a string, which may be empty.
    EXPECT_GE(strlen(namespaced_version_prerelease), 0u);
    // build must be a string
    EXPECT_GE(strlen(namespaced_version_buildmetadata), 0u);
    // stirng version must be a string
    EXPECT_GE(strlen(namespaced_version_string), 0u);
    // build must be a string
    EXPECT_GE(strlen(namespaced_version_full), 0u);
}
