#include <thrust/version.h>
#include <cub/version.cuh>

#include "gtest/gtest.h"

TEST(TestDependencyVersions, ThrustVersion) {
    const int EXPECTED_THRUST_VERSION = 100910;
    EXPECT_GE(THRUST_VERSION, EXPECTED_THRUST_VERSION);
}
TEST(TestDependencyVersions, CubVersion) {
    const int EXPECTED_CUB_VERSION = 100910;
    EXPECT_GE(CUB_VERSION, EXPECTED_CUB_VERSION);
}
