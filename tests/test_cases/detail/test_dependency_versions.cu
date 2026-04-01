#include <thrust/version.h>

#ifdef FLAMEGPU_USE_CUDA
#include <cub/version.cuh>
#elif defined(FLAMEGPU_USE_HIP)
#include <hipcub/hipcub_version.hpp>
#endif  // FLAMEGPU_USE_HIP


#include "gtest/gtest.h"

namespace flamegpu {

TEST(TestDependencyVersions, ThrustVersion) {
    // todo: this value is out of date?
    const int EXPECTED_THRUST_VERSION = 100910;
    EXPECT_GE(THRUST_VERSION, EXPECTED_THRUST_VERSION);
}
TEST(TestDependencyVersions, CubVersion) {
    // todo: this value is out of date?
    const int EXPECTED_CUB_VERSION = 100910;
#ifdef FLAMEGPU_USE_CUDA
    EXPECT_GE(CUB_VERSION, EXPECTED_CUB_VERSION);
#elif defined(FLAMEGPU_USE_HIP)
    EXPECT_GE(HIPCUB_VERSION, EXPECTED_CUB_VERSION);
#endif
}
}  // namespace flamegpu
