#include <string>
#include "flamegpu/detail/numeric.h"

#include "gtest/gtest.h"

namespace flamegpu {

TEST(TestNumeric, approxExactlyDivisible) {
    /**
     * Test that approxExactlyDivisible returns the expected value(s) for a number of cases, including values which were an issue for the original wrapped spatial messaging implementation (see #1157) 
     */
    ASSERT_TRUE(detail::numeric::approxExactlyDivisible<float>(1, 0.05f));
    ASSERT_TRUE(detail::numeric::approxExactlyDivisible<float>(1, 0.04f));
    ASSERT_TRUE(detail::numeric::approxExactlyDivisible<float>(2, 0.05f));
    ASSERT_TRUE(detail::numeric::approxExactlyDivisible<float>(1, 0.005f));
    ASSERT_TRUE(detail::numeric::approxExactlyDivisible<float>(50.0f, 10));
    ASSERT_TRUE(detail::numeric::approxExactlyDivisible<float>(100000, 0.05f));

    ASSERT_FALSE(detail::numeric::approxExactlyDivisible<float>(1, 0.03f));
    ASSERT_FALSE(detail::numeric::approxExactlyDivisible<float>(50.1f, 10));
    ASSERT_FALSE(detail::numeric::approxExactlyDivisible<float>(100000, 0.03f));
}

}  // namespace flamegpu
