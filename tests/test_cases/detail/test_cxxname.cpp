#include <string>
#include "flamegpu/detail/cxxname.hpp"

#include "gtest/gtest.h"

namespace test_cxxname {

/**
 * Tests if cxxname::getUnqualified name behaves as intended.
 */
TEST(TestUtilCxxname, getUnqualifiedName) {
    // Check with no qualification
    EXPECT_EQ(flamegpu::detail::cxxname::getUnqualifiedName(std::string("ClassName")), std::string("ClassName"));
    // Check with one qualifier
    EXPECT_EQ(flamegpu::detail::cxxname::getUnqualifiedName(std::string("namespace::ClassName")), std::string("ClassName"));

    // Check with two qualifiers
    EXPECT_EQ(flamegpu::detail::cxxname::getUnqualifiedName(std::string("namespace::subnamespace::ClassName")), std::string("ClassName"));

    // Check with const char * as an argument
    EXPECT_EQ(flamegpu::detail::cxxname::getUnqualifiedName("namespace::ClassName"), "ClassName");
}

}  // namespace test_cxxname
