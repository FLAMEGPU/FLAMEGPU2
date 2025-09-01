#ifndef INCLUDE_FLAMEGPU_DETAIL_NUMERIC_H_
#define INCLUDE_FLAMEGPU_DETAIL_NUMERIC_H_

#include <algorithm>
#include <limits>
#include <cmath>

namespace flamegpu {
namespace detail {
/**
 * Internal (detail) methods related to numbers.
 */
namespace numeric {

/**
 * Templated method to check whether a value (x) is approximately exactly divisible by another value (y).
 * 
 * @param x the numerator
 * @param y the denominator
 * 
 * @return boolean indicating if x is exactly divisible by y, within reason given the use of floating point numbers.
 */
template <typename T>
bool approxExactlyDivisible(T x, T y) {
    // Scale machine epsilon by the magnitude of the larger value
    T scaledEpsilon = std::max(std::abs(x), std::abs(y)) * std::numeric_limits<T>::epsilon();
    // Compute the remainder
    T v = std::fmod(x, y);
    // approx equal if the remainder is within scaledEpsilon of 0 or b (fmod(1, 0.05f) returns ~0.05f)
    return v <= scaledEpsilon || v > y - scaledEpsilon;
}

}  // namespace numeric
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_NUMERIC_H_
