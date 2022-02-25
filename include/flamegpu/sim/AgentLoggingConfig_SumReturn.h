#ifndef INCLUDE_FLAMEGPU_SIM_AGENTLOGGINGCONFIG_SUMRETURN_H_
#define INCLUDE_FLAMEGPU_SIM_AGENTLOGGINGCONFIG_SUMRETURN_H_

#include <cstdint>

namespace flamegpu {

/**
 * Template for converting a type to the most suitable type of the same format with greatest range
 * Useful when summing unknown values
 * e.g. sum_input_t<float>::result_t == double
 * e.g. sum_input_t<uint8_t>::result_t == uint64_t
 */
template <typename T> struct sum_input_t;
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<float> { typedef double result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<double> { typedef double result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<char> { typedef uint64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<uint8_t> { typedef uint64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<uint16_t> { typedef uint64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<uint32_t> { typedef uint64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<uint64_t> { typedef uint64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<int8_t> { typedef int64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<int16_t> { typedef int64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<int32_t> { typedef int64_t result_t; };
/**
 * @see sum_input_t
 */
template <> struct sum_input_t<int64_t> { typedef int64_t result_t; };
/**
 * @see sum_input_t
 */
template <typename T> struct sum_input_t { typedef T result_t; };

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_SIM_AGENTLOGGINGCONFIG_SUMRETURN_H_
