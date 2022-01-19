#ifndef INCLUDE_FLAMEGPU_SIM_AGENTLOGGINGCONFIG_REDUCTIONS_CUH_
#define INCLUDE_FLAMEGPU_SIM_AGENTLOGGINGCONFIG_REDUCTIONS_CUH_

#include <functional>

namespace flamegpu {
namespace detail {

/**
 * standard_deviation_add_impl is a manual expansion of FLAMEGPU_CUSTOM_REDUCTION()
 * This is required, so that the instance definition can be extern'd, as it is placed in a header
 */
struct standard_deviation_add_impl {
 public:
    /**
     * Calculates the mean for standard deviation
     * @param a 1st argument
     * @param b 2nd argument
     * @tparam OutT The return type
     */
    template <typename OutT>
    struct binary_function {
        __device__ __forceinline__ OutT operator()(const OutT &a, const OutT &b) const;
    };
};
/**
 * standard_deviation_subtract_mean_impl is a manual expansion of FLAMEGPU_CUSTOM_TRANSFORM()
 * This is required, so that the instance definition can be extern'd, as it is placed in a header
 */
struct standard_deviation_subtract_mean_impl {
 public:
     /**
      * Calculates pow(a - mean, 2) for standard deviation
      * @param a 1st argument
      * @tparam InT The input type
      * @tparam OutT The return type
      */
    template<typename InT, typename OutT>
    struct unary_function {
        __host__ __device__ OutT operator()(const InT &a) const;
    };
};
extern __constant__ double STANDARD_DEVIATION_MEAN;
extern std::mutex STANDARD_DEVIATION_MEAN_mutex;
extern standard_deviation_add_impl standard_deviation_add;
extern standard_deviation_subtract_mean_impl standard_deviation_subtract_mean;
template <typename OutT>
__device__ __forceinline__ OutT standard_deviation_add_impl::binary_function<OutT>::operator()(const OutT & a, const OutT & b) const {
    return a + b;
}
template<typename InT, typename OutT>
__device__ __forceinline__ OutT standard_deviation_subtract_mean_impl::unary_function<InT, OutT>::operator()(const InT &a) const {
    return pow(a - detail::STANDARD_DEVIATION_MEAN, 2.0);
}

}  // namespace detail

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

#endif  // INCLUDE_FLAMEGPU_SIM_AGENTLOGGINGCONFIG_REDUCTIONS_CUH_
