#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_

/**
 * This header exists to allow a convenient way to switch between curand implementations
 */

#include <curand_kernel.h>

namespace flamegpu {
namespace util {
namespace detail {

#if defined(FLAMEGPU_CURAND_MRG32k3a)
typedef curandStateMRG32k3a_t curandState;
#elif defined(FLAMEGPU_CURAND_XORWOW)
typedef curandStateXORWOW_t curandState;
#else  // defined(FLAMEGPU_CURAND_Philox4_32_10)
typedef curandStatePhilox4_32_10_t curandState;
#endif

}  // namespace detail
}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_
