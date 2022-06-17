#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_

/**
 * This header exists to allow a convenient way to switch between curand implementations
 */

#include <curand_kernel.h>

namespace flamegpu {
namespace util {
namespace detail {

#if defined(CURAND_MRG32k3a)
typedef curandStateMRG32k3a_t curandState;
#elif defined(CURAND_Philox4_32_10)
typedef curandStatePhilox4_32_10_t curandState;
#else  // defined(CURAND_XORWOW)
typedef curandStateXORWOW_t curandState;
#endif

}  // namespace detail
}  // namespace util
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_
