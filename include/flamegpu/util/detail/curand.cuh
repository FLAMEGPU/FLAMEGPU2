#ifndef INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_
#define INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_

/**
 * This header exists to allow a convenient way to switch between curand implementations
 */

#include <curand_kernel.h>

#if defined(CURAND_MRG32k3a)
typedef curandStateMRG32k3a_t curandStateFLAMEGPU;
#elif defined(CURAND_Philox4_32_10)
typedef curandStatePhilox4_32_10_t curandStateFLAMEGPU;
#else  // defined(CURAND_XORWOW)
typedef curandStateXORWOW_t curandStateFLAMEGPU;
#endif

#endif  // INCLUDE_FLAMEGPU_UTIL_DETAIL_CURAND_CUH_
