#ifndef INCLUDE_FLAMEGPU_DETAIL_CURAND_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_CURAND_CUH_

// This header exists to allow a convenient way to switch between curand implementations
// todo: rename curandState gpurandState or similar

#ifdef FLAMEGPU_USE_CUDA
#include <curand_kernel.h>
#elif FLAMEGPU_USE_HIP
#include <hiprand_kernel.h>
#endif

namespace flamegpu {
namespace detail {

#ifdef FLAMEGPU_USE_CUDA
#if defined(FLAMEGPU_CURAND_MRG32k3a)
typedef curandStateMRG32k3a_t curandState;
#elif defined(FLAMEGPU_CURAND_XORWOW)
typedef curandStateXORWOW_t curandState;
#else  // defined(FLAMEGPU_CURAND_Philox4_32_10)
typedef curandStatePhilox4_32_10_t curandState;
#endif
#elif FLAMEGPU_USE_HIP
#if defined(FLAMEGPU_CURAND_MRG32k3a)
typedef hipandStateMRG32k3a_t curandState;
#elif defined(FLAMEGPU_CURAND_XORWOW)
typedef hiprandStateXORWOW_t curandState;
#else  // defined(FLAMEGPU_CURAND_Philox4_32_10)
typedef hiprandStatePhilox4_32_10_t curandState;
#endif
#endif

}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_CURAND_CUH_
