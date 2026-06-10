#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_RAND_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_RAND_CUH_

// This header exists to allow a convenient way to switch between curand implementations

#include <utility>

#ifdef FLAMEGPU_USE_CUDA
#include <curand_kernel.h>
#elif FLAMEGPU_USE_HIP
#include <hiprand_kernel.h>
#endif

#include "flamegpu/detail/gpu/macros.hpp"

namespace flamegpu {
namespace detail {
namespace gpu {

#if defined(FLAMEGPU_CURAND_MRG32k3a)
typedef FLAMEGPU_GPU_DRIVER_SYMBOL(randStateMRG32k3a_t) gpurandState;
#elif defined(FLAMEGPU_CURAND_XORWOW)
typedef FLAMEGPU_GPU_DRIVER_SYMBOL(randStateXORWOW_t) gpurandState;
#else  // defined(FLAMEGPU_CURAND_Philox4_32_10)
typedef FLAMEGPU_GPU_DRIVER_SYMBOL(randStatePhilox4_32_10_t) gpurandState;
#endif

// lightly-wrapped/abstracted curand/hiprand functions used within core FLAMEGPU
__device__ inline auto gpurand(auto&&... args) {
    return FLAMEGPU_GPU_DRIVER_SYMBOL(rand)(std::forward<decltype(args)>(args)...);
}

__device__ inline auto gpurand_normal(auto&&... args) {
    return FLAMEGPU_GPU_DRIVER_SYMBOL(rand_normal)(std::forward<decltype(args)>(args)...);
}

__device__ inline auto gpurand_normal_double(auto&&... args) {
    return FLAMEGPU_GPU_DRIVER_SYMBOL(rand_normal_double)(std::forward<decltype(args)>(args)...);
}

__device__ inline auto gpurand_log_normal(auto&&... args) {
    return FLAMEGPU_GPU_DRIVER_SYMBOL(rand_log_normal)(std::forward<decltype(args)>(args)...);
}

__device__ inline auto gpurand_log_normal_double(auto&&... args) {
    return FLAMEGPU_GPU_DRIVER_SYMBOL(rand_log_normal_double)(std::forward<decltype(args)>(args)...);
}

__device__  inline auto gpurand_poisson(auto&&... args) {
    return FLAMEGPU_GPU_DRIVER_SYMBOL(rand_poisson)(std::forward<decltype(args)>(args)...);
}

__device__  inline auto gpurand_init(auto&&... args) {
    return FLAMEGPU_GPU_DRIVER_SYMBOL(rand_init)(std::forward<decltype(args)>(args)...);
}

}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_RAND_CUH_
