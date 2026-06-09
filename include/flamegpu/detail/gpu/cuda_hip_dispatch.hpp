#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_CUDA_HIP_DISPATCH_HPP_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_CUDA_HIP_DISPATCH_HPP_

// File abstracting CUDA/HIP calls into a single location, minimising use of disgusting macros at the cost of additional abstraction
// Where functions are a direct mapping, variadic templating and std::forward makes this fairly copy paste
// Where the implementations differ require a bit more effort (but that was already the case).

#include <utility>

#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/gpu/types.hpp"

#if defined(FLAMEGPU_USE_HIP)
#include <hip/hip_runtime.h>
#else  // if defined(FLAMEGPU_USE_CUDA)
#include <cuda_runtime.h>
#include <cuda.h>
#endif

namespace flamegpu {
namespace detail {
namespace gpu {

// todo: c++20 auto&& instead, no need for tempalte?
// auto foo(auto&&... args) {std::forward<decltype(args)>(args)...)

template <typename... Args>
inline flamegpu::detail::gpu::Error_t gpuEventCreate(Args&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(EventCreate)(std::forward<Args>(args)...);
}

}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_CUDA_HIP_DISPATCH_HPP_
