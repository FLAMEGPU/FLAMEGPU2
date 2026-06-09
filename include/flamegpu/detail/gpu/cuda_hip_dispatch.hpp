#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_CUDA_HIP_DISPATCH_HPP_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_CUDA_HIP_DISPATCH_HPP_

// File abstracting CUDA/HIP calls into a single location, minimising use of disgusting macros at the cost of additional abstraction
// Where functions are a direct mapping, abbreviated function templates with std::forward & decltype makes this pretty clean
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

inline auto gpuMalloc(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(Malloc)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMallocManaged(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(MallocManaged)(std::forward<decltype(args)>(args)...);
}

inline auto gpuEventCreate(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(EventCreate)(std::forward<decltype(args)>(args)...);
}

inline auto gpuEventDestroy(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(EventDestroy)(std::forward<decltype(args)>(args)...);
}

inline auto gpuEventRecord(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(EventRecord)(std::forward<decltype(args)>(args)...);
}

inline auto gpuEventSynchronize(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(EventSynchronize)(std::forward<decltype(args)>(args)...);
}

inline auto gpuEventElapsedTime(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(EventElapsedTime)(std::forward<decltype(args)>(args)...);
}

}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_CUDA_HIP_DISPATCH_HPP_
