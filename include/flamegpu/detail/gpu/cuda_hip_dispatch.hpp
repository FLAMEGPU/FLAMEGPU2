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

// Functions

inline auto gpuGetDeviceCount(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceCount)(std::forward<decltype(args)>(args)...);
}

inline auto gpuGetDevice(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDevice)(std::forward<decltype(args)>(args)...);
}

inline auto gpuGetDeviceProperties(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(GetDeviceProperties)(std::forward<decltype(args)>(args)...);
}

inline auto gpuDeviceGetAttribute(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceGetAttribute)(std::forward<decltype(args)>(args)...);
}

inline auto gpuSetDevice(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(SetDevice)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMalloc(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(Malloc)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMallocManaged(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(MallocManaged)(std::forward<decltype(args)>(args)...);
}

inline auto gpuHostAlloc(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(HostAlloc)(std::forward<decltype(args)>(args)...);
}

inline auto gpuFree(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(Free)(std::forward<decltype(args)>(args)...);
}

inline auto gpuFreeHost(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(FreeHost)(std::forward<decltype(args)>(args)...);
}

inline auto gpuPointerGetAttributes(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(PointerGetAttributes)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMemcpy(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(Memcpy)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMemcpyAsync(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyAsync)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMemcpyToSymbol(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyToSymbol)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMemcpyToSymbolAsync(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyToSymbolAsync)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMemcpyFromSymbol(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyFromSymbol)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMemset(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(Memset)(std::forward<decltype(args)>(args)...);
}

inline auto gpuMemsetAsync(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(MemsetAsync)(std::forward<decltype(args)>(args)...);
}

inline auto gpuDeviceReset(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceReset)(std::forward<decltype(args)>(args)...);
}

inline auto gpuDeviceSynchronize(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceSynchronize)(std::forward<decltype(args)>(args)...);
}

inline auto gpuGetErrorName(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(GetErrorName)(std::forward<decltype(args)>(args)...);
}

inline auto gpuGetErrorString(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(GetErrorString)(std::forward<decltype(args)>(args)...);
}

inline auto gpuGetLastError(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(GetLastError)(std::forward<decltype(args)>(args)...);
}

inline auto gpuPeekAtLastError(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(PeekAtLastError)(std::forward<decltype(args)>(args)...);
}

inline auto gpuStreamCreate(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(StreamCreate)(std::forward<decltype(args)>(args)...);
}

inline auto gpuStreamDestroy(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(StreamDestroy)(std::forward<decltype(args)>(args)...);
}

inline auto gpuStreamSynchronize(auto&&... args) {
    return FLAMEGPU_GPU_RUNTIME_SYMBOL(StreamSynchronize)(std::forward<decltype(args)>(args)...);
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

// Enums
inline constexpr auto gpuSuccess = FLAMEGPU_GPU_RUNTIME_SYMBOL(Success);

inline constexpr auto gpuMemcpyHostToDevice = FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyHostToDevice);
inline constexpr auto gpuMemcpyDeviceToHost = FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyDeviceToHost);
inline constexpr auto gpuMemcpyDeviceToDevice = FLAMEGPU_GPU_RUNTIME_SYMBOL(MemcpyDeviceToDevice);
inline constexpr auto gpuMemoryTypeHost = FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeHost);
inline constexpr auto gpuMemoryTypeDevice = FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeDevice);
inline constexpr auto gpuMemoryTypeUnregistered = FLAMEGPU_GPU_RUNTIME_SYMBOL(MemoryTypeUnregistered);
inline constexpr auto gpuHostAllocDefault = FLAMEGPU_GPU_RUNTIME_SYMBOL(HostAllocDefault);


}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_CUDA_HIP_DISPATCH_HPP_
