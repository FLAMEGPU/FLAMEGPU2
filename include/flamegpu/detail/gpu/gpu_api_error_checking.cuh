#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_GPU_API_ERROR_CHECKING_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_GPU_API_ERROR_CHECKING_CUH_

#ifndef __CUDACC_RTC__

#ifdef FLAMEGPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <source_location>
#include <string>
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/gpu/types.hpp"

namespace flamegpu {
namespace detail {

/**
 * Error check function for safe CUDA/HIP API calling
 * @param code CUDA/HIP Runtime API return code
 * @param file File where errorcode was reported (e.g. __FILE__)
 * @param line Line no where errorcode was reported (e.g. __LINE__)
 * @throws CUDAError If code != cudaSuccess / hipSuccess
 */
inline void gpuAssert(flamegpu::detail::gpu::Error_t code, const char *file, int line) {
    if (code != FLAMEGPU_GPU_RUNTIME_SYMBOL(Success)) {
        THROW exception::CUDAError("CUDA Error: %s(%d): %s %s", file, line, FLAMEGPU_GPU_RUNTIME_SYMBOL(GetErrorName)(code), FLAMEGPU_GPU_RUNTIME_SYMBOL(GetErrorString)(code));
    }
}

#ifdef FLAMEGPU_USE_CUDA
 /**
  * Error check function for safe CUDA API calling
  * @param code CUDA Driver API return code
  * @param file File where errorcode was reported (e.g. __FILE__)
  * @param line Line no where errorcode was reported (e.g. __LINE__)
  * @throws CUDAError If code != CUDA_SUCCESS
  */
inline void gpuAssert(CUresult code, const char* file, int line) {
    if (code != CUDA_SUCCESS) {
        const char *error_str;
        THROW exception::CUDAError("CUDA Driver Error: %s(%d): %s", file, line, cuGetErrorString(code, &error_str));
    }
}
#endif

 /**
  * Error check function for checking for the most recent GPU API error
  * @param file File where errorcode was reported (e.g. __FILE__)
  * @param line Line no where errorcode was reported (e.g. __LINE__)
  * @throws CUDAError If If code != cudaSuccess / hipSuccess
  * @see gpuAssert(flamegpu::detail::gpu::Error_t code, const char *file, int line)
  * @note Only synchronises in debug builds
  */
inline void gpuLaunchAssert(const char *file, int line) {
#ifdef _DEBUG
    gpuAssert(FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceSynchronize)(), file, line);
#endif
    gpuAssert(FLAMEGPU_GPU_RUNTIME_SYMBOL(PeekAtLastError)(), file, line);
}

/**
 * Error check function for safe CUDA/HIP API calling
 * 
 * Uses source_location so this can be a namespace-scoped function rather than macro
 * 
 * @param code CUDA/HIP Error type, templated for runtime or device API calls.
 * @param loc std::source_location object, which when current() is used as the default argument the value represents the call site
 * @throws CUDAError if If code != cudaSuccess / hipSuccess
 */
template <typename T>
inline void gpuCheck(T code, const std::source_location loc = std::source_location::current()) {
    flamegpu::detail::gpuAssert(code, loc.file_name(), loc.line());
}

/**
 * Error checking function for checking the most recent error, i.e. after a kernel launch
 * 
 * Uses source_location so this can be a namespace-scoped function rather than macro
 * @throws CUDAError If code != cudaSuccess / hipSuccess
 * @see gpuAssert(flamegpu::detail::gpu::Error_t code, const char *file, int line)
 * @note Only synchronises in debug builds
 */
inline void gpuCheckLaunch(const std::source_location loc = std::source_location::current()) {
    flamegpu::detail::gpuLaunchAssert(loc.file_name(), loc.line());
}

}  // namespace detail
}  // namespace flamegpu

#endif  // __CUDACC_RTC__
#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_GPU_API_ERROR_CHECKING_CUH_
