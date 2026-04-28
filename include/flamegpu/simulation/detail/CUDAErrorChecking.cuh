#ifndef INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAERRORCHECKING_CUH_
#define INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAERRORCHECKING_CUH_

#ifndef __CUDACC_RTC__

#ifdef FLAMEGPU_USE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include <source_location>
#include <string>
// #include <stdexcept>
#include "flamegpu/exception/FLAMEGPUException.h"
#include "flamegpu/detail/gpu/macros.hpp"
#include "flamegpu/detail/gpu/types.hpp"

namespace flamegpu {
namespace detail {

/**
 * Error check function for safe CUDA API calling
 * @param code CUDA Runtime API return code
 * @param file File where errorcode was reported (e.g. __FILE__)
 * @param line Line no where errorcode was reported (e.g. __LINE__)
 * @throws CUDAError If code != cudaSuccess
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
  * Error check function for checking for the most recent error
  * @param file File where errorcode was reported (e.g. __FILE__)
  * @param line Line no where errorcode was reported (e.g. __LINE__)
  * @throws CUDAError If code != cudaSuccess
  * @see gpuAssert(cudaError_t code, const char *file, int line)
  * @note Only synchronises in debug builds
  */
inline void gpuLaunchAssert(const char *file, int line) {
#ifdef _DEBUG
    gpuAssert(FLAMEGPU_GPU_RUNTIME_SYMBOL(DeviceSynchronize)(), file, line);
#endif
    gpuAssert(FLAMEGPU_GPU_RUNTIME_SYMBOL(PeekAtLastError)(), file, line);
}

/**
 * Error check function for safe CUDA API calling, using source_location for a function rather than macro, avoiding global namespace pollution
 * 
 * @param code CUDA Error type, templated for runtime or device API calls.
 * @param loc std::source_location object, which when current() is used as the default argument the value represents the call site
 * @throws CUDAError if code != cudaSuccess
 */
template <typename T>
inline void gpuCheck(T code, const std::source_location loc = std::source_location::current()) {
    flamegpu::detail::gpuAssert(code, loc.file_name(), loc.line());
}

/**
 * Error checkinfg function for checking the most recent error, i.e. after a kernel launch. Non-macro variant using source_location
 * @throws CUDAError If code != cudaSuccess
 * @see gpuAssert(cudaError_t code, const char *file, int line)
 * @note Only synchronises in debug builds
 */
inline void gpuCheckLaunch(const std::source_location loc = std::source_location::current()) {
    flamegpu::detail::gpuLaunchAssert(loc.file_name(), loc.line());
}


}  // namespace detail
}  // namespace flamegpu

#endif  // __CUDACC_RTC__

#endif  // INCLUDE_FLAMEGPU_SIMULATION_DETAIL_CUDAERRORCHECKING_CUH_
