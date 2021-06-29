#ifndef INCLUDE_FLAMEGPU_GPU_CUDAERRORCHECKING_CUH_
#define INCLUDE_FLAMEGPU_GPU_CUDAERRORCHECKING_CUH_

// #include <device_launch_parameters.h>
// #include <cuda_runtime.h>
#include <cuda.h>

#include <string>
// #include <stdexcept>
#include "flamegpu/exception/FLAMEGPUException.h"

namespace flamegpu {

/**
 * Error check function for safe CUDA API calling
 * Wrap any cuda runtime API calls with this macro to automatically check the returned cudaError_t
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
/**
 * Error check function for safe CUDA API calling
 * @param code CUDA Runtime API return code
 * @param file File where errorcode was reported (e.g. __FILE__)
 * @param line Line no where errorcode was reported (e.g. __LINE__)
 * @throws CUDAError If code != cudaSuccess
 */
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        THROW exception::CUDAError("CUDA Error: %s(%d): %s", file, line, cudaGetErrorString(code));
    }
}

/**
 * Error check function for safe CUDA Driver API calling
 * Wrap any cuda drive API calls with this macro to automatically check the returned CUresult
 */
#define gpuErrchkDriverAPI(ans) { gpuAssert((ans), __FILE__, __LINE__); }
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

/**
 * Error check function for use after asynchronous methods
 * Call this macro function after async calls such as kernel launches to automatically check the latest error
 * In debug builds this will perform a synchronisation to catch any errors, in non-debug builds errors may be propagated.
 */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
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
    gpuAssert(cudaDeviceSynchronize(), file, line);
#endif
    gpuAssert(cudaPeekAtLastError(), file, line);
}

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_GPU_CUDAERRORCHECKING_CUH_
