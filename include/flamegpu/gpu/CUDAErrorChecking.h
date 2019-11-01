 /**
 * @file CUDAErrorChecking.h
 * @authors Paul
 * @date 5 Mar 2014
 * @brief
 *
 * @see
 * @warning
 */

#pragma once


#include <string>
#include <stdexcept>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


/* Error check function for safe CUDA API calling */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line)
{
   if (code != cudaSuccess)
   {
	  throw std::runtime_error("CUDA Error: " + std::string(cudaGetErrorString(code)) + " " + file + " " + std::to_string(line));
   }
}

/* Error check function for post CUDA Kernel calling */
#define gpuErrchkLaunch() { gpuLaunchAssert(__FILE__, __LINE__); }
inline void gpuLaunchAssert(const char *file, int line)
{
#ifdef _DEBUG
	gpuAssert( cudaDeviceSynchronize(), file, line );
#endif
    gpuAssert( cudaPeekAtLastError(), file, line );
}
