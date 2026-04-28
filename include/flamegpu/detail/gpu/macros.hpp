#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_MACROS_HPP_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_MACROS_HPP_

/**
 * Macros used for the thin CUDA/HIP abstraction.
 * Todo: This should be temporary, and combined into a single flamegpu/detail/gpu header which is the only place the macros are used? 
 */

#if defined(FLAMEGPU_USE_CUDA)
#include <cuda_runtime.h>
#include <cuda.h>
#endif 

#if defined(FLAMEGPU_USE_HIP)
#include <hip/hip_runtime.h>
#endif

#if defined(FLAMEGPU_USE_CUDA)
/**
 * The lowercase prefix for CUDA/HIP runtime api symbols
 */
#define FLAMEGPU_GPU_RUNTIME_PREFIX cuda
/**
 * The lowercase prefix for CUDA/HIP driver api symbols
 */
#define FLAMEGPU_GPU_DRIVER_PREFIX cu
/**
 * The uppercase prefix for CUDA/HIP runtime api symbols
 */
#define FLAMEGPU_GPU_RUNTIME_PREFIX_UPPER CUDA
/**
 * The uppercase prefix for CUDA/HIP driver api symbols
 */
#define FLAMEGPU_GPU_DRIVER_PREFIX_UPPER CU

#elif defined(FLAMEGPU_USE_HIP)

/**
 * The lowercase prefix for CUDA/HIP runtime api symbols
 */
#define FLAMEGPU_GPU_RUNTIME_PREFIX hip
/**
 * The lowercase prefix for CUDA/HIP driver api symbols
 */
#define FLAMEGPU_GPU_DRIVER_PREFIX hip
/**
 * The uppercase prefix for CUDA/HIP runtime api symbols
 */
#define FLAMEGPU_GPU_RUNTIME_PREFIX_UPPER HIP
/**
 * The uppercase prefix for CUDA/HIP driver api symbols
 */
#define FLAMEGPU_GPU_DRIVER_PREFIX_UPPER HIP

#else
// Raise a compiler error if this file is included without CUDA or HIP enabled?
// Todo: This breaks highlighting (when it is not aware of these defines). Maybe assume CUDA instead?
#error "CUDA or HIP must be enabled"

#endif

/**
 * Internal macro for string concatenation (inner of 2)
 */
#define FLAMEGPU_GPU_CONCAT_INNER(a, b) a ## b
/**
 * Internal macro for string concatenation (outer of 2)
 */
#define FLAMEGPU_GPU_CONCAT(a, b) FLAMEGPU_GPU_CONCAT_INNER(a, b)

/**
 * Macro for getting the cuda/hip version of a CUDA runtime api method. This is not fully portable due to differences between CUDA and HIP
 */
#define FLAMEGPU_GPU_RUNTIME_SYMBOL(STMT) FLAMEGPU_GPU_CONCAT(FLAMEGPU_GPU_RUNTIME_PREFIX, STMT)
/**
 * Macro for getting the cuda/hip version of a CUDA driver api method. This is not fully portable due to differences between CUDA and HIP
 */
#define FLAMEGPU_GPU_DRIVER_SYMBOL(STMT) FLAMEGPU_GPU_CONCAT(FLAMEGPU_GPU_DRIVER_PREFIX, STMT)

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_MACROS_HPP_
