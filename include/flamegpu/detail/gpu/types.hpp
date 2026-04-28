#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_TYPES_HPP_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_TYPES_HPP_

#ifdef FLAMEGPU_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#ifdef FLAMEGPU_USE_HIP
#include <hip/hip_runtime.h>
#endif

namespace flamegpu {
namespace detail {

/**
 * Internal (detail) namespace abstracting differences between CUDA and HIP
 * 
 * Todo: Improve this
 * Todo: Consider where Stream_t should be defined (seeing as it is used in parts of the public api I.e. HostAPI::HostAPI). Should this actually be flamegpu::gpu for types with other parts in flamegpu::gpu::detail instead?
 */
namespace gpu {

// Using statement for cuda/hip streams, which are part of the public API
// Todo: should this just use the macro instead? Should this be in detail seeing as it is used in parts of the (sortof) public API?
#if defined(FLAMEGPU_USE_CUDA)
using Stream_t = cudaStream_t;
#elif defined(FLAMEGPU_USE_HIP)
using Stream_t = hipStream_t;
#else
// naked struct for intellisense, this should never occur for actual compilation Todo: this is not correct.
#error "CUDA/HIP required"
typedef struct stream* Stream_t;
#endif


// Using statement for cuda/hip error_t, which is part of the private API?
// Todo: Should this just use the macro instead?
// Todo: Move type definitions and macros for this to a separate header for lighter includes? flamegpu/gpu/types.h or similar, and then flamegpu/gpu/macros.h and flamegpu/gpu/
// Should this actually be in detail?
#if defined(FLAMEGPU_USE_CUDA)
using Error_t = cudaError_t;
#elif defined(FLAMEGPU_USE_HIP)
using Error_t = hipError_t;
#else
// naked struct for intellisense, this should never occur for actual compilation. Todo: this is not correct.
typedef struct error* Error_t;
#endif

// pointerAttributes is _t in hip :(
#if defined(FLAMEGPU_USE_CUDA)
using PointerAttributes_t = cudaPointerAttributes;
#elif defined(FLAMEGPU_USE_HIP)
using PointerAttributes_t = hipPointerAttribute_t;
#else
// naked struct for intellisense, this should never occur for actual compilation Todo: this is not correct.
typedef struct pointerAttribtues* PointerAttributes_t;
#endif

}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_TYPES_HPP_
