#ifndef INCLUDE_FLAMEGPU_DETAIL_GPU_TYPES_HPP_
#define INCLUDE_FLAMEGPU_DETAIL_GPU_TYPES_HPP_

#ifdef FLAMEGPU_USE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#ifdef FLAMEGPU_USE_HIP
#include <hip/hip_runtime.h>
#endif

#include "flamegpu/detail/gpu/macros.hpp"

namespace flamegpu {
namespace detail {

/**
 * Internal (detail) namespace abstracting differences between CUDA and HIP
 * 
 *  * Todo: Consider where Stream_t should be defined (seeing as it is used in parts of the public api I.e. HostAPI::HostAPI). Should this actually be flamegpu::gpu for types with other parts in flamegpu::gpu::detail instead?
 */
namespace gpu {

/**
 * Abstraction for cudaStream_t or hipStream_t as appropriate
 * 
 * Todo: Should this be in detail given it is used as part of the public (ish) API?
 */
using Stream_t = FLAMEGPU_GPU_RUNTIME_SYMBOL(Stream_t);

/**
 * Abstraction for cudaError_t or hipError_t as appropriate
 * 
 * Todo: Should this be in detail given it is used as part of the public (ish) API?
 */
using Error_t = FLAMEGPU_GPU_RUNTIME_SYMBOL(Error_t);

/**
 * Abstraction for cudaPointerAttributes or hipPointerAttribute_t as appropriate
 * 
 * Note: This is different between CUDA and HIP, 
 */
#if defined(FLAMEGPU_USE_HIP)
using PointerAttributes_t = hipPointerAttribute_t;
#else  // if defined(FLAMEGPU_USE_CUDA)
using PointerAttributes_t = cudaPointerAttributes;
#endif

/**
 * Abstraction for cudaEvent_t or hipEvent_t as appropriate
 */
using Event_t = FLAMEGPU_GPU_RUNTIME_SYMBOL(Event_t);

}  // namespace gpu
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_GPU_TYPES_HPP_
