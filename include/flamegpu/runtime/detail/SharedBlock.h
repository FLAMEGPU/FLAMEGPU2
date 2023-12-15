#ifndef INCLUDE_FLAMEGPU_RUNTIME_DETAIL_SHAREDBLOCK_H_
#define INCLUDE_FLAMEGPU_RUNTIME_DETAIL_SHAREDBLOCK_H_

#include "flamegpu/runtime/detail/curve/Curve.cuh"

namespace flamegpu {
namespace exception {
struct DeviceExceptionBuffer;
}  // namespace exception
namespace detail {
/**
 * This struct represents the data we package into shared memory
 * The ifndef __CUDACC_RTC__ will cause the size to be too large for RTC builds, but that's not (currently) an issue
 */
struct SharedBlock {
#ifndef __CUDACC_RTC__
    curve::Curve::VariableHash curve_hashes[curve::Curve::MAX_VARIABLES];
    char* curve_variables[curve::Curve::MAX_VARIABLES];
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    unsigned int curve_type_size[curve::Curve::MAX_VARIABLES];
    unsigned int curve_elements[curve::Curve::MAX_VARIABLES];
#endif
    unsigned int curve_count[curve::Curve::MAX_VARIABLES];
    const char* env_buffer;
    const char* agent_name;
    const char* state_name;
#endif
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    exception::DeviceExceptionBuffer *device_exception;
#endif
};
/**
 * Returns a pointer to a common item in shared memory
 */
__forceinline__ __device__ SharedBlock *sm() {
#ifdef __CUDACC__
    __shared__ SharedBlock _sm;
    return &_sm;
#else
    // GCC causes this function to warn, as it believes a ptr to local memory is being returned (return-local-addr)
    return nullptr;
#endif
}
}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_DETAIL_SHAREDBLOCK_H_
