#ifndef INCLUDE_FLAMEGPU_RUNTIME_DETAIL_SHAREDBLOCK_H_
#define INCLUDE_FLAMEGPU_RUNTIME_DETAIL_SHAREDBLOCK_H_

namespace flamegpu {
namespace exception {
struct DeviceExceptionBuffer;
}  // namespace exception
namespace detail {
namespace curve {
struct CurveTable;
}  // namespace curve
/**
 * This struct represents the data we package into shared memory
 * The ifndef __CUDACC_RTC__ will cause the size to be too large for RTC builds, but that's not (currently) an issue
 */
struct SharedBlock {
#ifndef __CUDACC_RTC__
    const curve::CurveTable* curve;
    const char* env_buffer;
#endif
#if !defined(SEATBELTS) || SEATBELTS
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
