#ifndef INCLUDE_FLAMEGPU_DETAIL_CUDAEVENTTIMER_CUH_
#define INCLUDE_FLAMEGPU_DETAIL_CUDAEVENTTIMER_CUH_

#include "flamegpu/detail/GPUEventTimer.cuh"


namespace flamegpu {
namespace detail {

using CUDAEventTimer [[deprecated("CUDAEventTimer is deprecated; use GPUEventTimer instead.")]] = GPUEventTimer;


}  // namespace detail
}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_DETAIL_CUDAEVENTTIMER_CUH_
