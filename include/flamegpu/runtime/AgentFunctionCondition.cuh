#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_CUH_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/DeviceAPI.cuh"
#include "flamegpu/runtime/AgentFunctionCondition_shim.cuh"
#ifndef __CUDACC_RTC__
#include "flamegpu/runtime/detail/curve/DeviceCurve.cuh"
#endif

namespace flamegpu {

// ! FLAMEGPU function return type
typedef void(AgentFunctionConditionWrapper)(
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    exception::DeviceExceptionBuffer *error_buffer,
#endif
#ifndef __CUDACC_RTC__
    const detail::curve::CurveTable* d_curve_table,
    const char* d_env_buffer,
#endif
    const unsigned int popNo,
    detail::curandState *d_rng,
    unsigned int *scanFlag_conditionResult);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param error_buffer Buffer used for detecting and reporting exception::DeviceErrors (flamegpu must be built with FLAMEGPU_SEATBELTS enabled for this to be used)
 * @param d_curve_table Pointer to curve hash table in device memory
 * @param d_env_buffer Pointer to env buffer in device memory
 * @param popNo Total number of agents exeucting the function (number of threads launched)
 * @param d_rng Array of curand states for this kernel
 * @param scanFlag_conditionResult Scanflag array for condition result (this uses same buffer as agent death)
 * @tparam AgentFunctionCondition The modeller defined agent function condition (defined as FLAMEGPU_AGENT_FUNCTION_CONDITION in model code)
 * @note This is basically a cutdown version of agent_function_wrapper
 */
template<typename AgentFunctionCondition>
__global__ void agent_function_condition_wrapper(
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
    exception::DeviceExceptionBuffer *error_buffer,
#endif
#ifndef __CUDACC_RTC__
    const detail::curve::CurveTable* __restrict__ d_curve_table,
    const char* d_env_buffer,
#endif
    const unsigned int popNo,
    detail::curandState *d_rng,
    unsigned int *scanFlag_conditionResult) {
    // We place these at the start of shared memory, so we can locate it anywhere in device code without a reference
    using detail::sm;
    if (threadIdx.x == 0) {
#if !defined(FLAMEGPU_SEATBELTS) || FLAMEGPU_SEATBELTS
        sm()->device_exception = error_buffer;
#endif
#ifndef __CUDACC_RTC__
        sm()->env_buffer = d_env_buffer;
#endif
    }
#ifndef __CUDACC_RTC__
    detail::curve::DeviceCurve::init(d_curve_table);
#endif

#if defined(__CUDACC__)  // @todo - This should not be required. This template should only ever be processed by a CUDA compiler.
    // Sync the block after Thread 0 has written to shared.
    __syncthreads();
#endif  // __CUDACC__
    // Must be terminated here, else AgentRandom has bounds issues inside DeviceAPI constructor
    if (ReadOnlyDeviceAPI::getIndex() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    ReadOnlyDeviceAPI api = ReadOnlyDeviceAPI(d_rng);

    // call the user specified device function
    {
        // Negate the return value, we want false at the start of the scattered array
        bool conditionResult = !(AgentFunctionCondition()(&api));
        // (scan flags will be processed to filter agents
        scanFlag_conditionResult[ReadOnlyDeviceAPI::getIndex()] = conditionResult;
    }
}

}  // namespace flamegpu


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_CUH_
