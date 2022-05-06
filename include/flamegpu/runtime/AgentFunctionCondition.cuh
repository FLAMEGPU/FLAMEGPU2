#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_CUH_

#include <cuda_runtime.h>

#include "flamegpu/runtime/DeviceAPI.cuh"
#include "flamegpu/runtime/AgentFunctionCondition_shim.cuh"

namespace flamegpu {

// ! FLAMEGPU function return type
typedef void(AgentFunctionConditionWrapper)(
#if !defined(SEATBELTS) || SEATBELTS
    exception::DeviceExceptionBuffer *error_buffer,
#endif
    detail::curve::Curve::NamespaceHash instance_id_hash,
#ifndef __CUDACC_RTC__
    const detail::curve::Curve::CurveTable* d_curve_table,
#endif
    detail::curve::Curve::NamespaceHash agent_func_name_hash,
    const unsigned int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param error_buffer Buffer used for detecting and reporting exception::DeviceErrors (flamegpu must be built with SEATBELTS enabled for this to be used)
 * @param instance_id_hash CURVE hash of the CUDASimulation's instance id
 * @param d_curve_table Pointer to curve hash table in device memory
 * @param agent_func_name_hash CURVE hash of the agent + function's names
 * @param popNo Total number of agents exeucting the function (number of threads launched)
 * @param d_rng Array of curand states for this kernel
 * @param scanFlag_conditionResult Scanflag array for condition result (this uses same buffer as agent death)
 * @tparam AgentFunctionCondition The modeller defined agent function condition (defined as FLAMEGPU_AGENT_FUNCTION_CONDITION in model code)
 * @note This is basically a cutdown version of agent_function_wrapper
 */
template<typename AgentFunctionCondition>
__global__ void agent_function_condition_wrapper(
#if !defined(SEATBELTS) || SEATBELTS
    exception::DeviceExceptionBuffer *error_buffer,
#endif
    detail::curve::Curve::NamespaceHash instance_id_hash,
#ifndef __CUDACC_RTC__
    const detail::curve::Curve::CurveTable* d_curve_table,
#endif
    detail::curve::Curve::NamespaceHash agent_func_name_hash,
    const unsigned int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult) {
    // We place these at the start of shared memory, so we can locate it anywhere in device code without a reference
    if (threadIdx.x == 0) {
        extern __shared__ const void* sm[];
#ifndef __CUDACC_RTC__
        sm[0] = d_curve_table;
#endif
#if !defined(SEATBELTS) || SEATBELTS
        sm[1] = error_buffer;
#endif
    }

#if defined(__CUDACC__)  // @todo - This should not be required. This template should only ever be processed by a CUDA compiler.
    // Sync the block after Thread 0 has written to shared.
    __syncthreads();
#endif  // __CUDACC__
    // Must be terminated here, else AgentRandom has bounds issues inside DeviceAPI constructor
    if (ReadOnlyDeviceAPI::getThreadIndex() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    ReadOnlyDeviceAPI api = ReadOnlyDeviceAPI(
        instance_id_hash,
        agent_func_name_hash,
        d_rng);

    // call the user specified device function
    {
        // Negate the return value, we want false at the start of the scattered array
        bool conditionResult = !(AgentFunctionCondition()(&api));
        // (scan flags will be processed to filter agents
        scanFlag_conditionResult[ReadOnlyDeviceAPI::getThreadIndex()] = conditionResult;
    }
}

}  // namespace flamegpu


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_CUH_
