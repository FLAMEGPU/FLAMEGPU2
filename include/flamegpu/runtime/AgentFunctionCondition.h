#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/DeviceAPI.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

// ! FLAMEGPU function return type
typedef void(AgentFunctionConditionWrapper)(
#if !defined(SEATBELTS) || SEATBELTS
    DeviceExceptionBuffer *error_buffer,
#endif
    Curve::NamespaceHash instance_id_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const unsigned int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param instance_id_hash CURVE hash of the CUDASimulation's instance id
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
    DeviceExceptionBuffer *error_buffer,
#endif
    Curve::NamespaceHash instance_id_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const unsigned int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult) {
#if !defined(SEATBELTS) || SEATBELTS
    // We place this at the start of shared memory, so we can locate it anywhere in device code without a reference
    extern __shared__ DeviceExceptionBuffer *shared_mem[];
    if (threadIdx.x == 0) {
        shared_mem[0] = error_buffer;
    }
#endif
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


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
