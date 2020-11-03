#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

// ! FLAMEGPU function return type
typedef void(AgentFunctionConditionWrapper)(
#ifndef NO_SEATBELTS
    DeviceExceptionBuffer *error_buffer,
#endif
    Curve::NamespaceHash instance_id_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const unsigned int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult);  // Can't put __global__ in a typedef

typedef void(AgentFunctionConditionEnsembleWrapper)(
#ifndef NO_SEATBELTS
    DeviceExceptionBuffer *error_buffer,
#endif
    unsigned int total_instances,
    unsigned int *instance_offsets,
    Curve::NamespaceHash *instance_id_hash,
    Curve::NamespaceHash agent_func_name_hash,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent function conditions
 * Initialises FLAMEGPU_API instance
 * @param error_buffer If provided, this buffer is used for reporting device exceptions.
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
#ifndef NO_SEATBELTS
    DeviceExceptionBuffer *error_buffer,
#endif
    Curve::NamespaceHash instance_id_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const unsigned int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult) {
#ifndef NO_SEATBELTS
    // We place this at the start of shared memory, so we can locate it anywhere in device code without a reference
    extern __shared__ DeviceExceptionBuffer *shared_mem[];
    shared_mem[0] = error_buffer;
#endif
    // Must be terminated here, else AgentRandom has bounds issues inside FLAMEGPU_DEVICE_API constructor
    if (FLAMEGPU_READ_ONLY_DEVICE_API::TID() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    FLAMEGPU_READ_ONLY_DEVICE_API api = FLAMEGPU_READ_ONLY_DEVICE_API(
        instance_id_hash,
        agent_func_name_hash,
        d_rng);

    // call the user specified device function
    {
        // Negate the return value, we want false at the start of the scattered array
        bool conditionResult = !(AgentFunctionCondition()(&api));
        // (scan flags will be processed to filter agents
        scanFlag_conditionResult[FLAMEGPU_READ_ONLY_DEVICE_API::TID()] = conditionResult;
    }
}
/**
 * Wrapper function for launching agent function conditions as part of an ensemble
 * Initialises FLAMEGPU_API instance
 * @param error_buffer If provided, this buffer is used for reporting device exceptions.
 * @param total_instances Total number of instances in this ensemble launch
 * @param instance_offsets Array of where each instance begins based on global thread index
 * @param instance_id_hash_array Array of instance id hashes, these should be different for each instance
 * @param agent_func_name_hash CURVE hash of the agent + function's names
 * @param d_rng Array of curand states for this kernel
 * @param scanFlag_conditionResult Scanflag array for condition result (this uses same buffer as agent death)
 * @tparam AgentFunctionCondition The modeller defined agent function condition (defined as FLAMEGPU_AGENT_FUNCTION_CONDITION in model code)
 * @note This is basically a cutdown version of agent_function_wrapper
 */
template<typename AgentFunctionCondition>
__global__ void agent_function_condition_ensemble_wrapper(
#ifndef NO_SEATBELTS
    DeviceExceptionBuffer *error_buffer,
#endif
    unsigned int total_instances,
    unsigned int *instance_offsets,
    Curve::NamespaceHash *instance_id_hash_array,
    Curve::NamespaceHash agent_func_name_hash,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult) {
#ifndef NO_SEATBELTS
    // We place this at the start of shared memory, so we can locate it anywhere in device code without a reference
    extern __shared__ DeviceExceptionBuffer *shared_mem[];
    shared_mem[0] = error_buffer;
#endif
    // Must be terminated here, else AgentRandom has bounds issues inside FLAMEGPU_DEVICE_API constructor
    if (FLAMEGPU_READ_ONLY_DEVICE_API::TID() >= instance_offsets[total_instances])
        return;
    // Perform a search to extract correct instance_id_hash
    // Note this cannot handle active instances with 0 length    
    Curve::NamespaceHash instance_id_hash;
    {
        unsigned int min = 0;
        unsigned int max = total_instances;

        while (min + 1 != max) {
            unsigned int tid = FLAMEGPU_READ_ONLY_DEVICE_API::TID();
            const unsigned int val = min + ((max-min)/2);
            const unsigned int test = instance_offsets[val];
            if (tid < test) {
                max = val;
            } else if (tid >= test) {
                min = val;
            }
        }
        instance_id_hash = instance_id_hash_array[min];
    }
    // create a new device FLAME_GPU instance
    FLAMEGPU_READ_ONLY_DEVICE_API api = FLAMEGPU_READ_ONLY_DEVICE_API(
        instance_id_hash,
        agent_func_name_hash,
        d_rng);

    // call the user specified device function
    {
        // Negate the return value, we want false at the start of the scattered array
        bool conditionResult = !(AgentFunctionCondition()(&api));
        // (scan flags will be processed to filter agents
        scanFlag_conditionResult[FLAMEGPU_READ_ONLY_DEVICE_API::TID()] = conditionResult;
    }
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
