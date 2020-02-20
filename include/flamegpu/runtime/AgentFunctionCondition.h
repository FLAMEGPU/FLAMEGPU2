#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

// ! FLAMEGPU function return type
typedef void(AgentFunctionConditionWrapper)(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const int popNo,
    const unsigned int thread_in_layer_offset,
    const unsigned int streamId);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param model_name_hash CURVE hash of the model's name
 * @param agent_func_name_hash CURVE hash of the agent + function's names
 * @param popNo Total number of agents exeucting the function (number of threads launched)
 * @param thread_in_layer_offset Add this value to TID to calculate a thread-safe TID (TS_ID), used by ActorRandom for accessing curand array in a thread-safe manner
 * @tparam AgentFunctionCondition The modeller defined agent function condition (defined as FLAMEGPU_AGENT_FUNCTION_CONDITION in model code)
 * @note This is basically a cutdown version of agent_function_wrapper
 */
template<typename AgentFunctionCondition>
__global__ void agent_function_condition_wrapper(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const int popNo,
    const unsigned int thread_in_layer_offset,
    const unsigned int streamId) {
    // Must be terminated here, else AgentRandom has bounds issues inside FLAMEGPU_DEVICE_API constructor
    if (FLAMEGPU_READ_ONLY_DEVICE_API::TID() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    FLAMEGPU_READ_ONLY_DEVICE_API *api = new FLAMEGPU_READ_ONLY_DEVICE_API(
        thread_in_layer_offset,
        model_name_hash,
        agent_func_name_hash,
        streamId);

    // call the user specified device function
    {
        // Negate the return value, we want false at the start of the scattered array
        bool conditionResult = !(AgentFunctionCondition()(api));
        // (scan flags will be processed to filter agents
        flamegpu_internal::CUDAScanCompaction::ds_agent_configs[streamId].scan_flag[FLAMEGPU_READ_ONLY_DEVICE_API::TID()] = conditionResult;
    }

    delete api;
}


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
