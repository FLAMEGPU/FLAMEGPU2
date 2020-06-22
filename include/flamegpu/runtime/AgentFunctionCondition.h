#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/runtime/AgentFunctionCondition_shim.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

// ! FLAMEGPU function return type
typedef void(AgentFunctionConditionWrapper)(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param model_name_hash CURVE hash of the model's name
 * @param agent_func_name_hash CURVE hash of the agent + function's names
 * @param popNo Total number of agents exeucting the function (number of threads launched)
 * @param d_rng Array of curand states for this kernel
 * @param scanFlag_conditionResult Scanflag array for condition result (this uses same buffer as agent death)
 * @tparam AgentFunctionCondition The modeller defined agent function condition (defined as FLAMEGPU_AGENT_FUNCTION_CONDITION in model code)
 * @note This is basically a cutdown version of agent_function_wrapper
 */
template<typename AgentFunctionCondition>
__global__ void agent_function_condition_wrapper(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    const int popNo,
    curandState *d_rng,
    unsigned int *scanFlag_conditionResult) {
    // Must be terminated here, else AgentRandom has bounds issues inside FLAMEGPU_DEVICE_API constructor
    if (FLAMEGPU_READ_ONLY_DEVICE_API::TID() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    FLAMEGPU_READ_ONLY_DEVICE_API *api = new FLAMEGPU_READ_ONLY_DEVICE_API(
        model_name_hash,
        agent_func_name_hash,
        d_rng);

    // call the user specified device function
    {
        // Negate the return value, we want false at the start of the scattered array
        bool conditionResult = !(AgentFunctionCondition()(api));
        // (scan flags will be processed to filter agents
        scanFlag_conditionResult[FLAMEGPU_READ_ONLY_DEVICE_API::TID()] = conditionResult;
    }

    delete api;
}


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_H_
