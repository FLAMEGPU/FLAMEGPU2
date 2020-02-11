#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

typedef void(AgentFunctionWrapper)(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    const int popNo,
    const unsigned int messageList_size,
    const unsigned int thread_in_layer_offset,
    const unsigned int streamId);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param model_name_hash Required by DeviceEnvironment
 * @param agent_func_name_hash
 * @param messagename_inp_hash
 * @param popNo
 * @param messageList_size
 * @param thread_in_layer_offset Add this value to TID to calculate a thread-safe TID (TS_ID), used by ActorRandom for accessing curand array in a thread-safe manner
 * @tparam AgentFunction The modeller defined agent function
 * @tparam MsgIn Message handler for messageinput
 * @tparam MsgOut Message handler for messageoutput
 */
template<typename AgentFunction, typename MsgIn, typename MsgOut>
__global__ void agent_function_wrapper(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    const int popNo,
    const unsigned int messageList_size,
    const unsigned int thread_in_layer_offset,
    const unsigned int streamId) {
    // Must be terminated here, else AgentRandom has bounds issues inside FLAMEGPU_DEVICE_API constructor
    if (FLAMEGPU_DEVICE_API<MsgIn, MsgOut>::TID() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    FLAMEGPU_DEVICE_API<MsgIn, MsgOut> *api = new FLAMEGPU_DEVICE_API<MsgIn, MsgOut>(thread_in_layer_offset, model_name_hash, streamId, MsgIn(agent_func_name_hash, messagename_inp_hash, messageList_size), MsgOut(agent_func_name_hash, messagename_outp_hash, messageList_size));

    // ! set namespace for agent name
    api->setAgentNameSpace(agent_func_name_hash);

    // printf("hello from wrapper %d %u\n",threadIdx.x,agentname_hash);

    // call the user specified device function
    {
        FLAME_GPU_AGENT_STATUS flag = AgentFunction()(api);
        // (scan flags will not be processed unless agent death has been requested in model definition)
        flamegpu_internal::CUDAScanCompaction::ds_agent_configs[streamId].scan_flag[FLAMEGPU_DEVICE_API<MsgIn, MsgOut>::TID()] = flag;
    }
    // do something with the return value to set a flag for deletion

    delete api;
}


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_
