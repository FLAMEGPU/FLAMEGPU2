#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/runtime/flamegpu_device_api.h"
#include "flamegpu/gpu/CUDAScanCompaction.h"

// ! FLAMEGPU function return type
enum FLAME_GPU_AGENT_STATUS { ALIVE = 1, DEAD = 0 };

typedef void(AgentFunctionWrapper)(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    Curve::NamespaceHash agent_output_hash,
    const int popNo,
    const void *messagelist_metadata,
    const unsigned int thread_in_layer_offset,
    const unsigned int streamId);  // Can't put __global__ in a typedef

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param model_name_hash CURVE hash of the model's name
 * @param agent_func_name_hash CURVE hash of the agent + function's names
 * @param messagename_inp_hash CURVE hash of the input message's name
 * @param messagename_outp_hash CURVE hash of the output message's name
 * @param agent_output_hash CURVE hash of "_agent_birth" or 0 if agent birth not present
 * @param popNo Total number of agents exeucting the function (number of threads launched)
 * @param messagelist_metadata Pointer to the MsgIn metadata struct, it is interpreted by MsgIn
 * @param thread_in_layer_offset Add this value to TID to calculate a thread-safe TID (TS_ID), used by ActorRandom for accessing curand array in a thread-safe manner
 * @tparam AgentFunction The modeller defined agent function (defined as FLAMEGPU_AGENT_FUNCTION in model code)
 * @tparam MsgIn Message handler for input messages (e.g. MsgNone, MsgBruteForce, MsgSpatial3D)
 * @tparam MsgOut Message handler for output messages (e.g. MsgNone, MsgBruteForce, MsgSpatial3D)
 */
template<typename AgentFunction, typename MsgIn, typename MsgOut>
__global__ void agent_function_wrapper(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    Curve::NamespaceHash agent_output_hash,
    const int popNo,
    const void *messagelist_metadata,
    const unsigned int thread_in_layer_offset,
    const unsigned int streamId) {
    // Must be terminated here, else AgentRandom has bounds issues inside FLAMEGPU_DEVICE_API constructor
    if (FLAMEGPU_DEVICE_API<MsgIn, MsgOut>::TID() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    FLAMEGPU_DEVICE_API<MsgIn, MsgOut> *api = new FLAMEGPU_DEVICE_API<MsgIn, MsgOut>(
        thread_in_layer_offset,
        model_name_hash,
        agent_func_name_hash,
        agent_output_hash,
        streamId,
        MsgIn::In(agent_func_name_hash, messagename_inp_hash, messagelist_metadata),
        MsgOut::Out(agent_func_name_hash, messagename_outp_hash, streamId));

    // call the user specified device function
    {
        FLAME_GPU_AGENT_STATUS flag = AgentFunction()(api);
        // (scan flags will not be processed unless agent death has been requested in model definition)
        flamegpu_internal::CUDAScanCompaction::ds_configs[flamegpu_internal::CUDAScanCompaction::AGENT_DEATH][streamId].scan_flag[FLAMEGPU_DEVICE_API<MsgIn, MsgOut>::TID()] = flag;
    }
    // do something with the return value to set a flag for deletion

    delete api;
}


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_
