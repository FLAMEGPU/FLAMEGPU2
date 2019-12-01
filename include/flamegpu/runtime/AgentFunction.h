#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "flamegpu/runtime/cuRVE/curve.h"
#include "flamegpu/runtime/flamegpu_device_api.h"

typedef void(AgentFunctionWrapper)(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    const int popNo,
    const unsigned int messageList_size,
    const unsigned int thread_in_layer_offset);  // Can't put __global__ in a typedef

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
 */
template<typename AgentFunction>
__global__ void agent_function_wrapper(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    const int popNo,
    const unsigned int messageList_size,
    const unsigned int thread_in_layer_offset) {
    // Must be terminated here, else AgentRandom has bounds issues inside FLAMEGPU_DEVICE_API constructor
    if (FLAMEGPU_DEVICE_API::TID() >= popNo)
        return;
    // create a new device FLAME_GPU instance
    FLAMEGPU_DEVICE_API *api = new FLAMEGPU_DEVICE_API(thread_in_layer_offset, model_name_hash);

    api->setMessageListSize(messageList_size);

    // ! set namespace for agent name
    api->setAgentNameSpace(agent_func_name_hash);

    // ! set namespace for input message name
    api->setMessageInpNameSpace(messagename_inp_hash);

    // ! set namespace for output message name
    api->setMessageOutpNameSpace(messagename_outp_hash);


    // printf("hello from wrapper %d %u\n",threadIdx.x,agentname_hash);

    // call the user specified device function
    {
        FLAME_GPU_AGENT_STATUS flag = AgentFunction()(api);
        if (flag == 1) {
            // delete the agent
            printf("Agent DEAD!\n");
        } else {
            // printf("Agent ALIVE!\n");
        }
    }
    // do something with the return value to set a flag for deletion

    delete api;
}


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_H_
