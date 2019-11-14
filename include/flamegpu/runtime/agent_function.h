#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENT_FUNCTION_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENT_FUNCTION_H_

#include <cuda_runtime.h>

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param model_name_hash Required by DeviceEnvironment
 * @param agent_func_name_hash
 * @param messagename_inp_hash
 * @param func
 * @param popNo
 * @param messageList_size
 * @param thread_in_layer_offset Add this value to TID to calculate a thread-safe TID (TS_ID), used by ActorRandom for accessing curand array in a thread-safe manner
 */
__global__ void agent_function_wrapper(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    FLAMEGPU_AGENT_FUNCTION_POINTER func,
    int popNo,
    unsigned int messageList_size,
    const unsigned int thread_in_layer_offset);

#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENT_FUNCTION_H_
