#ifndef AGENT_FUNCTION_H_
#define AGENT_FUNCTION_H_


#include <cuda_runtime.h>
//#include "flame_functions_api.h"

/**
 * Wrapper function for launching agent functions
 * Initialises FLAMEGPU_API instance
 * @param agent_func_name_hash
 * @param messagename_inp_hash
 * @param func
 * @param popNo
 * @param messageList_size
 * @param thread_in_layer_offset Add this value to TID to calculate a thread-safe TID (TS_ID), used by ActorRandom for accessing curand array in a thread-safe manner
 */
__global__ void agent_function_wrapper(CurveNamespaceHash agent_func_name_hash, CurveNamespaceHash messagename_inp_hash, CurveNamespaceHash messagename_outp_hash, FLAMEGPU_AGENT_FUNCTION_POINTER func, int popNo, unsigned int messageList_size, const unsigned int thread_in_layer_offset);
#endif
