#ifndef AGENT_FUNCTION_H_
#define AGENT_FUNCTION_H_


#include <cuda_runtime.h>
//#include "flame_functions_api.h"


__global__ void agent_function_wrapper(CurveNamespaceHash agentname_hash, FLAMEGPU_AGENT_FUNCTION_POINTER func, int popNo);
#endif
