#include <cuda_runtime.h>

#include "flame_functions_api.h"


__global__ void agent_function_wrapper(CurveNamespaceHash agentname_hash, FLAMEGPU_AGENT_FUNCTION_POINTER func, int popNo)
{

    //create a new device FLAME_GPU instance
    FLAMEGPU_API *api = new FLAMEGPU_API();
	api->setNameSpace(agentname_hash);



    //printf("hello from wrapper %d %u\n",threadIdx.x,agentname_hash);
	//set namespace for agent name



    //call the user specified device function
    int tid = (blockDim.x * blockIdx.x) + threadIdx.x;//threadIdx.x + blockIdx.x * gridDim.x;
    if(tid < popNo)
    {
        FLAME_GPU_AGENT_STATUS flag = func(api);
        if (flag == 1)
        {
            // delete the agent
            printf("Agent DEAD!\n");
        }
        else
        {
            //printf("Agent ALIVE!\n");
        }

    }

    //do something with the return value to set a flag for deletion
}
