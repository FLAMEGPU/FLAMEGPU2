#include <cuda_runtime.h>

#include "../flame_functions_api.h"

__global__ void agent_function_wrapper(const char* func_name, FLAMEGPU_AGENT_FUNCTION_POINTER func)
{
    //create a new device FLAME_GPU instance
    FLAMEGPU_API *api = new FLAMEGPU_API();

    //call the user specified device function
    FLAME_GPU_AGENT_STATUS flag = func(api);

    if (flag == 1){
    // delete the agent
    }


    //do something with the return value to set a flag for deletion
}
