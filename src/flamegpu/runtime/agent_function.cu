#include <cuda_runtime.h>

#include "flamegpu/runtime/flamegpu_device_api.h"

__global__ void agent_function_wrapper(
    Curve::NamespaceHash model_name_hash,
    Curve::NamespaceHash agent_func_name_hash,
    Curve::NamespaceHash messagename_inp_hash,
    Curve::NamespaceHash messagename_outp_hash,
    FLAMEGPU_AGENT_FUNCTION_POINTER func,
    int popNo,
    unsigned int messageList_size,
    const unsigned int thread_in_layer_offset
    ) {
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
        FLAME_GPU_AGENT_STATUS flag = func(api);
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
