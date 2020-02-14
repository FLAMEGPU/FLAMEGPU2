#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONSHIM_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONSHIM_H_

#include "flamegpu/runtime/AgentFunction.h"

// TODO: Some example code of the handle class and an example function
// ! FLAMEGPU_API is a singleton class
template<typename MsgIn, typename MsgOut>
class FLAMEGPU_DEVICE_API;  // Forward declaration (class defined below)


/**
 * Macro for defining agent transition functions with the correct input. Must always be a device function to be called by CUDA.
 *
 * struct SomeAgentFunction {
 *     __device__ __forceinline__ FLAME_GPU_AGENT_STATUS operator()(FLAMEGPU_DEVICE_API *FLAMEGPU) const {
 *         // do something
 *         return 0;
 *     }
 * };
 *}
 */
#define FLAMEGPU_AGENT_FUNCTION(funcName, msg_in, msg_out)\
struct funcName ## _impl {\
    __device__ __forceinline__ FLAME_GPU_AGENT_STATUS operator()(FLAMEGPU_DEVICE_API<msg_in, msg_out> *FLAMEGPU) const;\
    static constexpr AgentFunctionWrapper *fnPtr() { return &agent_function_wrapper<funcName ## _impl, msg_in, msg_out>; }\
};\
funcName ## _impl funcName;\
__device__ __forceinline__ FLAME_GPU_AGENT_STATUS funcName ## _impl::operator()(FLAMEGPU_DEVICE_API<msg_in, msg_out> *FLAMEGPU) const

// Advanced macro for defining agent transition functions
#define FLAMEGPU_AGENT_FUNC __device__ __forceinline__

#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONSHIM_H_
