#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONSHIM_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONSHIM_H_

#include "flamegpu/runtime/AgentFunction.h"

template<typename MsgIn, typename MsgOut>
class FLAMEGPU_DEVICE_API;  // Forward declaration (class defined below)

/**
 * Macro for defining agent transition functions with the correct input. Must always be a device function to be called by CUDA.
 *
 * struct SomeAgentFunction {
 *    // User Implemented agent function behaviour
 *     __device__ __forceinline__ FLAME_GPU_AGENT_STATUS operator()(FLAMEGPU_DEVICE_API<msg_in, msg_out> *FLAMEGPU) const {
 *         // do something
 *         return 0;
 *     }
 *     // Returns a function pointer to the agent function wrapper for this function
 *     // Including it here, force instantiates the template, without requiring the user to specify the template args elsewhere
 *     static constexpr AgentFunctionWrapper *fnPtr() { return &agent_function_wrapper<funcName ## _impl, msg_in, msg_out>; }
 *     // These are used to validate that the bound message description matches the specified type
 *     static constexpr std::type_index inType() { return std::type_index(typeid(msg_in)); }
 *     static constexpr std::type_index outType() { return std::type_index(typeid(msg_out)); }
 * };
 *}
 */
#define FLAMEGPU_AGENT_FUNCTION(funcName, msg_in, msg_out)\
struct funcName ## _impl {\
    __device__ __forceinline__ FLAME_GPU_AGENT_STATUS operator()(FLAMEGPU_DEVICE_API<msg_in, msg_out> *FLAMEGPU) const;\
    static constexpr AgentFunctionWrapper *fnPtr() { return &agent_function_wrapper<funcName ## _impl, msg_in, msg_out>; }\
    static std::type_index inType() { return std::type_index(typeid(msg_in)); }\
    static std::type_index outType() { return std::type_index(typeid(msg_out)); }\
};\
funcName ## _impl funcName;\
__device__ __forceinline__ FLAME_GPU_AGENT_STATUS funcName ## _impl::operator()(FLAMEGPU_DEVICE_API<msg_in, msg_out> *FLAMEGPU) const


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONSHIM_H_
