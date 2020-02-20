#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_SHIM_H_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_SHIM_H_

#include "flamegpu/runtime/AgentFunctionCondition.h"

class FLAMEGPU_READ_ONLY_DEVICE_API;

/**
 * Macro for defining agent transition functions conditions with the correct input. Must always be a device function to be called by CUDA.
 *
 * struct SomeAgentFunctionCondition {
 *    // User Implemented agent function condition behaviour
 *     __device__ __forceinline__ bool operator()(FLAMEGPU_READ_ONLY_DEVICE_API *FLAMEGPU) const {
 *         // do something
 *         return something ? true : false;
 *     }
 *     // Returns a function pointer to the agent function condition wrapper for this function
 *     // Including it here, force instantiates the template, without requiring the user to specify the template args elsewhere
 *     static constexpr AgentFunctionConditionWrapper *fnPtr() { return &agent_function_condition_wrapper<SomeAgentFunctionCondition>; }
 * };
 *}
 */
#define FLAMEGPU_AGENT_FUNCTION_CONDITION(funcName)\
struct funcName ## _cdn_impl {\
    __device__ __forceinline__ bool operator()(FLAMEGPU_READ_ONLY_DEVICE_API *FLAMEGPU) const;\
    static constexpr AgentFunctionConditionWrapper *fnPtr() { return &agent_function_condition_wrapper<funcName ## _cdn_impl>; }\
};\
funcName ## _cdn_impl funcName;\
__device__ __forceinline__ bool funcName ## _cdn_impl::operator()(FLAMEGPU_READ_ONLY_DEVICE_API *FLAMEGPU) const


#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_SHIM_H_
