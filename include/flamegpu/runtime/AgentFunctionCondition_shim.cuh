#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_SHIM_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_SHIM_CUH_


class ReadOnlyDeviceAPI;

namespace flamegpu {

/**
 * Macro for defining agent transition functions conditions. Must always be a device function to be called by CUDA.
 *
 * Saves users from manually defining agent function conditions, e.g.:
 * @code{.cpp}
 * struct SomeAgentFunctionCondition {
 *    // User Implemented agent function condition behaviour
 *     __device__ __forceinline__ bool operator()(ReadOnlyDeviceAPI *FLAMEGPU) const {
 *         // do something
 *         return something ? true : false;
 *     }
 *     // Returns a function pointer to the agent function condition wrapper for this function
 *     // Including it here, force instantiates the template, without requiring the user to specify the template args elsewhere
 *     static constexpr AgentFunctionConditionWrapper *fnPtr() { return &agent_function_condition_wrapper<SomeAgentFunctionCondition>; }
 * };
 * SomeAgentFunctionCondition_cdn_impl SomeAgentFunctionCondition;
 * @endcode
 */

#define FLAMEGPU_AGENT_FUNCTION_CONDITION(funcName)\
struct funcName ## _cdn_impl {\
    __device__ __forceinline__ bool operator()(flamegpu::ReadOnlyDeviceAPI *FLAMEGPU) const;\
    static constexpr flamegpu::AgentFunctionConditionWrapper *fnPtr() { return &flamegpu::agent_function_condition_wrapper<funcName ## _cdn_impl>; }\
};\
funcName ## _cdn_impl funcName;\
__device__ __forceinline__ bool funcName ## _cdn_impl::operator()(flamegpu::ReadOnlyDeviceAPI *FLAMEGPU) const

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTIONCONDITION_SHIM_CUH_
