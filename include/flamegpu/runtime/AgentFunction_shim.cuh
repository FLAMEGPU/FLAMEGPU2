#ifndef INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_SHIM_CUH_
#define INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_SHIM_CUH_

namespace flamegpu {


template<typename MessageIn, typename MessageOut>
class DeviceAPI;

/**
 * Macro for defining agent transition functions with the correct input. Must always be a device function to be called by CUDA.
 *
 * struct SomeAgentFunction {
 *    // User Implemented agent function behaviour
 *     __device__ __forceinline__ AGENT_STATUS operator()(DeviceAPI<message_in, message_out> *FLAMEGPU) const {
 *         // do something
 *         return 0;
 *     }
 *     // Returns a function pointer to the agent function wrapper for this function
 *     // Including it here, force instantiates the template, without requiring the user to specify the template args elsewhere
 *     static constexpr AgentFunctionWrapper *fnPtr() { return &agent_function_wrapper<SomeAgentFunction, message_in, message_out>; }
 *     // These are used to validate that the bound message description matches the specified type
 *     static constexpr std::type_index inType() { return std::type_index(typeid(message_in)); }
 *     static constexpr std::type_index outType() { return std::type_index(typeid(message_out)); }
 * };
 *}
 */
#ifndef __CUDACC_RTC__
#define FLAMEGPU_AGENT_FUNCTION(funcName, message_in, message_out)\
struct funcName ## _impl {\
    __device__ __forceinline__ flamegpu::AGENT_STATUS operator()(flamegpu::DeviceAPI<message_in, message_out> *FLAMEGPU) const;\
    static constexpr flamegpu::AgentFunctionWrapper *fnPtr() { return &flamegpu::agent_function_wrapper<funcName ## _impl, message_in, message_out>; }\
    static std::type_index inType() { return std::type_index(typeid(message_in)); }\
    static std::type_index outType() { return std::type_index(typeid(message_out)); }\
};\
funcName ## _impl funcName;\
__device__ __forceinline__ flamegpu::AGENT_STATUS funcName ## _impl::operator()(flamegpu::DeviceAPI<message_in, message_out> *FLAMEGPU) const
#else
#define FLAMEGPU_AGENT_FUNCTION(funcName, message_in, message_out)\
struct funcName ## _impl {\
    __device__ __forceinline__ flamegpu::AGENT_STATUS operator()(flamegpu::DeviceAPI<message_in, message_out> *FLAMEGPU) const;\
    static constexpr flamegpu::AgentFunctionWrapper *fnPtr() { return &flamegpu::agent_function_wrapper<funcName ## _impl, message_in, message_out>; }\
}; \
funcName ## _impl funcName; \
__device__ __forceinline__ flamegpu::AGENT_STATUS funcName ## _impl::operator()(flamegpu::DeviceAPI<message_in, message_out> *FLAMEGPU) const
#endif

/**
 * Macro so users can define their own device functions
 */
#define FLAMEGPU_DEVICE_FUNCTION __device__ __forceinline__

/**
 * Macro so users can define their own host functions, to support host device functions.
 */
#define FLAMEGPU_HOST_DEVICE_FUNCTION __host__ FLAMEGPU_DEVICE_FUNCTION

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_AGENTFUNCTION_SHIM_CUH_
