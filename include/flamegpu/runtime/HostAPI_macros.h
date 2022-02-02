#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_MACROS_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_MACROS_H_

namespace flamegpu {

class HostAPI;

/**
 * @brief FLAMEGPU host function pointer definition
 *  this runs on the host as an init/step/exit or host layer function
 */
typedef void (*FLAMEGPU_HOST_FUNCTION_POINTER)(HostAPI *api);

/**
 * They all share the same definition
 */
typedef FLAMEGPU_HOST_FUNCTION_POINTER FLAMEGPU_INIT_FUNCTION_POINTER;
typedef FLAMEGPU_HOST_FUNCTION_POINTER FLAMEGPU_STEP_FUNCTION_POINTER;
typedef FLAMEGPU_HOST_FUNCTION_POINTER FLAMEGPU_EXIT_FUNCTION_POINTER;
/**
 * Macro for defining host (init) functions with the correct input. 
 * @see FLAMEGPU_HOST_FUNCTION
 */
#define FLAMEGPU_INIT_FUNCTION(funcName) FLAMEGPU_HOST_FUNCTION(funcName)
/**
 * Macro for defining host (step) functions with the correct input.
 * @see FLAMEGPU_HOST_FUNCTION
 */
#define FLAMEGPU_STEP_FUNCTION(funcName) FLAMEGPU_HOST_FUNCTION(funcName)
/**
 * Macro for defining host (exit) functions with the correct input.
 * @see FLAMEGPU_HOST_FUNCTION
 */
#define FLAMEGPU_EXIT_FUNCTION(funcName) FLAMEGPU_HOST_FUNCTION(funcName)
/**
 * Macro for defining host functions with the correct input. 
 * Ugly, but has same usage as device functions
 *
 * Saves users from manually defining host functions, e.g.:
 * @code{.cpp}
 * // User Implemented host condition behaviour
 * void SomeHostFunction_impl(flamegpu::HostAPI* FLAMEGPU) {
 *     // do something
 * }
 * flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER SomeHostFunction = SomeHostFunction_impl;
 * @endcode
 */
#define FLAMEGPU_HOST_FUNCTION(funcName) \
void funcName ## _impl(flamegpu::HostAPI* FLAMEGPU); \
flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER funcName = funcName ## _impl;\
void funcName ## _impl(flamegpu::HostAPI* FLAMEGPU)

/**
 * Return type for FLAMEGPU conditions
 */
enum CONDITION_RESULT { CONTINUE, EXIT };
/**
 * @brief FLAMEGPU host function pointer definition
 *  this runs on the host as an init/step/exit or host layer function
 */
typedef CONDITION_RESULT (*FLAMEGPU_HOST_CONDITION_POINTER)(HostAPI *api);

/**
 * Only use of condition at current is exit fn
 * @todo Use these for global conditions?
 */
typedef FLAMEGPU_HOST_CONDITION_POINTER FLAMEGPU_EXIT_CONDITION_POINTER;
/**
 * Macro for defining host (exit) conditions with the correct input/return.
 * @see FLAMEGPU_HOST_CONDITION
 */
#define FLAMEGPU_EXIT_CONDITION(funcName) FLAMEGPU_HOST_CONDITION(funcName)

/**
 * Macro for defining host functions with the correct input.
 * Ugly, but has same usage as device functions
 *
 * Saves users from manually defining host conditions, e.g.:
 * @code{.cpp}
 * // User Implemented host condition behaviour
 * flamegpu::CONDITION_RESULT SomeHostCondition_impl(flamegpu::HostAPI* FLAMEGPU) {
 *     // do something
 *     return something ? flamegpu::CONTINUE : flamegpu::EXIT;
 * }
 * flamegpu::FLAMEGPU_HOST_CONDITION_POINTER SomeHostCondition = SomeHostCondition_impl;
 * @endcode
 */
#define FLAMEGPU_HOST_CONDITION(funcName) \
flamegpu::CONDITION_RESULT funcName ## _impl(flamegpu::HostAPI* FLAMEGPU); \
flamegpu::FLAMEGPU_HOST_CONDITION_POINTER funcName = funcName ## _impl;\
flamegpu::CONDITION_RESULT funcName ## _impl(flamegpu::HostAPI* FLAMEGPU)

}  // namespace flamegpu

#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_MACROS_H_
