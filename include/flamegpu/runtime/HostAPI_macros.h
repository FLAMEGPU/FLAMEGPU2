#ifndef INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_MACROS_H_
#define INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_MACROS_H_

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
#define FLAMEGPU_INIT_FUNCTION(funcName) FLAMEGPU_HOST_FUNCTION(funcName)
#define FLAMEGPU_STEP_FUNCTION(funcName) FLAMEGPU_HOST_FUNCTION(funcName)
#define FLAMEGPU_EXIT_FUNCTION(funcName) FLAMEGPU_HOST_FUNCTION(funcName)
/**
 * Macro for defining host functions with the correct input. 
 * Ugly, but has same usage as device functions
 */
#define FLAMEGPU_HOST_FUNCTION(funcName) \
void funcName ## _impl(HostAPI* FLAMEGPU); \
FLAMEGPU_HOST_FUNCTION_POINTER funcName = funcName ## _impl;\
void funcName ## _impl(HostAPI* FLAMEGPU)

/**
 * Return type for FLAMEGPU conditions
 */
enum FLAME_GPU_CONDITION_RESULT { CONTINUE, EXIT };
/**
 * @brief FLAMEGPU host function pointer definition
 *  this runs on the host as an init/step/exit or host layer function
 */
typedef FLAME_GPU_CONDITION_RESULT (*FLAMEGPU_HOST_CONDITION_POINTER)(HostAPI *api);

/**
 * Only use of condition at current is exit fn
 * @todo Use these for global conditions?
 */
typedef FLAMEGPU_HOST_CONDITION_POINTER FLAMEGPU_EXIT_CONDITION_POINTER;
#define FLAMEGPU_EXIT_CONDITION(funcName) FLAMEGPU_HOST_CONDITION(funcName)

/**
 * Macro for defining host functions with the correct input.
 * Ugly, but has same usage as device functions
 */
#define FLAMEGPU_HOST_CONDITION(funcName) \
FLAME_GPU_CONDITION_RESULT funcName ## _impl(HostAPI* FLAMEGPU); \
FLAMEGPU_HOST_CONDITION_POINTER funcName = funcName ## _impl;\
FLAME_GPU_CONDITION_RESULT funcName ## _impl(HostAPI* FLAMEGPU)

#endif  // INCLUDE_FLAMEGPU_RUNTIME_HOSTAPI_MACROS_H_
