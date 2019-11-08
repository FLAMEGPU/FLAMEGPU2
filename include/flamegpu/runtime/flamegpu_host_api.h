#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_

#include <string>

class FLAMEGPU_HOST_API;
class FLAMEGPU_HOST_AGENT_API;

/**
 * @brief FLAMEGPU host function pointer definition
 *  this runs on the host as an init/step/exit or host layer function
 */
typedef void (*FLAMEGPU_HOST_FUNCTION_POINTER)(FLAMEGPU_HOST_API *api);

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
void funcName ## _impl(FLAMEGPU_HOST_API* FLAMEGPU); \
FLAMEGPU_HOST_FUNCTION_POINTER funcName = funcName ## _impl;\
void funcName ## _impl(FLAMEGPU_HOST_API* FLAMEGPU)

// FLAMEGPU function return type
enum FLAME_GPU_CONDITION_RESULT { CONTINUE, EXIT };
/**
 * @brief FLAMEGPU host function pointer definition
 *  this runs on the host as an init/step/exit or host layer function
 */
typedef FLAME_GPU_CONDITION_RESULT (*FLAMEGPU_HOST_CONDITION_POINTER)(FLAMEGPU_HOST_API *api);

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
FLAME_GPU_CONDITION_RESULT funcName ## _impl(FLAMEGPU_HOST_API* FLAMEGPU); \
FLAMEGPU_HOST_CONDITION_POINTER funcName = funcName ## _impl;\
FLAME_GPU_CONDITION_RESULT funcName ## _impl(FLAMEGPU_HOST_API* FLAMEGPU)
class CUDAAgentModel;
class FLAMEGPU_HOST_AGENT_API;
/**
 * @brief    A flame gpu api class for use by host functions only
 * This class should only be used by init/step/exit/exitcondition functions.
 */
class FLAMEGPU_HOST_API {
    friend class FLAMEGPU_HOST_AGENT_API;
 public:
    FLAMEGPU_HOST_API(CUDAAgentModel &_agentModel)
        :agentModel(_agentModel),
        d_cub_temp(nullptr),
        d_cub_temp_size(0),
        d_output_space(nullptr),
        d_output_space_size(0)
    {
        
    }
    /**
     * Returns methods that work on all agents of a certain type
     */
    FLAMEGPU_HOST_AGENT_API agent(const std::string &agent_name);
    /**
    * Returns methods that work on all agents of a certain type currently in a given state
    */
    FLAMEGPU_HOST_AGENT_API agent(const std::string &agent_name, const std::string &stateName);
    // const HostRandom;

private:
    CUDAAgentModel &agentModel;
    void *d_cub_temp;
    size_t d_cub_temp_size;
    void *d_output_space;
    size_t d_output_space_size;
};

#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_
