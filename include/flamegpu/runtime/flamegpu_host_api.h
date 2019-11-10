#ifndef INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_
#define INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_

#include <cuda_runtime_api.h>
#include <cassert>
#include <string>
#include <utility>
#include <functional>
#include <unordered_map>

#include "flamegpu/gpu/CUDAErrorChecking.h"

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
    /**
     * Initailises pointers to 0
     * Stores reference of CUDAAgentModel
     */
     explicit FLAMEGPU_HOST_API(CUDAAgentModel &_agentModel);
    /**
     * Frees held device memory
     */
     ~FLAMEGPU_HOST_API();
    /**
     * TODO: Returns methods that work on all agents of a certain type
     */
    // FLAMEGPU_HOST_AGENT_API agent(const std::string &agent_name);
    /**
    * Returns methods that work on all agents of a certain type currently in a given state
    */
    FLAMEGPU_HOST_AGENT_API agent(const std::string &agent_name, const std::string &stateName = "default");
    // const HostRandom;

 private:
    /**
     * Used internally for tracking what CUB has already calculated temp memory for
     * Pete claims this calc launches kernel, so best to avoid where possible
     */
    enum CUB_Operation {
        MIN,
        MAX,
        SUM,
        CUSTOM_REDUCE,
        HISTOGRAM_EVEN
    };
    // Can't put type_info in map, deleted constructors, so we use it's hash code
    // Histogram always has type int, so we use number of bins instead
    typedef std::pair<CUB_Operation, size_t> CUB_Config;
    /**
     * Need to define a hash_fn for tuple
     */
    struct key_hash : public std::unary_function<CUB_Config, std::size_t> {
        std::size_t operator()(const CUB_Config& k) const {
            return static_cast<size_t>(std::get<0>(k)) ^ std::get<1>(k);
        }
    };
    std::unordered_map<CUB_Config, unsigned int, key_hash> cub_largestAllocatedOp;
    bool tempStorageRequiresResize(const CUB_Config &cc, const unsigned int &items);
    void resizeTempStorage(const CUB_Config &cc, const unsigned int &items, const size_t &newSize);
    template<typename T>
    void resizeOutputSpace(const unsigned int &items = 1);
    CUDAAgentModel &agentModel;
    void *d_cub_temp;
    size_t d_cub_temp_size;
    void *d_output_space;
    size_t d_output_space_size;
};

template<typename T>
void FLAMEGPU_HOST_API::resizeOutputSpace(const unsigned int &items) {
    if (sizeof(T) * items > d_output_space_size) {
        if (d_output_space_size) {
            gpuErrchk(cudaFree(d_output_space));
        }
        gpuErrchk(cudaMalloc(&d_output_space, sizeof(T) * items));
        d_output_space_size = sizeof(T) * items;
    }
}

#endif  // INCLUDE_FLAMEGPU_RUNTIME_FLAMEGPU_HOST_API_H_
